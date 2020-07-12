# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import misc.utils as utils

from .AttModel import pack_wrapper, AttModel
from .BertAoAModule import SublayerConnection, PositionwiseFeedForward, clones, GramSchmidt

"""opt INFO
* opt.nhead : 2
* opt.nlayer : 6
* opt.input_json : data/cocotalk.json
* opt.input_fc_dir : data/stylized_cocotalk_fc
* opt.input_att_dir : data/stylized_cocotalk_att
* opt.input_box_dir : data/cocotalk_box
* opt.input_label_h5 : data/cocotalk_label.h5
* opt.start_from : log/log_paper_head
* opt.cached_tokens : coco-train-idxs
* opt.caption_model : aoa
* opt.rnn_size : 1024
* opt.num_layers : 2
* opt.rnn_type : lstm
* opt.input_encoding_size : 1024
* opt.att_hid_size : 512
* opt.fc_feat_size : 2048
* opt.att_feat_size : 2048
* opt.logit_layers : 1
* opt.use_bn : 0
* opt.mean_feats : 1
* opt.refine : 1
* opt.refine_aoa : 1
* opt.use_ff : 0
* opt.dropout_aoa : 0.3
* opt.ctx_drop : 1
* opt.decoder_type : BertAoA
* opt.use_multi_head : 2
* opt.num_heads : 8
* opt.multi_head_scale : 1
* opt.use_warmup : 0
* opt.acc_steps : 1
* opt.norm_att_feat : 0
* opt.use_box : 0
* opt.norm_box_feat : 0
* opt.max_epochs : 1
* opt.batch_size : 10
* opt.grad_clip : 0.1
* opt.drop_prob_lm : 0.5
* opt.self_critical_after : -1
* opt.seq_per_img : 5
* opt.beam_size : 1
* opt.max_length : 20
* opt.length_penalty :
* opt.block_trigrams : 0
* opt.remove_bad_endings : 0
* opt.optim : adam
* opt.learning_rate : 0.0002
* opt.learning_rate_decay_start : 0
* opt.learning_rate_decay_every : 3
* opt.learning_rate_decay_rate : 0.8
* opt.optim_alpha : 0.9
* opt.optim_beta : 0.999
* opt.optim_epsilon : 1e-08
* opt.weight_decay : 0
* opt.label_smoothing : 0.2
* opt.noamopt : False
* opt.noamopt_warmup : 2000
* opt.noamopt_factor : 1
* opt.reduce_on_plateau : False
* opt.scheduled_sampling_start : 0
* opt.scheduled_sampling_increase_every : 5
* opt.scheduled_sampling_increase_prob : 0.05
* opt.scheduled_sampling_max_prob : 0.5
* opt.val_images_use : -1
* opt.save_checkpoint_every : 6000
* opt.save_history_ckpt : 0
* opt.checkpoint_path : log/log_paper_head
* opt.language_eval : 1
* opt.losses_log_every : 25
* opt.load_best_score : 1
* opt.id : paper_head
* opt.train_only : 0
* opt.cider_reward_weight : 1
* opt.bleu_reward_weight : 0
"""



class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1+self.use_ff)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_mask=mask)[0])
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x

class AoA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = nn.MultiheadAttention(embed_dim=opt.rnn_size, num_heads=opt.num_heads)
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 6)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = embed_dim // num_heads
        self.h = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from embed_dim => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        """
        :param d_model: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        self.attention = MultiHeadedAttention(embed_dim=d_model, num_heads=nhead)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=dim_feedforward, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, src_mask, context=None, **kwargs):
        if context != None:
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, context, context, mask=src_mask)[0])
        else:
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=src_mask)[0])
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, context=None, mask=None, src_key_padding_mask=None):
        output = src
        if context!=None:
            for idx, mod in enumerate(self.layers):
                if idx == 0 : output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, context=context)
                else        : output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                    
        else:
            for mod in self.layers:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None : output = self.norm(output)

        return output


    
class BertAoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(BertAoA_Decoder_Core, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=opt.rnn_size, nhead=opt.nhead, dim_feedforward=opt.rnn_size * 4, dropout=opt.drop_prob_lm)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=opt.nlayer)
        self.out_drop = nn.Dropout(opt.drop_prob_lm)
        self.att2ctx = nn.Sequential(nn.Linear(2 * opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        if opt.gs_type:
            self.gs_type = opt.gs_type
            self.gramschmidt = GramSchmidt()
        else:
            self.gs_type = None

    def forward(self, xt, mean_feats, att_feats, p_att_feats, att_masks=None):
        # state[0][1] is the context vector at the last step
        """
	[debug] xt : torch.Size([50, 18, 1024])
	[debug] mean_feats : torch.Size([50, 1024])
        [debug] state[0][1] : torch.Size([50, 1024])
	[debug] x : torch.Size([50, 2048])
	[debug] h[0] : torch.Size([50, 1024])
	[debug] h[1] : torch.Size([50, 1024])
        [debug] h_att : torch.Size([50, 1024])
        [debug] c_att : torch.Size([50, 1024])
        [debug] att_feats : torch.Size([50, 196, 1024])
        [debug] p_att_feats : torch.Size([50, 196, 1024])
        [debug] att : torch.Size([50, 1, 1024])
        [debug] ctx_input : torch.Size([50, 2048])

        """
        """
        [debug] xt : torch.Size([50, 18, 1024])
        [debug] p_att_feats : torch.Size([50, 196, 1024])   
        """
        if self.gs_type == 'first': xt = self.gramschmidt(xt)
        x = self.transformer_encoder(xt, context=p_att_feats)
        if self.gs_type == 'last': x = self.gramschmidt(x)
        #x = self.att2ctx(x)

        #x = x + h_att
        #x = self.out_drop(x)
        return x



class BertAoAModel(AttModel):
    def __init__(self, opt):
        super(BertAoAModel, self).__init__(opt)
        self.num_layers = 2
        # mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)
        self.ctx2att = nn.Linear(opt.rnn_size, opt.rnn_size)
        #self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
        if self.use_mean_feats:
            del self.fc_embed
        if opt.refine:
            self.refiner = AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x,y : x
        self.core = BertAoA_Decoder_Core(opt)


    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed att feats
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.refiner(att_feats, att_masks)

        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = torch.mean(att_feats, dim=1)
            else:
                mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks
