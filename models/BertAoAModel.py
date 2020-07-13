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
    def __init__(self, d_model, nhead, dim_feedforward, dropout, opt=None):
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
        if opt.gs_type == 'inner':
            self.gs_type = opt.gs_type
            self.gramschmidt = GramSchmidt()
        else:
            self.gs_type = None

    def forward(self, x, src_mask, context=None, **kwargs):
        if context != None:
            if self.gs_type == 'inner':
                context = self.gramschmidt(context)
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, context, context, mask=src_mask)[0])
        else:
            if self.gs_type == 'inner': 
                context = self.gramschmidt(x)
                x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, context, context, mask=src_mask)[0])
            else:
                x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=src_mask)[0])
        
        x = self.output_sublayer(x, self.feed_forward)

        return self.dropout(x)


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, opt=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.add_aoa = opt.add_aoa

        # gram_schmidt process
        if opt.gs_type == 'first' or 'last':
            self.gs_type = opt.gs_type
            self.gramschmidt = GramSchmidt()
        else:
            self.gs_type = None

        # attention on attention
        if self.add_aoa:
            self.att2ctx = nn.Sequential(nn.Linear(opt.rnn_size, 2 * opt.rnn_size), nn.GLU())

    def forward(self, src, context=None, mask=None, src_key_padding_mask=None):
        memory = []
        memory.append(src.clone())
        
        output = src

        for idx, mod in enumerate(self.layers):
            if self.gs_type == 'first': 
                if idx == 0:
                    context = self.gramschmidt(context)
            if self.gs_type == 'last':
                if idx == self.num_layers-1 :
                    context = self.gramschmidt(context)
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, context=context)
            memory.append(output.clone())
        
        if self.add_aoa:output = self.att2ctx(output)
        if self.norm is not None : output = self.norm(output)
        
        return output, memory


    
class BertAoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(BertAoA_Decoder_Core, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=opt.rnn_size, nhead=opt.nhead, dim_feedforward=opt.rnn_size * 16, dropout=opt.drop_prob_lm, opt=opt)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=opt.nlayer, opt=opt)
        self.out_drop = nn.Dropout(opt.drop_prob_lm)
        
        
    def forward(self, xt, mean_feats, att_feats, p_att_feats, att_masks=None):
        x, state = self.transformer_encoder(xt, context=p_att_feats)
        return x, state



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
