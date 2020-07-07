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

from .AttModel import pack_wrapper, AttModel, Attention
from .TransformerModel import LayerNorm, attention, TransformerAttention, clones, SublayerConnection, PositionwiseFeedForward



class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3, gsp=1):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x:x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer =  nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x:x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x:x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

    

        # Do all the linear projections in batch from d_model => h x d_k 
        if self.project_k_v == 0:
            query_ =  self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x

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
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x

class AoA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, scale=opt.multi_head_scale, do_aoa=opt.refine_aoa, norm_q=0, dropout_aoa=getattr(opt, 'dropout_aoa', 0.3), gsp=opt.gsp)
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class AoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'AoA')
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)
        self.gsp = opt.gsp

        if self.decoder_type == 'AoA':
            # AoA layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            # LSTM layer
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            # Base linear layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size), nn.ReLU())

        # if opt.use_multi_head == 1: # TODO, not implemented for now           
        #     self.attention = MultiHeadedAddAttention(opt.num_heads, opt.d_model, scale=opt.multi_head_scale)
        if opt.use_multi_head == 2:            
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0, scale=opt.multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1, gsp=opt.gsp)
        else:            
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)        
        else:
            self.ctx_drop = lambda x :x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # state[0][1] is the context vector at the last step
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))

        if self.use_multi_head == 2:
            att = self.attention(h_att, p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model), p_att_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            # save the context vector to state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state



# 2-dimensional
class GramSchmidt02(nn.Module):
    def projection(self, x, y):
        z1 = torch.einsum('l,kl->k', [x,y])
        z2 = torch.einsum('kl,kl->k', [y,y])
        z = z1.div(z2)
        return torch.einsum('k,kl->l', [z,y])

    def forward(self, x):
        for i in range(1, x.size(-2)):
            x[i,:].sub_(self.projection(x[i,:], x[0:i,:]).detach())
        z = torch.einsum('kl,kl->k', [x,x])
        z = torch.div(torch.tensor(1.), torch.sqrt(z))
        x = torch.einsum('k,kl->kl', [z,x])
        #x = x / x.norm(dim=2, keepdim=True).detach()
        return x


# 3-dimensional
class GramSchmidt03(nn.Module):
    def projection(self, x, y):
        z1 = torch.einsum('il,ikl->ik', [x,y])
        z2 = torch.einsum('ikl,ikl->ik', [y,y])
        z = z1.div(z2)
        return torch.einsum('ik,ikl->il', [z,y])

    def forward(self, x):
        for i in range(1, x.size(-2)):
            x[:,i,:].sub_(self.projection(x[:,i,:], x[:,0:i,:]).detach())
        z = torch.einsum('ikl,ikl->ik', [x,x])
        z = torch.div(torch.tensor(1.), torch.sqrt(z))
        x = torch.einsum('ik,ikl->ikl', [z,x])
        return x


# 4-dimensional
class GramSchmidt04(nn.Module):
    def projection(self, x, y):
        z1 = torch.einsum('ijl,ijkl->ijk', [x,y])
        z2 = torch.einsum('ijkl,ijkl->ijk', [y,y])
        z = z1.div(z2)
        return torch.einsum('ijk,ijkl->ijl', [z,y])

    def forward(self, x):
        for i in range(1, x.size(-2)):
            x[:,:,i,:].sub_(self.projection(x[:,:,i,:], x[:,:,0:i,:]).detach())
        z = torch.einsum('ijkl,ijkl->ijk', [x,x])
        z = torch.div(torch.tensor(1.), torch.sqrt(z))
        x = torch.einsum('ijk,ijkl->ijkl', [z,x])
        return x


class GramSchmidt(nn.Module):
    def __init__(self):
        super(GramSchmidt, self).__init__()
        self.gramschmidt02 = GramSchmidt02()
        self.gramschmidt03 = GramSchmidt03()
        self.gramschmidt04 = GramSchmidt04()

    def forward(self, x):
        dim = len(x.size());                             print(f'x : {x.size()}')
        x = getattr(self, 'gramschmidt0'+str(dim))(x);   print(f'x : {x.size()}')
        
        # validation
        for i in range(x.size(-2)):
            if dim == 2   : print((x[0]*x[i]).sum())
            elif dim == 3 : print((x[0][0]*x[0][i]).sum())
            elif dim == 4 : print((x[0][0][0]*x[0][0][i]).sum())

        return x



class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class GramSchmidt(nn.Module):
    def projection(self, x, y):
        z1 = torch.einsum('il,ikl->ik', [x,y])
        z2 = torch.einsum('ikl,ikl->ik', [y,y])
        z2 = torch.div(torch.tensor(1.), z2)
        z = torch.einsum('ik,ik->ik', [z1,z2])
        return torch.einsum('ik,ikl->il', [z,y])

    def forward(self, x):
        for i in range(1, x.size(1)):
            x[:,i,:].sub_(self.projection(x[:,i,:], x[:,0:i,:]).detach())
        z = torch.einsum('ikl,ikl->ik', [x,x])
        z = torch.div(torch.tensor(1.), torch.sqrt(z))
        x = torch.einsum('ik,ikl->ikl', [z,x])
        #x = x / x.norm(dim=2, keepdim=True).detach()
        return x
    



class BertAoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(BertAoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale

        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'BertAoA')

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)
        self.gsp = opt.gsp
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=opt.nhead)
        
        self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        """
        if self.decoder_type == 'LSTM':
            # LSTM layer
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        elif self.decoder_type == 'BertAoA':
            self.att2ctx = nn.TransformerEncoder(self.encoder_layer, num_layers=opt.nlayer)
        
        else:
            # Base linear layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size), nn.ReLU())
        """

        # if opt.use_multi_head == 1: # TODO, not implemented for now           
        #     self.attention = MultiHeadedAddAttention(opt.num_heads, opt.d_model, scale=opt.multi_head_scale)
        if opt.use_multi_head == 2:            
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0, scale=opt.multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1, gsp=opt.gsp)
        else:            
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)        
        else:
            self.ctx_drop = lambda x :x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        
        # state[0][1] is the context vector at the last step
        x = torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1)
        h = (state[0][0], state[1][0])
        
        h_att, c_att = self.att_lstm(x, h)
        
        """
        if self.use_multi_head == 2:
            att = self.attention(h_att, p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model), p_att_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        """

        ctx_input = torch.cat([xt, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
            """
            elif self.decoder_type == 'BertAoA':
                print(ctx_input.size())
                print(att_feats.size())           
                output = self.att2ctx(att_feats).sum(dim=-2)
                state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))
                print('===========================')
                #output = self.att2ctx(ctx_input)
            """
        else:
            output = self.att2ctx(ctx_input)
            # save the context vector to state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state



class BertAoAModel(AttModel):
    def __init__(self, opt):
        super(BertAoAModel, self).__init__(opt)
        self.num_layers = 2
        # mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)
        if opt.use_multi_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.multi_head_scale * opt.rnn_size)

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
