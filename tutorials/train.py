import os, sys
sys.path.append('../')

import torch
import torch.nn as nn

import tutor_opts
from tutor_loader import DataLoader

import models
from misc.loss_wrapper import LossWrapper



opt = tutor_opts.parse_opt()
loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length
opt.vocab = loader.get_vocab()

model = models.setup(opt).cuda(); del opt.vocab
dp_model = torch.nn.DataParallel(model)
lw_model = LossWrapper(model, opt)
dp_lw_model = torch.nn.DataParallel(lw_model)
dp_lw_model.train()

data = loader.get_batch('train')
tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
tmp = [_ if _ is None else _.cuda() for _ in tmp]
fc_feats, att_feats, labels, masks, att_masks = tmp

model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag=False)
