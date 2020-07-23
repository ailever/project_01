import argparse
import os, sys
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np

import tutor_opts
from tutor_loader import DataLoader
from object_info import information

import misc.utils as utils
import models
from debugging import Debugger
debugger = Debugger()


# Input arguments and options
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='../log/log_grnet/model.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='../log/log_grnet/infos_grnet.pkl',
                help='path to infos to evaluate')
parser.add_argument('--gsp', type=int, default=0,
                help='gram schmidt process')
parser.add_argument('--seq_per_img', type=int, default=5,
                help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
tutor_opts.add_eval_options(parser)
opt = parser.parse_args()

# Load infos
with open('../log/log_grnet_aoa/infos_grnet_aoa.pkl', 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping
opt.vocab = vocab

loader = DataLoader(opt)
loader.reset_iterator(split=opt.split)

model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

eval_kwargs = vars(opt)
num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))


n = 0
predictions = []
while True:
    data = loader.get_batch(split='test')
    n = n + loader.batch_size


    # forward the model to also get generated samples for each image
    # Only leave one feature for each image, in case duplicate sample
    tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
           data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
           data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
    tmp = [_.cuda() if _ is not None else _ for _ in tmp]
    fc_feats, att_feats, att_masks = tmp
    
    # forward the model to also get generated samples for each image
    with torch.no_grad():
        seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

    sents = utils.decode_sequence(loader.get_vocab(), seq)
    for k, sent in enumerate(sents):
        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
        if eval_kwargs.get('dump_path', 0) == 1:
            entry['file_name'] = data['infos'][k]['file_path']
        predictions.append(entry)
    
    ix1 = data['bounds']['it_max']
    if num_images != -1     : ix1 = min(ix1, num_images)
    for i in range(n - ix1) : predictions.pop()
 
    
    if n < 100:
        debugger(seq, sents, data, entry, ix1, logname='predic')
    elif n == 100:
        del debugger
    else:
        pass

    if data['bounds']['wrapped']           : break
    if num_images >= 0 and n >= num_images : break
