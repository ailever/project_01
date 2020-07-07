from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch



from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .AttModel import *
from .TransformerModel import TransformerModel
from .AoAModel import AoAModel
from .BertAoAModel import BertAoAModel

def setup(opt):
    print(opt.caption_model)
    model = AoAModel(opt)

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        print(opt.start_from)
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model
