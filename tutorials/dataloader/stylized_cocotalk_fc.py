import os
import numpy as np

fc_feat = os.path.join('../../data/stylized_cocotalk_fc', '1000.npy')
loader = np.load(fc_feat)
print(len(loader))
print(loader)
