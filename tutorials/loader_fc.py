import os, sys
sys.path.append('../')

import tutor_opts
from tutor_loader import DataLoader
import inspect

opts = tutor_opts.parse_opt()
loader = DataLoader(opts)
fc_loader = getattr(loader, 'fc_loader')


