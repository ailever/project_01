import torch
import torch.nn as nn


class TEST(nn.Module):
    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)


class test1(TEST):
    def _forward(self, x):
        return 1*x


class test2(TEST):
    def _forward(self, x):
        return 2*x


class test3(TEST):
    def _forward(self, x):
        return 3*x


test = TEST()
test()
