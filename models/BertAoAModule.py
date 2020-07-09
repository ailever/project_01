from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        return x
