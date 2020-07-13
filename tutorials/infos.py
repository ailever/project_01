import os, sys
sys.path.append('../')
import misc.utils as utils


# Load infos
with open('../log/log_grnet_aoa/infos_grnet_aoa.pkl', 'rb') as f:
    infos = utils.pickle_load(f)

iter = infos['iter'];                       #print(f'* iter : {iter}\n')
epoch = infos['epoch'];                     #print(f'* epoch : {epoch}\n')
iterators = infos['iterators'];             #print(f'* iterators : {iterators}\n')
split_ix = infos['split_ix'];               #print(f'* split_ix : {split_ix}\n')
vocab = infos['vocab'];                     #print(f'* vocab : {vocab}\n')
opt = infos['opt'];                         #print(f'* opt : {opt}\n')
best_val_score = infos['best_val_score'];   #print(f'* best_val_score : {best_val_score}\n')




train = split_ix['train']
l = []
for i in train:l.append(int(i))
print(f'* len of infos["split_ix"]["train"] : {len(l)}') # len : 113287
print(f'* min of infos["split_ix"]["train"] : {min(l)}') # min : 1
print(f'* max of infos["split_ix"]["train"] : {max(l)}') # max : 123286

val = split_ix['val']
l = []
for i in val:l.append(int(i))
print(f'* len of infos["split_ix"]["val"] : {len(l)}') # len : 5000
print(f'* min of infos["split_ix"]["val"] : {min(l)}') # min : 2
print(f'* max of infos["split_ix"]["val"] : {max(l)}') # max : 40482

test = split_ix['test']
l = []
for i in test:l.append(int(i))
print(f'* len of infos["split_ix"]["test"] : {len(l)}') # len : 5000
print(f'* min of infos["split_ix"]["test"] : {min(l)}') # min : 0
print(f'* max of infos["split_ix"]["test"] : {max(l)}') # max : 40503

l = []
for i in vocab:l.append(int(i))
print(f'* len of infos["vocab"] : {len(l)}') # len : 9487
print(f'* min of infos["vocab"] : {min(l)}') # min : 1
print(f'* max of infos["vocab"] : {max(l)}') # max : 9487

for key, value in vocab.items():
    if key == '1' : print(value)
