import os, sys
sys.path.append('../')
import json
import tutor_opts
import torch.utils.data as data
from tutor_loader import DataLoader
from debugging import Debugger

def get_batch():
    opts = tutor_opts.parse_opt()
    loader = DataLoader(opts)
    get_batch = loader.get_batch(split='train', batch_size=None)
    att_feats = get_batch['att_feats']; print(f' - x=loader.get_batch(), x["att_feats"] : {att_feats.size(), att_feats.type()}')


def prefetch():
    opts = tutor_opts.parse_opt()
    loader = DataLoader(opts)
    """
    opts.seq_per_img # 5
    opts.batch_size # 16
    """
    tmp_fc, tmp_att, tmp_seq, ix, tmp_wrapped = loader._prefetch_process['train'].get()
    print(f'* tmp_fc :\n{tmp_fc}\n')
    print(f'* tmp_att :\n{tmp_att}\n')
    print(f'* tmp_seq :\n{tmp_seq}\n')
    print(f'* ix :\n{ix}\n')
    print(f'* tmp_wrapped :\n{tmp_wrapped}\n')


def split_ix():
    opts = tutor_opts.parse_opt()
    info = json.load(open(opts.input_json))
    split_ix = {'train': [], 'val': [], 'test': []}
    for ix in range(len(info['images'])):
        split_ix['train'].append(ix)


def dataloader():
    class SubsetSampler(data.sampler.Sampler):
        r"""Samples elements randomly from a given list of indices, without replacement.
        Arguments:
            indices (list): a list of indices
        """

        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return (self.indices[i] for i in range(len(self.indices)))

        def __len__(self):
            return len(self.indices)


    opts = tutor_opts.parse_opt()
    info = json.load(open(opts.input_json))
    split_ix = {'train': [], 'val': [], 'test': []}
    iterators = {'train': 0, 'val': 0, 'test': 0}
    
    for ix in range(len(info['images'])):
        split_ix['train'].append(ix)
    
    dadaloader = data.Dataset()
    setattr(dataloader, 'split_ix', split_ix)
    setattr(dataloader, 'iterators', iterators)

    split_loader = iter(data.DataLoader(dataset=dataloader,
                                        batch_size=1,
                                        sampler=SubsetSampler(dataloader.split_ix['train'][dataloader.iterators['train']:]),
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=4, # 4 is usually enough
                                        collate_fn=lambda x: x[0]))
    
    tmp = split_loader.next()

def main():
    #get_batch()
    #prefetch()
    #split_ix()
    dataloader()
    
if __name__ == "__main__":
    main()
