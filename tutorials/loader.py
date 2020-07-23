import os, sys
sys.path.append('../')

import tutor_opts
from tutor_loader import DataLoader

opts = tutor_opts.parse_opt()
loader = DataLoader(opts)

def loader_attr():
    print('\n* loader dir info')
    d = []
    for i in dir(loader):
        if str(i)[0:2] != "__":
            print(f' - {i}')
            d.append(str(i))
    
    print('\n* loader vars info')
    v = []
    for i in vars(loader):
        print(f' - {i}')
        v.append(str(i))
    
    print('\n* loader method info')
    method = set(d) - set(v)
    for i in method:
        print(f' - {i}')


def loader_simple_information():
    print('\n* loader_simple_information')
    batch_size = getattr(loader, 'batch_size');               print(f' - loader.batch_size : {batch_size}')
    seq_per_img = getattr(loader, 'seq_per_img');             print(f' - loader.seq_per_img : {seq_per_img}')
    use_fc = getattr(loader, 'use_fc');                       print(f' - loader.use_fc : {use_fc}')
    use_att = getattr(loader, 'use_att');                     print(f' - loader.use_att : {use_att}')
    use_box = getattr(loader, 'use_box');                     print(f' - loader.use_box : {use_box}')
    norm_att_feat = getattr(loader, 'norm_att_feat');         print(f' - loader.norm_att_feat : {norm_att_feat}')
    norm_box_feat = getattr(loader, 'norm_box_feat');         print(f' - loader.norm_box_feat : {norm_box_feat}')
    vocab_size = getattr(loader, 'vocab_size');               print(f' - loader.vocab_size : {vocab_size}')
    seq_length = getattr(loader, 'seq_length');               print(f' - loader.seq_length : {seq_length}')
    num_images = getattr(loader, 'num_images');               print(f' - loader.num_images : {num_images}')


def loader_opt():
    print('\n* loader_opt')
    opt = getattr(loader, 'opt');                             print(f' - loader.opt : {opt}')


def loader_info():
    print('\n* loader_info')

    info = getattr(loader, 'info');                           
    ix_to_word = info['ix_to_word'];                          #print(f' - info["ix_to_word"] : {ix_to_word}')
    images = info['images'];                                  #print(f' - info["images"] : {images}')
    

    indices = [] 
    for ix in ix_to_word:indices.append(int(ix))
    
    for idx, ix in enumerate(ix_to_word):
        if idx == 0 :
            print(f'  - loader.info["ix_to_word"][ix] : {ix_to_word[ix]}')
            print(f'   - loader.info["ix_to_word"], len of ix : {len(indices)}')
            print(f'   - loader.info["ix_to_word"], min of ix : {min(indices)}')
            print(f'   - loader.info["ix_to_word"], max of ix : {max(indices)}')

    images_id = []
    for image in images:
        images_id.append(int(image['id']))
        
    for idx, image in enumerate(images):
        if idx == 0:
            print(f'  - loader.info["images"]["split"] : {image["split"]}')
            print(f'  - loader.info["images"]["file_path"] : {image["file_path"]}')
            print(f'  - loader.info["images"]["id"] : {image["id"]}')
            print(f'   - loader.info["images"]["id"], len of id : {len(images_id)}')
            print(f'   - loader.info["images"]["id"], min of id : {min(images_id)}')
            print(f'   - loader.info["images"]["id"], max of id : {max(images_id)}')


def loader_ix_to_word():
    print('\n* loader_ix_to_word')
    
    ix_to_word = getattr(loader, 'ix_to_word');               #print(f' - loader.ix_to_word : {ix_to_word}')
    
    for idx, ix in enumerate(ix_to_word):
        if idx == 0 :
            print(f'  - loader.ix_to_word[ix] : {ix_to_word[ix]}')


def loader_h5_label_file():
    print('\n* loader_h5_label_file')
    h5_label_file = getattr(loader, 'h5_label_file');         print(f' - loader.h5_label_file : {h5_label_file}')


def loader_label():
    print('\n* loader_label')
    label = getattr(loader, 'label');                         print(f' - loader.label : {label}')
    label_start_ix = getattr(loader, 'label_start_ix');       print(f' - loader.label_start_ix : {label_start_ix}')
    label_end_ix = getattr(loader, 'label_end_ix');           print(f' - loader.label_end_ix : {label_end_ix}')


def loader_loader():
    print('\n* loader_loader')
    fc_loader = getattr(loader, 'fc_loader');                 print(f' - loader.fc_loader : {fc_loader}')
    att_loader = getattr(loader, 'att_loader');               print(f' - loader.att_loader : {att_loader}')
    box_loader = getattr(loader, 'box_loader');               print(f' - loader.box_loader : {box_loader}')


def loader_split_ix():
    print('\n* loader_split_ix')
    split_ix = getattr(loader, 'split_ix');                   #print(f' - loader.split_ix : {split_ix}')
    train = split_ix['train'];                                print(f'  - loader.split_ix["train"], len : {len(train)}')
    val = split_ix['val'];                                    print(f'  - loader.split_ix["val"], len : {len(val)}')
    test = split_ix['test'];                                  print(f'  - loader.split_ix["test"], len : {len(test)}')


def loader_etc():
    print('\n* loader_etc')
    iterators = getattr(loader, 'iterators');                 print(f' - loader.iterators : {iterators}')
    _prefetch_process = getattr(loader, '_prefetch_process'); print(f' - loader._prefetch_process : {_prefetch_process}')


def loader_method_get_batch():
    get_batch = loader.get_batch(split='train', batch_size=None)
    
    fc_feats = get_batch['fc_feats'];   print(f' - x=loader.get_batch(), x["fc_feats"] : {fc_feats.size(), fc_feats.type()}')
    att_feats = get_batch['att_feats']; print(f' - x=loader.get_batch(), x["att_feats"] : {att_feats.size(), att_feats.type()}')
    att_masks = get_batch['att_masks']; print(f' - x=loader.get_batch(), x["att_masks"] : {att_masks}')
    labels = get_batch['labels'];       print(f' - x=loader.get_batch(), x["labels"] : {labels.size(), labels.type()}')
    masks = get_batch['masks'];         print(f' - x=loader.get_batch(), x["masks"] : {masks.size(), masks.type()}')
    bounds = get_batch['bounds'];       print(f' - x=loader.get_batch(), x["bounds"] : {bounds}')
    
    infos = get_batch['infos']
    for i in infos :                    print(f' - x=loader.get_batch(), x["infos"] : {i}')
    
    gts = get_batch['gts']
    for i in gts :                      print(f' - x=loader.get_batch(), x["gts"] : {len(i)}')



def loader_method_etc():
    get_seq_length = loader.get_seq_length();                   print(f' - loader.get_seq_length() : {get_seq_length}')
    get_vocab_size = loader.get_vocab_size();                   print(f' - loader.get_vocab_size() : {get_vocab_size}')
    get_vocab = loader.get_vocab();                             print(f' - loader.get_vocab() : {len(get_vocab)}')
    reset_iterator = loader.reset_iterator(split='train');      print(f' - loader.reset_iterator() : {reset_iterator}')
    get_caption = loader.get_captions(ix=1, seq_per_img=5);     print(f' - loader.get_caption() : {get_caption}')


def main():
    #loader_attr()
    #loader_simple_information()
    #loader_opt()
    #loader_info()
    #loader_ix_to_word()
    #loader_h5_label_file()
    #loader_label()
    #loader_loader()
    #loader_split_ix()
    #loader_etc()
    
    loader_method_get_batch()
    #loader_method_etc()

if __name__ == "__main__":
    main()




