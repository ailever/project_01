#!bin/bash

#Download COCO captions and preprocess them
python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train

#Download COCO dataset and pre-extract the image features (Skip if you are using bottom-up feature)
python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root images/coco
python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/stylized_cocotalk --images_root images/stylized_coco



