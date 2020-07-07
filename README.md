# GRNet

## Table of contents

- Implementation
  - 1． Download & Installation
  - 2． Prepare dataset
- Related work
  - Image captioning
    - Survey
    - Natural Language Processing
    - Evaluation : metrics
---

## Implemetation

### 1. Download & Installation
`./`
```bash
$ git clone https://github.com/ailever/grnet.git
```
  - grnet


<br><br>
`./grnet/`
```bash
$ git clone https://github.com/ruotianluo/coco-caption.git
$ git clone https://github.com/ruotianluo/cider.git
$ pip install -r requirements.txt
```
  - grnet/coco-caption
  - grnet/cider


<br><br>
`./grnet/coco-caption/`
```bash
$ bash get_stanford_models.sh
$ bash bash get_google_word2vec_model.sh
```


<br><br>
`./grnet/data/`
```
$ wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
$ unzip caption_datasets.zip
$ mkdir imagenet_weights && wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth && mv resnet101-5d3b4d8f.pth imagenet_weights/resnet101.pth
```
  - grnet/data/dataset_coco.json
  - grnet/data/dataset_flickr30k.json
  - grnet/data/dataset_flickr8k.json
  - grnet/data/imagenet_weights/resnet101.pth


<br><br>

### 2. Prepare dataset
`./grnet/`
```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
  - grnet/data/cocotalk.json
  - grnet/data/cocotalk_label.h5


<br><br>
`./grnet/`
```bash
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root images/coco
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/stylized_cocotalk --images_root images/stylized_coco
```
  - data/cocotalk_fc
  - data/cocotalk_att
  - data/stylized_cocotalk_fc
  - data/stylized_cocotalk_att


<br><br>
### Additional resources
- image source : https://www.wikiart.org/
- tensorflow tutorial : https://www.tensorflow.org/tutorials/text/image_captioning
- captioning metrics : https://github.com/wangleihitcs/CaptionMetrics
- image captioning transformer : https://www.ctolib.com/krasserm-fairseq-image-captioning.html
- trainval.zip : `wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip`
- GoogleNews-vectors-negative300.bin : `wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"`

<br><br><br>

## Related work

### Image captioning
- Transform and Tell: Entity-Aware News Image Captioning (Apr 2020)｜[pdf](https://arxiv.org/abs/2004.08070)｜[github](https://github.com/alasdairtran/transform-and-tell)
- VL-BERT: Pre-training of Generic Visual-Linguistic Representations (Mar 2020)｜[pdf](https://openreview.net/forum?id=SygXPaEYvH)
- <b>X-Linear Attention Networks for Image Captioning (Mar 2020)</b>｜[pdf](https://arxiv.org/abs/2003.14080)｜[github](https://github.com/Panda-Peter/image-captioning)
- Show, Edit and Tell: A Framework for Editing Image Captions (Mar 2020)｜[pdf](https://arxiv.org/abs/2003.03107)
- Captioning Images Taken by People Who Are Blind (Feb 2020)｜[pdf](https://arxiv.org/abs/2002.08565)
- Analysis of diversity-accuracy tradeoff in image captioning (Feb 2020)｜[pdf](https://arxiv.org/abs/2002.11848)
- Show, Recall, and Tell: Image Captioning with Recall Mechanism (Jan 2020)｜[pdf](https://arxiv.org/abs/2001.05876)
- <b>Image Captioning: Transforming Objects into Words (Jan 2020)</b>｜[pdf](https://arxiv.org/abs/1906.05963)
- <b>Attention on Attention for Image Captioning (Aug 2019)</b>｜[pdf](https://arxiv.org/abs/1908.06954)｜[github](https://github.com/husthuaan/AoANet)
  - [REF] Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering (Mar 2018)｜[pdf](https://arxiv.org/abs/1707.07998)
- Show, Control and Tell:A Framework for Generating Controllable and Grounded Captions (May 2019)｜[pdf](https://arxiv.org/pdf/1811.10652.pdf)｜[github](https://github.com/aimagelab/show-control-and-tell)
- Auto-Encoding Scene Graphs for Image Captioning (Dec 2018)｜[pdf](https://arxiv.org/abs/1812.02378)
- Exploring Visual Relationship for Image Captioning (Sep 2018)｜[pdf](https://arxiv.org/abs/1809.07041)
- <b>Recurrent Fusion Network for Image Captioning (Jul 2018)</b>｜[pdf](https://arxiv.org/abs/1807.09986)
- <b>Areas of attention for image captioning (Aug 2017)｜[pdf](https://arxiv.org/abs/1612.01033)</b>
- <b>Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (Apr 2016)</b>｜[pdf](https://arxiv.org/abs/1502.03044)
- Auto-Encoding Scene Graphs for Image Captioning｜[pdf](https://arxiv.org/abs/1812.02378)
- Image Captioning with Semantic Attention (Mar 2016)｜[pdf](https://arxiv.org/abs/1603.03925)

#### Survey

- A Systematic Literature Review on Image Captioning (May 2019)｜[pdf](https://www.mdpi.com/2076-3417/9/10/2024)
- Visual to Text: Survey of Image and Video Captioning (Jan 2019)｜[pdf](https://www.researchgate.net/publication/330708929_Visual_to_Text_Survey_of_Image_and_Video_Captioning)
- A Comprehensive Survey of Deep Learning for ImageCaptioning (Oct 2018)｜[pdf](https://arxiv.org/pdf/1810.04020.pdf)
- A survey on deep neural network-based image captioning (June 2018)｜[pdf](https://link.springer.com/article/10.1007/s00371-018-1566-y)


#### Video captioning

- STAT: Spatial-Temporal Attention Mechanismfor Video Captioning (Jan 2020)｜[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744407&tag=1)

#### Natural Language Processing
- <b>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, [Bert, Transformer, Bi-Encoder, Masked Language model, Pre-training], (May 2019)</b>｜[pdf](https://arxiv.org/abs/1810.04805)
  - DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter, Mar 2020｜[pdf](https://arxiv.org/abs/1910.01108)
  - Cross-lingual Language Model Pretraining, XLM, (Jan 2019)｜[pdf](https://arxiv.org/abs/1901.07291)
  - SLNet: Stereo face liveness detection via dynamic disparity-maps and convolutional neural network, (March 2020)｜[pdf](https://www.sciencedirect.com/science/article/abs/pii/S0957417419307195)
  - RoBERTa: A Robustly Optimized BERT Pretraining Approach, (Jul 2019)｜[pdf](https://arxiv.org/abs/1907.11692)
- <b>Universal Language Model Fine-tuning for Text Classification , ULMFiT, (May 2018)</b>｜[pdf](https://arxiv.org/abs/1801.06146v5)
- <b>Language Models are Unsupervised Multitask Learners, [GPT, Transformer, Uni-Encoder]</b> ｜[pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- <b>Deep contextualized word representations, [ELMo, CNN, biLM, Pre-training], (Mar 2018)</b>｜[pdf](https://arxiv.org/abs/1802.05365v2)
- <b>Attention Is All You Need, [Transformer, Attention], (Dec 2017)</b>｜[pdf](https://arxiv.org/abs/1706.03762)

#### Ontology
- Domain-Specific Image Caption Generator with Semantic Ontology (Feb 2020)｜[pdf](https://ieeexplore.ieee.org/document/9070680)
- Exploiting Ontologies for Automatic Image Annotation (Aug 2005)｜[pdf](https://dl.acm.org/doi/pdf/10.1145/1076034.1076128?download=true)

#### Evaluation : metrics

- Learning to Evaluate Image Captioning (Jun 2018)｜[pdf](https://arxiv.org/abs/1806.06422)
