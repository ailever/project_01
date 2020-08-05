import json


obj = json.load(open('../../data/dataset_coco.json'))

images = obj['images']      # list
dataset = obj['dataset']    # str : coco

"""images
images[0]['filepath']
images[0]['sentids']
images[0]['filename']
images[0]['imgid']
images[0]['split']
images[0]['sentences'][0~4]['tokens']
images[0]['sentences'][0~4]['raw']
images[0]['sentences'][0~4]['imgid']
images[0]['sentences'][0~4]['sentid']
images[0]['cocoid']

"""

for image in images:
    for key, value in image.items():
        if key == 'sentences':
            print(f'- {key}')
            for v in value:
                for _k, _v in v.items():
                    if _k =='tokens':
                        print(f'  - {_k} : {_v}')
            for v in value:
                for _k, _v in v.items():
                    if _k =='raw':
                        print(f'  - {_k} : {_v}')
            for v in value:
                for _k, _v in v.items():
                    if _k =='imgid':
                        print(f'  - {_k} : {_v}')
            for v in value:
                for _k, _v in v.items():
                    if _k =='sentid':
                        print(f'  - {_k} : {_v}')
        else:
            print(f'- {key}')
            print(value)
            print()
    break

