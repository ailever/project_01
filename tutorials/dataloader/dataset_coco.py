import json


obj = json.load(open('../../data/dataset_coco.json'))

images = obj['images']      # list
dataset = obj['dataset']    # str : coco

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
