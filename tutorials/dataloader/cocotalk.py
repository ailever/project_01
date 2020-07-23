import json


obj = json.load(open('../../data/cocotalk.json'))

ix_to_word = obj['ix_to_word']
images = obj['images']

# ix_to_word
count = 0
for ix in ix_to_word:
    if count == 1 : break
    count += 1
    print(ix)
    print()

# images
count = 0
for image in images:
    if count == 1 : break
    count += 1
    
    print('- image\n  ', image)
    print('- image_split\n  ', image['split'])
    print('- image_file_path\n  ', image['file_path'])
    print('- image_id\n  ', image['id'])
    print()



