import json

info = json.load(open('../data/cocotalk.json'))
ix_to_word = info['ix_to_word']; #print(f'* ix_to_word : {ix_to_word}')
images = info['images'];         #print(f'* images : {images}')




print('* min of info["ix_to_word"] : ', min(ix_to_word.keys()))
print('* max of info["ix_to_word"] : ', max(ix_to_word.keys()))
for ix, word in ix_to_word.items():
    # 0 < ix =< 999
    if ix == '999':
        print(ix ,word)


print('\n* num of info["images"] : ', len(images))
for i, image in enumerate(images):
    # 0 =< i < 123287
    if i == 123286:
        print(i, image)
        print(' - info["images"]["split"] : ', image['split'])
        print(' - info["images"]["file_path"] : ', image['file_path'])
        print(' - info["images"]["id"] : ', image['id'])
