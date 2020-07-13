import json

obj = json.load(open('./grnet_val.json'))
overall = obj['overall']
imgToEval = obj['imgToEval']


print('\n* overall')
Bleu_1 = overall['Bleu_1'];                     print(f' - Bleu_1 : {Bleu_1}')
Bleu_2 = overall['Bleu_2'];                     print(f' - Bleu_2 : {Bleu_2}')
Bleu_3 = overall['Bleu_3'];                     print(f' - Bleu_3 : {Bleu_3}')
Bleu_4 = overall['Bleu_4'];                     print(f' - Bleu_4 : {Bleu_4}')
METEOR = overall['METEOR'];                     print(f' - METEOR : {METEOR}')
ROUGE_L = overall['ROUGE_L'];                   print(f' - ROUGE_L : {ROUGE_L}')
CIDEr = overall['CIDEr'];                       print(f' - CIDEr : {CIDEr}')
SPICE = overall['SPICE'];                       print(f' - SPICE : {SPICE}')
WMD = overall['WMD'];                           print(f' - WMD : {WMD}')
bad_count_rate = overall['bad_count_rate'];     print(f' - bad_count_rate : {bad_count_rate}')




print('\n* imgToEval')
for idx, (key, value)in enumerate(imgToEval.items()):
    if idx == 0:
        image_id = imgToEval[key]['image_id'];      print(f' - image_id : {image_id}')
        Bleu_1 = imgToEval[key]['Bleu_1'];          print(f' - Bleu_1 : {Bleu_1}')
        Bleu_2 = imgToEval[key]['Bleu_2'];          print(f' - Bleu_2 : {Bleu_2}')
        Bleu_3 = imgToEval[key]['Bleu_3'];          print(f' - Bleu_3 : {Bleu_3}')
        Bleu_4 = imgToEval[key]['Bleu_4'];          print(f' - Bleu_4 : {Bleu_4}')
        METEOR = imgToEval[key]['METEOR'];          print(f' - METEOR : {METEOR}')
        ROUGE_L = imgToEval[key]['ROUGE_L'];        print(f' - ROUGE_L : {ROUGE_L}')
        CIDEr = imgToEval[key]['CIDEr'];            print(f' - CIDEr : {CIDEr}')
        WMD = imgToEval[key]['WMD'];                print(f' - WMD : {WMD}')
        caption = imgToEval[key]['caption'];        print(f' - caption : {caption}')
        SPICE = imgToEval[key]['SPICE'];            print(' - SPICE')
        All = imgToEval[key]['SPICE']['All'];                   print(f'  - All : {All}')
        Relation = imgToEval[key]['SPICE']['Relation'];         print(f'  - Relation : {Relation}')
        Cardinality = imgToEval[key]['SPICE']['Cardinality'];   print(f'  - Cardinality : {Cardinality}')
        Attribute = imgToEval[key]['SPICE']['Attribute'];       print(f'  - Attribute : {Attribute}')
        Size = imgToEval[key]['SPICE']['Size'];                 print(f'  - Size : {Size}')
        Color = imgToEval[key]['SPICE']['Color'];               print(f'  - Color : {Color}')
        Object = imgToEval[key]['SPICE']['Object'];             print(f'  - Object : {Object}')
    
    """
    image_id = imgToEval[key]['image_id']
    caption = imgToEval[key]['caption']
    print(f'{image_id:6} , {caption}')
    """
