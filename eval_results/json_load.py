import json

obj = json.load(open('./grnet_val.json'))
overall = obj['overall']
imgToEval = obj['imgToEval']



Bleu_1 = overall['Bleu_1'];                     print(f'* Bleu_1 : {Bleu_1}')
Bleu_2 = overall['Bleu_2'];                     print(f'* Bleu_2 : {Bleu_2}')
Bleu_3 = overall['Bleu_3'];                     print(f'* Bleu_3 : {Bleu_3}')
Bleu_4 = overall['Bleu_4'];                     print(f'* Bleu_4 : {Bleu_4}')
METEOR = overall['METEOR'];                     print(f'* METEOR : {METEOR}')
ROUGE_L = overall['ROUGE_L'];                   print(f'* ROUGE_L : {ROUGE_L}')
CIDEr = overall['CIDEr'];                       print(f'* CIDEr : {CIDEr}')
SPICE = overall['SPICE'];                       print(f'* SPICE : {SPICE}')
WMD = overall['WMD'];                           print(f'* WMD : {WMD}')
bad_count_rate = overall['bad_count_rate'];     print(f'* bad_count_rate : {bad_count_rate}')
