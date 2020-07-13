import sys
sys.path.append('../')

import opts
import argparse

args = opts.parse_opt()
print('* parse_opt : ', vars(args))

parser = argparse.ArgumentParser()
opts.add_eval_options(parser)
args = parser.parse_args()
print('\n* add_eval_options', vars(args))
