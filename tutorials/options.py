import sys
sys.path.append('../')

import opts
import argparse

def parse_opt():
    args = opts.parse_opt()
    print('* parse_opt : ', vars(args))


def add_eval_options():
    parser = argparse.ArgumentParser()
    opts.add_eval_options(parser)
    args = parser.parse_args()
    print('\n* add_eval_options', vars(args))


def main():
    parse_opt()
    add_eval_options()


if __name__ == "__main__":
    main()
