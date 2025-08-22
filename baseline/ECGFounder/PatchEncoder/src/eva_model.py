import argparse
import sys

from helper_code import *
from code_v5_eva import eva_classify_model

# Parse arguments
def get_parser():
    description = 'Train the Challenge model(s).'
    parser = argparse.ArgumentParser(description=description)

    # 0 : 3 x 4
    # 1 : 3 x 4 + Ⅱ
    # 2 : 3 x 4 + Ⅱ + V1
    # 3 : 2 x 6 
    # 4 : 2 x 6+Ⅱ
    # 5 : 12 
    # 6 : 各种排布随机出现
    parser.add_argument('-l', '--layout', type=int, required=True,help="choose layout")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p1','--pic_path',type=str, required=True)
    parser.add_argument('-p2','--table_path',type=str, required=True)
    
    return parser

# Run the code
def run(args):
    #train_digitization_model(args.data_folder, args.model_folder, args.verbose) ### Teams: Implement this function!!!
    eva_classify_model(args.pic_path,args.table_path,args.layout,args.verbose) ### Teams: Implement this function!!!


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))