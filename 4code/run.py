# encoding: utf-8
"""
@author: julse@qq.com
@time: 2021/4/15 16:56
@desc:
"""
import argparse

import os
from support import getFeature, savepredict


def main():
    dir_feature_db = os.path.join(dirout_result,'feadb')
    dirout_feature = os.path.join(dirout_result,'feature')
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    savepredict(fin_pair, dirout_feature, fin_model, dirout_result, batch_size=500)
if __name__ == '__main__':
    '''
    cmd = python run.py --model ../2model/0/_my_model.h5 --fasta ../3dataset/DIP_mus/2pair.fasta --pair ../3dataset/DIP_mus/0/all.txt --output_path result/
    cmd = python run.py -m ../2model/0/_my_model.h5 -f ../3dataset/DIP_mus/2pair.fasta -p ../3dataset/DIP_mus/0/all.txt -o result/
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='../2model/0/_my_model.h5')
    parser.add_argument('-f', '--fasta', default='../3dataset/DIP_mus/2pair.fasta')
    parser.add_argument('-p', '--pair', default='../3dataset/DIP_mus/0/all.txt')
    parser.add_argument('-o', '--output_path', default='result/')
    args = parser.parse_args()

    fin_model = args.model
    fin_fasta = args.fasta
    fin_pair = args.pair

    dirout_result = args.output_path



    # fin_model = '../2model/0/_my_model.h5'
    # fin_fasta = '../3dataset/DIP_mus/2pair.fasta'
    # fin_pair = '../3dataset/DIP_mus/0/all.txt'
    #
    # dirout_result = 'result'
    # dir_feature_db = 'result/feadb'
    # dirout_feature = 'result/feature'

    main()


