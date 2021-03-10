import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DatabaseOperation2 import DataOperation
from dao import queryPfam


def loadFeature(self, fin_pair,dir_in):
    # dir_in = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    fin_pair = 'file/4train/0/validate.txt'
    df = pd.read_table(fin_pair, header=None)[1:10]
    df1 = df.apply(lambda x: func(dir_in,x),axis=1).values
def func(dir_in,x):
    eachfile = os.path.join(dir_in, '%s_%s.npy' % (x[0], x[1]))
    print(eachfile)
    return np.append(np.load(eachfile,x[2]))
def funcPfam(x):
    # ac = 'P03372'
    do = DataOperation('uniprot', 'uniprot_sprot')
    result = queryPfam(x[0], do, tophit=True)
    x[1] = result[0]
    x[2] = result[1]
    return x


if __name__ == '__main__':
    dirout = 'file/5statistic/positive'
    f1tmp = os.path.join(dirout,'1tmp.list')

    fposi = 'file/3cluster/4posi.tsv'
    dirout = 'file/5statistic/positive'
    f2gpcr = os.path.join(dirout,'2gpcr.list')
    f3gpcr_con = os.path.join(dirout,'3gpcr_con.tsv')


    f5tmpPfam = os.path.join(dirout,'5tmpPfam.tsv')
    f5nontmpPfam = os.path.join(dirout,'5nontmpPfam.tsv')




