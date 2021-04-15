# Title     : _7IntactSample.py
# Created by: julse@qq.com
# Created on: 2021/2/28 21:16
# des : TODO
import os
import time
import pandas as pd
from dao import getPairTag
from dataset import simplifyTable



if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    '''
    1. get tmp nontmp pair
    time 1766.4131457805634
    '''
    fin = '/home/jjhnenu/SeqTMPPI20201226/file/1intAct_pair_norepeat.txt'
    dirout = 'file/7IntactSample'

    f1pairWithTag_info = os.path.join(dirout,'1pairWithTag_info.tsv')
    f1pairWithTag = os.path.join(dirout,'1pairWithTag.tsv')

    f2TMP_nonTMP = os.path.join(dirout,'2TMP_nonTMP.tsv')
    f2TMP_TMP = os.path.join(dirout,'2TMP_TMP.tsv')
    f2nonTMP_nonTMP = os.path.join(dirout,'2nonTMP_nonTMP.tsv')

    '''
    time 1732.4682846069336
    pair with tag
    0, TMP_SP
    1, TMP_TMP
    2, SP_SP
    '''
    # getPairTag(fin,f1pairWithTag_info,sep=',')
    # simplifyTable(f1pairWithTag_info, f1pairWithTag,cols=[0,7,14])

    '''
    split to three files
    '''
    # df = pd.read_table(f1pairWithTag,header=None)
    # df[df[2]==0].drop_duplicates().to_csv(f2TMP_nonTMP,header=None,index=None,sep='\t')
    # df[df[2]==1].to_csv(f2TMP_TMP,header=None,index=None,sep='\t')
    # df[df[2]==2].to_csv(f2nonTMP_nonTMP,header=None,index=None,sep='\t')


    # df = pd.read_table(f_keegdb_3pathway_disease,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)
