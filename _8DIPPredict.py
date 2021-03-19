# Title     : _8DIPPredict.py
# Created by: julse@qq.com
# Created on: 2021/3/19 9:12
# des : TODO


import time
import pandas as pd

from _1positiveSample import handlePair
from common import countline, check_path

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    '''
    combine negative
    '''
    dirDIP = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP_qualified'

    fin = '/home/jjhnenu/data/PPI/release/otherdata/DIP/Ecoli/2Ecoli20170205_id_pair_12246.txt'
    foutdir = 'file/8DIPPredict/data/Ecoli'
    check_path(foutdir)
    handlePair(foutdir,sep='\t',fin=fin,jumpStep=[5],keepOne=False)



    pass
    fin = 'file/8DIPPredict/data/Ecoli/2pair.tsv'
    countline(fin)
    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

