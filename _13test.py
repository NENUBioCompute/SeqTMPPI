# Title     : _13test.py
# Created by: julse@qq.com
# Created on: 2021/4/16 14:59
# des : TODO

import time

from _10humanTrain_support import saveRelated
from dao import getPairInfo

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    fin = 'file/12test/test_nolabel.txt'
    fout = 'file/12test/test_nolabel.fasta'
    # getPairInfo(fin, fout, sep='\t')
    dirout = 'file/12test/related'
    saveRelated(fout,dirout)


    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

