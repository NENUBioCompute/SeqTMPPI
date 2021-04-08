# Title     : _8ImexPredict.py
# Created by: julse@qq.com
# Created on: 2021/3/22 19:36
# des : TODO

import time

from _1positiveSample import handlePair
from common import check_path

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    fin = '/home/jjhnenu/data/PPI/stage2/uniprotPairFromImex.txt'
    foutdir = 'file/8ImexPredict'
    check_path(foutdir)
    handlePair(foutdir, sep='\t', fin=fin, jumpStep=[5], keepOne=True)
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

