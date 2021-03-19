# Title     : _11correct.py
# Created by: julse@qq.com
# Created on: 2021/3/19 14:31
# des : something is wroing with subcellular


import time
import pandas as pd

from dao import getSingleInfo

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    '''
    time 66.807687997818
    (27134, 8) 最后一列是空列
    '''
    # fin = 'file/1positive/1tmp_nontmp.tsv'
    # f1allinfo = 'file2.0/1positive/1allinfo.tsv'
    # getSingleInfo(fin, f1allinfo, fin_type='pair')


    pass
    # countline(f3pair)
    # fin = 'file/5statistic/positive/11TmpSubcellularCount.tsv'
    # df = pd.read_table(fin,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


