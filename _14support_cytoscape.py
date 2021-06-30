# Title     : _14support_cytoscape.py
# Created by: julse@qq.com
# Created on: 2021/4/28 11:02
# des : TMP nonTMP ttdtarget 三者组合，overlap
#     fin_posi = 'file/3cluster/4posi.tsv'
# P34709	P34708	1
# P34709	P34691	1
# Q61824	Q60631	1


import time
import os
import pandas as pd

from _9imgPlot import plotVenn
from common import check_path, readIDlist
def func(x):
    y = []
    y.append(x[0])
    y.append('%s_%s'%(x[1],x[2]))
    return pd.Series(y)
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    dirout = 'file/14cytoscape'
    check_path(dirout)

    ################  overlap
    # fins = ['file/5statistic/positive/1tmp.list',
    #         'file/5statistic/positive/1nontmp.list',
    #         'file/otherfile/3ttd.list']
    # names = ['Transmembrane Protein','non-Transmembrane Protein','TTD Drug Target']
    #
    # f7proteinType = os.path.join(dirout, 'TMP_nonTMP_ttd.png')
    # plotVenn(fins, names, f7proteinType,fontsize=20,dpi=300)

    # fins = ['file/5statistic/positive/1tmp.list',
    #         'file/5statistic/positive/1nontmp.list',
    #         'file/otherfile/2drugtarget.list']
    # names = ['Transmembrane Protein','non-Transmembrane Protein','DrugBank Drug Target']
    #
    # f7proteinType = os.path.join(dirout, 'TMP_nonTMP_drugbank.png')
    # plotVenn(fins, names, f7proteinType,fontsize=20,dpi=300)

    # fins = ['file/5statistic/positive/1tmp.list',
    #         'file/5statistic/positive/1nontmp.list',
    #         'file/otherfile/3ttd.list']
    #
    # fout_nodetype = os.path.join(dirout,'1posi_node_type.tsv')
    # tmp = pd.read_table(fins[0],header=None)
    # nontmp = pd.read_table(fins[1],header=None)
    # ttd = pd.read_table(fins[2],header=None)
    # tmp[1] = 'tmp'
    # nontmp[1] = 'nontmp'
    # ttd[1] = 'ttd_target'
    #
    # protein = pd.concat([tmp,nontmp])
    # ttd.columns = [0, 2]
    # nodetype = protein.merge(ttd,how='left')
    # nodetype = nodetype.fillna(value='not_target')
    # nodetype1 = nodetype.apply(lambda x:func(x),axis=1)
    # nodetype1.to_csv(fout_nodetype,header=None,index=None,sep='\t')

    pass
    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


