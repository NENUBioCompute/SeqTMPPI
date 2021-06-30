# Title     : _20pfam2clan.py
# Created by: julse@qq.com
# Created on: 2021/4/21 17:07
# des : pfam to clan
# 34.0版本的数据库中，共有 19179 个family， 7372个clan
# file/5statistic/support/Pfam-A.clans_34.0.tsv
# PF00001	CL0192	GPCR_A	7tm_1	7 transmembrane receptor (rhodopsin family)
# PF00002	CL0192	GPCR_A	7tm_2	7 transmembrane receptor (Secretin family)

# file/5statistic/positive/5tmpPfamsCount.tsv
# ('PF00001', '7tm_1')	246

import time
import pandas as pd

from dao import getFasta

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    fin = 'tool/queryFasta/UniprotID.txt'
    fout = 'tool/queryFasta/UniprotID.fasta'
    getFasta(fin,fout)

    pass
    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)



