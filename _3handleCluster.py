import os

from PairDealer import concatPAN
from common import concatFile
import pandas as pd


def cluster2Table(fin,fout):
    '''
    :param fin:
    :param fout:
    :return:
    '''
    # fin = 'file/cluster/allqualified_tmp_4.clstr'
    # fout = 'file/cluster/allqualified_tmp_4.tsv'
    # cluster2Table(fin, fout)

    # format of fin
    # >Cluster 0
    # 0	2000aa, >Q9NY46... *
    # 1	1820aa, >P02719... at 47.75%
    # 2	1841aa, >Q9ER60... at 54.32%
    # 3	1978aa, >O88420... at 74.77%
    # 4	1840aa, >P15390... at 54.57%
    # 5	1784aa, >Q20JQ7... at 41.48%
    # 6	1980aa, >Q9UQD0... at 74.95%
    # >Cluster 1
    # ...

    # format of fout
    # Q20JQ7	0
    # Q9UQD0	0
    # P23467	1
    # P36495	2
    # ...


    with open(fin, 'r') as fi, open(fout, 'w') as fo:
        clusterNum = ''
        line = fi.readline()
        while (line):
            if '>' == line[0]:
                clusterNum = line.split(' ')[1]
            else:
                ac = line[line.index('>') + 1:line.index('.')]
                fo.write('%s\t%s' % (ac, clusterNum))
                fo.flush()
            line = fi.readline()
    df = pd.read_csv(fout,header=None,sep='\t').drop_duplicates()
    df.to_csv(fout,header=None,index=None,sep='\t')
def pairWithClusterLable(fin_pair,fin_cluster_tmp_tsv,fin_cluster_nontmp_tsv,fout_clus=None,fout=None):
    '''
    :param fin_pair:
        label=0 means negative sample
        label=1 means positive sample

        # tmpac\tnontmpac\tlabel\n
        Q9HBL7	P59634	1
        P05023	P10636	1
        Q7BCK4	B6JN06	0
        E7QG89	B2FN41	0\n

    :param fin_cluster_tmp_tsv:
        tmpac\tclusnum\n
        P41597	3138
        Q9BBN9	3139
        A4QLQ2	3139\n

    :param fin_cluster_nontmp_tsv:
        similiar to fin_cluster_tmp_tsv

    :param fout_clus: df that sign A'-B' pair
        tmpac\tnontmpac\label\tmpclusnum\tnontmpclusnum\n
        P19634	Q15628	1	644.0	7375.0
        P25445	Q92851	1	3542.0	4009.0

    :param fout: df that drop A'-B' pair
        similiar to fout_clus

    :return: df that drop A'-B' pair
    example:
    # fin_pair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified_label.txt'
    # fin_cluster_tmp_tsv = 'file/cluster/allqualified_tmp_4.tsv'
    # fin_cluster_nontmp_tsv = 'file/cluster/allqualified_nontmp_4.tsv'
    # fout = 'file/cluster/all_qualified_nosim_4.txt'
    # fout_clus = 'file/cluster/all_qualified_clus_4.txt'
    # pairWithClusterLable(fin_pair,fin_cluster_tmp_tsv,fin_cluster_nontmp_tsv,fout_clus=fout_clus,fout=fout)
    '''
    df1 = pd.read_csv(fin_pair, sep='\t', header=None)
    df2 = pd.read_csv(fin_cluster_tmp_tsv, sep='\t', header=None)
    df3 = pd.read_csv(fin_cluster_nontmp_tsv, sep='\t', header=None)
    df2.columns = [0,3]
    df3.columns = [1,4]
    df4 = pd.merge(df1, df2,on=[0])
    df5 = pd.merge(df4, df3,on=[1]).drop_duplicates(subset=[0,1])
    df6 = df5.drop_duplicates(subset=[3,4])
    if fout_clus !=None: df5.to_csv(fout_clus, index=False, header=False, sep='\t')
    if fout !=None: df6.to_csv(fout, index=False, header=False, sep='\t')
    return df6
def saveRelated(f3pair,f4pair,f4posi,f4nega):
    # f4pair = 'file/3cluster/4pair.tsv'
    # f4posi = 'file/3cluster/4posi.tsv'
    # f4nega = 'file/3cluster/4nega.tsv'
    df = pd.read_csv(f3pair,sep='\t',header=None)[[0,1,2]]
    df.to_csv(f4pair,header=False,index=False,sep='\t')
    df[df[2]==1].to_csv(f4posi,header=False,index=False,sep='\t')
    df[df[2]==0].to_csv(f4nega,header=False,index=False,sep='\t')
if __name__ == '__main__':
    pass
    '''
    config path
    '''

    f1List_tmp_fasta = ['file/1positive/2tmp.fasta', 'file/2negative/4subcellular/2tmp.fasta']
    f1List_nontmp_fasta = ['file/1positive/2nontmp.fasta','file/2negative/4subcellular/2nontmp.fasta']
    fin_posi = 'file/1positive/2pair.tsv'
    fin_nega = 'file/2negative/4subcellular/2pair.tsv'

    foutdir = 'file/3cluster/'
    f2pair = 'file/3cluster/1all.tsv'
    f3out_tmp = 'file/3cluster/3tmp.tsv'
    f3out_nontmp = 'file/3cluster/3nontmp.tsv'
    f3pair = 'file/3cluster/3pair.tsv'
    f3pair_clstr = 'file/3cluster/3pair_clstr.tsv'
    f4pair = 'file/3cluster/4pair.tsv'
    f4posi = 'file/3cluster/4posi.tsv'
    f4nega = 'file/3cluster/4nega.tsv'

    # '''
    # tmp
    # '''
    # concatFile(f1List_tmp_fasta, os.path.join(foutdir,'1tmp.fasta'))
    # '''
    # nontmp
    # '''
    # # fout = 'file/3cluster/1nontmp.fasta'
    # concatFile(f1List_nontmp_fasta,  os.path.join(foutdir,'1nontmp.fasta'))
    # '''
    # concat positive and negative pair
    # '''
    # concatPAN(fin_posi, fin_nega, f2pair)
    '''
    cd hit 0.4 cd-hit tool :http://weizhong-lab.ucsd.edu/cdhit_suite/cgi-bin/index.cgi?cmd=cd-hit
    get  *.clstr file
    '''

    # fin_tmp = 'file/3cluster/2tmp.clstr'
    # cluster2Table(fin_tmp,f3out_tmp)
    #
    # fin_nontmp = 'file/3cluster/2nontmp.clstr'
    # cluster2Table(fin_nontmp, f3out_nontmp)


    # pairWithClusterLable(f2pair,f3out_tmp,f3out_nontmp,fout_clus=f3pair_clstr,fout=f3pair)

    '''
    extract positive,negative
    '''
    # saveRelated(f3pair,f4pair,f4posi,f4nega)
