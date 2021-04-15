# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/21 20:17
@desc:
"""
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
def dropSimPair(fin_pair,fin_cluster_tmp_tsv,fin_cluster_nontmp_tsv,fout):
    '''
    :param fin_pair:
        TMPID\tNONTMPID\n

        Q7BCK4	B6JN06  0
        E7QG89	B2FN41  1
        ...

    :param fin_cluster_tmp_tsv:
        ID\tclusterNum\n

        Q20JQ7	0
        Q9UQD0	0
        P23467	1
        P36495	2
        ...

    :param fin_cluster_nontmp_tsv:
        same as fin_cluster_tmp_tsv
    :param fout:
        TMPID\tNONTMPID\tTMPclusterNum\tNONTMPclusterNUM\n

        Q7BCK4  B6JN06  0  274.0   3704.0
        E7QG89  B2FN41  1  5226.0  4848.0
    ...

    :return:
    fin_pair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified.txt'
    fin_cluster_tmp_tsv = 'file/cluster/allqualified_tmp_4.tsv'
    fin_cluster_nontmp_tsv = 'file/cluster/allqualified_nontmp_4.tsv'
    fout = 'file/cluster/all_qualified_nosim_4.txt'
    dropSimPair(fin_pair, fin_cluster_tmp_tsv, fin_cluster_nontmp_tsv, fout)
    '''
    df1 = pd.read_csv(fin_pair,sep='\t',header=None)
    df2 = pd.concat([pd.read_csv(fin_cluster_tmp_tsv,sep='\t',header=None),pd.read_csv(fin_cluster_nontmp_tsv,sep='\t',header=None)])
    df3 = df1.applymap(lambda x:df2[df2[0]==x][1].values[0])
    df4 = pd.concat([df1,df3],axis=1)
    df4.columns = [0,1,2,3]
    df5 = df4[df4[2]!=df4[3]]
    df5.to_csv(fout,index=None,header=None,sep='\t')
# def myfunc(x):
#     print('---------------------------')
#     print(x)
#     print(df2[df2[0] == x])
#     return df2[df2[0] == x][1].values[0]



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
    df2 = pd.concat([pd.read_csv(fin_cluster_tmp_tsv, sep='\t', header=None),
                     pd.read_csv(fin_cluster_nontmp_tsv, sep='\t', header=None)])
    df3 = pd.concat([df1[0], df1[1]], axis=1).applymap(lambda x: df2[df2[0] == x][1].values[0])
    df4 = pd.concat([df1, df3], axis=1)
    df4.columns = [0, 1, 2, 3, 4]
    # df5 = df4.applymap(lambda x:myfunc(x),axis = 1)

    df4[5] = 0
    df4.at[3, 4] = 8176
    for row1 in range(df4.shape[0]):
        for row2 in range(row1 + 1, df4.shape[0]):
            # 0 vs 1-8
            # 1 vs 2-8
            # ...
            # 7 vs 8
            print(row1, row2)
            if df4.iat[row1, 3] == df4.iat[row2, 3]: df4.iat[row2, 5] = 1
    df5 = df4[df4[5] == 0]
    df5.to_csv(fout, index=False, header=False, sep='\t')
    if fout_clus !=None: df4.to_csv(fout_clus, index=False, header=False, sep='\t')
    if fout !=None: df5.to_csv(fout, index=False, header=False, sep='\t')
    return df5


if __name__ == '__main__':

    pass
    # fin = 'file/cluster/allqualified_tmp_4.clstr'
    # fout = 'file/cluster/allqualified_tmp_4.tsv'
    # cluster2Table(fin,fout)
    #
    # fin = 'file/cluster/allqualified_nontmp_4.clstr'
    # fout = 'file/cluster/allqualified_nontmp_4.tsv'
    # cluster2Table(fin, fout)

    fin_pair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified_label.txt'
    fin_cluster_tmp_tsv = 'file/cluster/allqualified_tmp_4.tsv'
    fin_cluster_nontmp_tsv = 'file/cluster/allqualified_nontmp_4.tsv'
    fout = 'file/cluster/all_qualified_nosim_4.txt'
    fout_clus = 'file/cluster/all_qualified_clus_4.txt'
    pairWithClusterLable(fin_pair,fin_cluster_tmp_tsv,fin_cluster_nontmp_tsv,fout_clus=fout_clus,fout=fout)














