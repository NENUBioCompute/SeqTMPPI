# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/27 14:20
@desc:
"""
import os

import pandas as pd

from FastaDealear import FastaDealer

def saveFasta(fin,fout,AC=0,Seq=6):
    df = pd.read_table(fin, header=None)
    df.index = df[AC]
    fd = FastaDealer()
    fd.dict2fasta(df[Seq].to_dict(), fout, multi=True)
    print('save %d tmp ' % df.shape[0], fout)
def extractPairAndFasta(fin,fout_fasta,fout_tmp_fasta=None,fout_nontmp_fasta=None):
    '''
    :param fin: pair info
    :param fout: qualified pair
    :param fout_fasta: qualified pair fasta
    :return:
    fin = 'file/_2pair_info.txt'
    fout = 'file/_3pair_info_qualified.txt'
    fout_fasta = 'file/_3pair.fasta'
    extractPairAndFasta(fin,fout,fout_fasta)
    '''
    '''get pair'''
    df1 = pd.read_csv(fin, sep='\t', header=None, skipinitialspace=True)
    '''get fasta'''
    tmp = df1.loc[:,[0,6]]
    nontmp = df1.loc[:,[7,13]]
    nontmp.columns = [0,6]
    df2 = pd.concat([tmp,nontmp])
    df2 = df2.drop_duplicates()
    df2.index = df2[0]
    dic  = df2[6].to_dict()
    print('save %d protein fasta'%df2.shape[0],fout_fasta)
    fd = FastaDealer()
    fd.dict2fasta(dic, fout_fasta, multi=True)
    '''get tmp fasta '''
    if fout_tmp_fasta:
        tmp.index = tmp[0]
        tmp = tmp.drop_duplicates()
        fd.dict2fasta(tmp[6].to_dict(), fout_tmp_fasta, multi=True)
        print('save %d tmp '%tmp.shape[0],fout_tmp_fasta)
    '''get nontmp fasta '''
    if fout_nontmp_fasta:
        nontmp.index = nontmp[0]
        nontmp = nontmp.drop_duplicates()
        fd.dict2fasta(nontmp[6].to_dict(), fout_nontmp_fasta, multi=True)
        print('save %d nontmp '%nontmp.shape[0],fout_nontmp_fasta)
def saveQualified(fin,fout):
    df = pd.read_csv(fin,sep='\t',header=None,skipinitialspace=True)
    df = df.iloc[:,:-1]
    df1 = df[df[3] & df[4] & df[10] & df[11]]
    df1 = df1.drop_duplicates(subset=[6,13]).drop_duplicates(subset=[1,8])
    print('save %d pair'%df1.shape[0])
    df1.to_csv(fout,header=False,index=False,sep='\t')


def getproteinlist(fin,
                   ftmp=None,fnontmp=None,fall=None,
                   ftmp_info=None, ftmp_nontmp_info=None, fall_info=None
                   ):
    df = pd.read_csv(fin,sep='\t',header=None)
    tmp = df.loc[:, 0].drop_duplicates()
    nontmp = df.loc[:, 7].drop_duplicates()
    nontmp.columns = [0]
    all = pd.concat([tmp,nontmp]).drop_duplicates()
    if ftmp:tmp.to_csv(ftmp,header=False,index=False)
    if fnontmp:nontmp.to_csv(fnontmp,header=False,index=False)
    if fall:all.to_csv(fall,header=False,index=False)

    tmp = df.loc[:, 0:6].drop_duplicates(subset=[0])
    nontmp = df.loc[:, 7:13].drop_duplicates(subset=[7])
    nontmp.columns = [x for x in range(0,7)]
    all = pd.concat([tmp, nontmp]).drop_duplicates()
    if ftmp_info:tmp.to_csv(ftmp_info,header=False,index=False)
    if ftmp_nontmp_info:nontmp.to_csv(ftmp_nontmp_info,header=False,index=False)
    if fall_info:all.to_csv(fall_info,header=False,index=False)
    # return tmp.values,nontmp.values,all.values
########################## support ################################
def simplifyTable(fin,fout,cols=[0,7]):
    df = pd.read_csv(fin, sep='\t', header=None)
    df_sim = df.loc[:, cols]
    df_sim.to_csv(fout, header=None, index=None, sep='\t')



######################### subcellular #############################
def handleRow(fin,fout,calcuSubcell):
    # fin = 'file/3intAct_pair_norepeat_info_qualified.txt'
    # fout = 'file/4subcellular.tsv'
    # handleRow(fin, fout,calcuSubcell)
    df = pd.read_csv(fin, sep='\t', header=None)
    df_subcel = df.apply(lambda x:calcuSubcell(x),axis=1)
    df_subcel.to_csv(fout, header=None, index=None, sep='\t')
    print(df_subcel.shape)

def calcuSubcell(x):
    tmp = set(eval(x[5]))
    nontmp = set(eval(x[12]))
    print(tmp & nontmp)
    print(tmp | nontmp)
    print(tmp - nontmp)
    print(nontmp - tmp)
    x[14] = list(tmp & nontmp)
    x[15] = list(tmp | nontmp)
    x[16] = list(tmp - nontmp)
    x[17] = list(nontmp - tmp)
    x[18] = len(x[14])==0
    return x
def saveDifferSubcell(fin,fout):
    df = pd.read_csv(fin, sep='\t', header=None)
    df_subcel = df[df[18]==True]
    df_subcel.to_csv(fout, header=None, index=None, sep='\t')
if __name__ == '__main__':
    pass
    # fin = 'file/3intAct_pair_norepeat_info_qualified.txt'
    # fout = 'file/_4subcellular.tsv'
    # handleRow(fin, fout,calcuSubcell)

    # fin = 'file/1positive/2tmp_nontmp_info_qualified.tsv'
    # fout = 'file/1positive/3subcellular/1subcellular.tsv'
    # handleRow(fin, fout,calcuSubcell)

    # fin = fin
    # fout = 'file/1positive/3subcellular/2Difersubcellular.tsv'
    # saveDifferSubcell(fin,fout)