import os
import time

from DatabaseOperation2 import DataOperation
from FastaDealear import FastaDealer
from common import countline, readIDlist, saveList
from dao import getPairInfo_TMP_nonTMP, ensomblePortein, save, composeTMP_nonTMP, queryProtein, \
    handleDicElement, getPairInfo
import pandas as pd

from dataset import extractPairAndFasta, getproteinlist, simplifyTable
from negativeData import  dropPositiveAndRepeate
def main():
    '''
    1. get tmp nontmp pair
    time 1766.4131457805634
    '''
    fin = 'file/1intAct_pair_norepeat.txt'
    fout = 'file/2intAct_pair_norepeat_info.txt'
    getPairInfo_TMP_nonTMP(fin, fout, sep=',')

    fin = 'file/2intAct_pair_norepeat_info.txt'
    fout = 'file/2TMP_nonTMP.pair'
    df = pd.read_csv(fin, sep='\t', header=None)
    df_sim = df.loc[:, [0, 7]]
    df_sim.to_csv(fout, header=None, index=None, sep='\t')

    '''
    2. get qualified tmp nontmp pair
    and fasta
    '''
    fin = 'file/2intAct_pair_norepeat_info.txt'
    fout = 'file/3intAct_pair_norepeat_info_qualified.txt'
    fout_fasta = 'file/3pair.fasta'
    fout_tmp_fasta = 'file/3tmp.fasta'
    fout_nontmp_fasta = 'file/3nontmp.fasta'
    extractPairAndFasta(fin, fout, fout_fasta, fout_tmp_fasta=fout_tmp_fasta, fout_nontmp_fasta=fout_nontmp_fasta)

    fin = 'file/3intAct_pair_norepeat_info_qualified.txt'
    fout = 'file/3positive.pair'
    df = pd.read_csv(fin, sep='\t', header=None)
    df_sim = df.loc[:, [0, 7]]
    df_sim.to_csv(fout, header=None, index=None, sep='\t')

    fin = 'file/3intAct_pair_norepeat_info_qualified.txt'
    ftmp = 'file/3tmp.list'
    fnontmp = 'file/3nontmp.list'
    fall = 'file/3all.list'
    _, _, alllist = getproteinlist(fin, ftmp=ftmp, fnontmp=fnontmp, fall=fall)
    print('those protein not save in the mongodb', save(alllist, 'seqtmppi_positive'))

def handlePair(fin,foutdir,sep=',',dbname=None,jumpStep=None):
    '''
    :param fin:
    :param foutdir:
    :param sep: sep of fin file
    :parameter dbname: name of mongodb
    :parameter jumpStep: skip some step in this method [1,2]
    :return:

    fin = 'file/1intAct_pair_norepeat.txt'
    foutdir = 'file/1positive'
    handlePair(fin,foutdir)
    '''

    '''
    config path
    '''
    f1tmp_nontmp_info = os.path.join(foutdir, '1tmp_nontmp_info.tsv')
    f1TMP_nonTMP = os.path.join(foutdir, '1TMP_nonTMP.tsv')
    f2tmp_nonTtmp_info_qualified = os.path.join(foutdir, '2tmp_nonTtmp_info_qualified.tsv')
    fout_fasta = os.path.join(foutdir, '2pair.fasta')
    fout_tmp_fasta = os.path.join(foutdir, '2tmp.fasta')
    fout_nontmp_fasta = os.path.join(foutdir, '2nontmp.fasta')
    f2positive = os.path.join(foutdir, '2positive.tsv')
    f2tmp = os.path.join(foutdir, '2tmp.list')
    f2nontmp = os.path.join(foutdir, '2nontmp.list')
    f2all = os.path.join(foutdir, '2all.list')
    f2tmp_info = os.path.join(foutdir, '2tmp_info.tsv')
    f2tmp_nontmp_info = os.path.join(foutdir, '2tmp_nontmp_info.tsv')
    f2all_info = os.path.join(foutdir, '2all_info.tsv')


    '''
    1. get tmp nontmp pair
    time 1766.4131457805634
    '''
    if jumpStep or 1 not in jumpStep:
        getPairInfo_TMP_nonTMP(fin, f1tmp_nontmp_info, sep=sep)
    simplifyTable(f1tmp_nontmp_info, f1TMP_nonTMP)
    '''
    2. get qualified tmp nontmp pair
    and fasta
    '''
    if jumpStep or 2 not in jumpStep:
        extractPairAndFasta(f1tmp_nontmp_info, f2tmp_nonTtmp_info_qualified, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
                            fout_nontmp_fasta=fout_nontmp_fasta)
        simplifyTable(f2tmp_nonTtmp_info_qualified, f2positive)

        getproteinlist(f2tmp_nonTtmp_info_qualified,
                       ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
                       ftmp_info=f2tmp_info,ftmp_nontmp_info=f2tmp_nontmp_info,fall_info=f2all_info)
    '''
    3. save to mongodb
    '''
    if dbname:
        notsvaelist = save(readIDlist(f2all), dbname)
        print('those protein not save in the mongodb',notsvaelist)

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    fin = 'file/1intAct_pair_norepeat.txt'
    foutdir = 'file/1positive'
    handlePair(fin, foutdir,dbname='seqtmppi_positive')

    '''
    test 
    '''
    # fin = 'file/_1pair.txt'
    # foutdir = 'file/1test'
    # handlePair(fin, foutdir,sep='\t',dbname='seqtmppi_positive_1')

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)