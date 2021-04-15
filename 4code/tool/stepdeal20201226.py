# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/20 16:46
@desc:
"""
import time

from DatabaseOperation2 import DataOperation
from FastaDealear import FastaDealer
from common import countline, readIDlist, saveList
from dao import getPairInfo_TMP_nonTMP, ensomblePortein, save, composeTMP_nonTMP, queryProtein, \
     getPairInfo
import pandas as pd

from dataset import extractPairAndFasta, getproteinlist
from negativeData import  dropPositiveAndRepeate






if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    '''
    check protein pair 
    '''

    '''
    1. get tmp nontmp pair
    time 1766.4131457805634
    '''
    # fin = 'file/1intAct_pair_norepeat.txt'
    # fout = 'file/2intAct_pair_norepeat_info.txt'
    # getPairInfo_TMP_nonTMP(fin,fout,sep=',')

    # fin = 'file/2intAct_pair_norepeat_info.txt'
    # fout = 'file/2TMP_nonTMP.pair'
    # df = pd.read_csv(fin, sep='\t', header=None)
    # df_sim = df.loc[:,[0,7]]
    # df_sim.to_csv(fout, header=None, index=None,sep='\t')

    '''
    2. get qualified tmp nontmp pair
    and fasta
    '''
    # fin = 'file/2intAct_pair_norepeat_info.txt'
    # fout = 'file/3intAct_pair_norepeat_info_qualified.txt'
    # fout_fasta = 'file/3pair.fasta'
    # fout_tmp_fasta = 'file/3tmp.fasta'
    # fout_nontmp_fasta ='file/3nontmp.fasta'
    # extractPairAndFasta(fin, fout, fout_fasta,fout_tmp_fasta=fout_tmp_fasta,fout_nontmp_fasta=fout_nontmp_fasta)

    # fin = 'file/3intAct_pair_norepeat_info_qualified.txt'
    # fout = 'file/3positive.pair'
    # df = pd.read_csv(fin, sep='\t', header=None)
    # df_sim = df.loc[:,[0,7]]
    # df_sim.to_csv(fout, header=None, index=None,sep='\t')

    # fin = 'file/3intAct_pair_norepeat_info_qualified.txt'
    # ftmp = 'file/3tmp.list'
    # fnontmp = 'file/3nontmp.list'
    # fall = 'file/3all.list'
    # _,_,alllist = getproteinlist(fin, ftmp=ftmp, fnontmp=fnontmp, fall=fall)
    # save(alllist)

    '''  test '''
    # fin = 'file/_2pair_info.txt'
    # fout = 'file/_3pair_info_qualified.txt'
    # fout_fasta = 'file/_3pair.fasta'
    # extractPairAndFasta(fin, fout, fout_fasta)

    # cluster http://weizhong-lab.ucsd.edu/cdhit_suite/cgi-bin/index.cgi?cmd=cd-hit

    '''
    get tmp list and nontmp list in qualified positive pair
    save to mongodb
    '''



    # fall = 'file/3all.list'
    # alllist = readIDlist(fall)
    # not_saveList = save(alllist)


######################################### negative pair ####################################

    '''
    get negative pair
    1. tmp list 
    2. nontmp list
    
    240258 nontmp {"keyword.@id":{$ne:'KW-0812'},keyword:{$exists:true},'comment.subcellularLocation.location.#text': {$exists:true}}
    73158 tmp {"keyword.@id":'KW-0812','comment.subcellularLocation.location.#text': {$exists:true}}
    '''
    # ftmp = 'file/4negative/1tmp.list'
    # fnontmp = 'file/4negative/1nontmp.list'
    # fpair = 'file/4negative/1pair.list'
    # composeTMP_nonTMP(ftmp,fnontmp,fpair,100000)

    '''
    删除正样本中出现的
    去重
    保留蛋白对合格的
    '''

    # fin = 'file/4negative/1pair.list'
    # fbase = 'file/3positive.pair'
    # fout = 'file/4negative/2pair_drop_positiveAndrepeate.list'
    # dropPositiveAndRepeate(fin, fbase, fout)

    '''
    get info
    '''
    # fin = 'file/4negative/2pair_drop_positiveAndrepeate.list'
    # fout = 'file/4negative/3pair_info_test.list'
    # getPairInfo(fin,fout,sep=',')

    '''
    get qualified tmp nontmp pair
    and fasta
    
    save 92766 pair
    save 131959 protein fasta
    save 68525 tmp 
    save 63434 nontmp 
    '''
    # fin = 'file/4negative/3pair_info.list'
    # fout = 'file/4negative/4pair_info_qualified.txt'
    # fout_fasta = 'file/4pair.fasta'
    # fout_tmp_fasta = 'file/4tmp.fasta'
    # fout_nontmp_fasta ='file/4nontmp.fasta'
    # extractPairAndFasta(fin, fout, fout_fasta,fout_tmp_fasta=fout_tmp_fasta,fout_nontmp_fasta=fout_nontmp_fasta)

    '''
    drop same subcellular
    '''



    '''
    get qualified tmp nontmp negative pair
    and fasta
    '''

    '''
    save to mongodb
    '''


















    # countline(fin,rename=False)
    # countline(fout,rename=False)
    # countline('file/1intAct_pair_norepeat.txt',rename=False)
    # countline(r'E:\data\intact\pair_norepeat.txt',rename=False)
    # countline(r'E:\data\intact\pair.txt',rename=False)
    # countline(r'E:\data\intact\intact.txt',rename=False)
    # countline(r'E:\data\intact\intact_negative.txt',rename=False)


    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)
    print('time', (time.time() - start)/60/60)


