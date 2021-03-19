import os
import time
from common import  readIDlist
from dao import getPairInfo_TMP_nonTMP, save
from dataset import extractPairAndFasta, getproteinlist, simplifyTable, handleRow, calcuSubcell, saveQualified


def handlePair(foutdir,sep=',',dbname=None,
               checkTMP = True,jumpStep=None,
               fin = None,f2tmp_nonTtmp_info_qualified=None,keepOne=False):
    '''
    数据量较少，直接逐行查询，很多蛋白被查询了多次
    :param foutdir:
    :param sep: sep of fin file
    :parameter dbname: name of mongodb
    :parameter jumpStep: skip some step in this method [1,2,3,4]
    :parameter fin:ignore this parameter when 1 in jumpStep
    :parameter f2tmp_nonTtmp_info_qualified: sign this path in the dir

    :return:

    fin = 'file/1intAct_pair_norepeat.txt'
    foutdir = 'file/1positive'
    handlePair(fin,foutdir)
    '''

    '''
    config path
    '''
    f1tmp_nontmp_info = os.path.join(foutdir, '1tmp_nontmp_info.tsv')
    f1TMP_nonTMP = os.path.join(foutdir, '1tmp_nontmp.tsv')
    if not f2tmp_nonTtmp_info_qualified:f2tmp_nonTtmp_info_qualified = os.path.join(foutdir, '2tmp_nontmp_info_qualified.tsv')
    fout_fasta = os.path.join(foutdir, '2pair.fasta')
    fout_tmp_fasta = os.path.join(foutdir, '2tmp.fasta')
    fout_nontmp_fasta = os.path.join(foutdir, '2nontmp.fasta')
    f2positive = os.path.join(foutdir, '2pair.tsv')
    f2tmp = os.path.join(foutdir, '2tmp.list')
    f2nontmp = os.path.join(foutdir, '2nontmp.list')
    f2all = os.path.join(foutdir, '2all.list')
    f2tmp_info = os.path.join(foutdir, '2tmp_info.tsv')
    f2nontmp_info = os.path.join(foutdir, '2nontmp_info.tsv')
    f2all_info = os.path.join(foutdir, '2all_info.tsv')
    f3subcell = os.path.join(foutdir,'3subcellular.tsv')



    '''
    1. get tmp nontmp pair
    time 1766.4131457805634
    '''
    if jumpStep==None or 1 not in jumpStep:
        getPairInfo_TMP_nonTMP(fin, f1tmp_nontmp_info, sep=sep,checkTMP=checkTMP,keepOne=keepOne)
        simplifyTable(f1tmp_nontmp_info, f1TMP_nonTMP)
    '''
    2. get qualified tmp nontmp pair
    '''
    if jumpStep== None or 2 not in jumpStep:
        saveQualified(f1tmp_nontmp_info, f2tmp_nonTtmp_info_qualified)
    '''
    3. get related list,fasta,pair
    '''
    if jumpStep == None or 3 not in jumpStep:
        simplifyTable(f2tmp_nonTtmp_info_qualified, f2positive)

        extractPairAndFasta(f2tmp_nonTtmp_info_qualified, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
                            fout_nontmp_fasta=fout_nontmp_fasta)
        getproteinlist(f2tmp_nonTtmp_info_qualified,
                       ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
                       ftmp_info=f2tmp_info, ftmp_nontmp_info=f2nontmp_info, fall_info=f2all_info)
    '''
    4. save to mongodb
    '''
    if jumpStep ==None or (4 not in jumpStep and dbname):
        notsvaelist = save(readIDlist(f2all), dbname)
        print('those protein not save in the mongodb',notsvaelist)
    '''
    5. calcu subcellular
    '''
    if jumpStep ==None or 5 not in jumpStep:
        handleRow(f2tmp_nonTtmp_info_qualified, f3subcell,calcuSubcell)

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    '''
    save 73784 pair
    save 26440 protein fasta
    save 8744 tmp 
    save 17696 nontmp 
    not save P04439
    not save 1
    those protein not save in the mongodb ['P04439']
    stop 2021-01-22 22:06:35
    time 3022.90842795372
    '''
    # fin = 'file/1intAct_pair_norepeat.txt'
    # foutdir = 'file/1positive'
    # handlePair(foutdir,dbname='seqtmppi_positive',fin=fin)

    '''
    test 
    '''
    # fin = 'file/_1pair.txt'
    # foutdir = 'file/1test'
    # handlePair(foutdir,sep='\t',dbname='seqtmppi_positive_1',fin=fin)

    '''
    _test21357 
    '''
    # fin = 'file/_test21357/positive.txt'
    # foutdir = 'file/_test21357/qualified'
    # handlePair(foutdir,sep='\t',dbname='seqtmppi_positive_1',fin=fin)
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)