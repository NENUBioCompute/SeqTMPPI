# Title     : _8ImexPredict.py
# Created by: julse@qq.com
# Created on: 2021/3/22 19:36
# des : TODO

import time

# from _1positiveSample import handlePair
from PairDealer import ComposeData
from _8DIPPredict_1 import _4getFeature
from _8DIPPredict_support import composeNegaPair
from common import check_path
from negativeData import dropPositiveAndRepeate

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    fin = '/home/jjhnenu/data/PPI/stage2/uniprotPairFromImex.txt'
    foutdir = 'file/8ImexPredict'
    # check_path(foutdir)
    # handlePair(foutdir, sep='\t', fin=fin, jumpStep=[5], keepOne=True)

    ''''
    feature
    '''
    fin_pair = 'file/8ImexPredict/2pair.tsv'
    fin_fasta= 'file/8ImexPredict/2pair.fasta'
    dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/%s/' % 'IMEx'
    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'IMEx'
    # _4getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    '''
    drop positive
    '''
    f1pair = 'file/8ImexPredict/2pair.tsv'
    fpositive = 'file/4train/0/all.txt'
    f2pair = 'file/8ImexPredict/3pair.tsv'
    # dropPositiveAndRepeate(f1pair, fpositive, f2pair)

    f1pair = 'file/8ImexPredict/2pair.tsv'
    fpositive = 'file/4train/5CV/data/train_vali.txt'
    f2pair = 'file/8ImexPredict/4pair.tsv'
    # dropPositiveAndRepeate(f1pair, fpositive, f2pair)

    '''
    compose negative
    '''
    currentdir = 'file/8ImexPredict/data/'
    fpositive = 'file/3cluster/4posi.tsv'
    foutdir = 'file/8ImexPredict/data_nega/'
    check_path(foutdir)
    # composeNegaPair(currentdir, fpositive, foutdir)

    '''
    compose p n
    '''
    # fin_p = 'file/8ImexPredict/data/4pair.tsv'
    # fin_n = 'file/8ImexPredict/data_nega/4pairInfo_subcell_differ_related/2pair.tsv'
    # f1out = 'file/8ImexPredict/predict'
    # flist = [fin_p, fin_n]
    # ratios_pn = [1, 1]
    # limit = 0
    # ComposeData().save(f1out, flist, ratios_pn, limit, groupcount=-1, repeate=False, labels=[1, 0])

    '''
    nega feature
    '''
    fin_pair = 'file/8ImexPredict/data_nega/4pairInfo_subcell_differ_related/2pair.tsv'
    fin_fasta = 'file/8ImexPredict/data_nega/4pairInfo_subcell_differ_related/2pair.fasta'
    dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/%s/' % 'IMEx'
    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'IMEx'
    _4getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
pass
print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print('time', time.time() - start)

