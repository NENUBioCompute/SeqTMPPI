# Title     : _8DIPPredict_support.py
# Created by: julse@qq.com
# Created on: 2021/3/22 9:46
# des : TODO

import os
import time
import pandas as pd

from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from PairDealer import ComposeData
from _10humanTrain_support import saveRelated, _2_1combineFasta
from _1positiveSample import handlePair
from common import countline, check_path
from dao import composeTMP_nonTMP, getPairInfo_TMP_nonTMP
from dataset import saveQualified, handleRow, saveDifferSubcell, calcuSubcell
from mySupport import savepredict
from negativeData import dropPositiveAndRepeate


def _1posiPair(dirDIP,dirout):
    '''
    posi tmp and nontmp

    for each
    save 1000 pair
    save 880 protein fasta file/8DIPPredict/data/Ecoli/2pair.fasta
    save 369 tmp  file/8DIPPredict/data/Ecoli/2tmp.fasta
    save 511 nontmp  file/8DIPPredict/data/Ecoli/2nontmp.fasta
    # fin = '/home/jjhnenu/data/PPI/release/otherdata/DIP/Ecoli/2Ecoli20170205_id_pair_12246.txt'
    # foutdir = 'file/8DIPPredict/data/Ecoli'
    # check_path(foutdir)
    # handlePair(foutdir,sep='\t',fin=fin,jumpStep=[5],keepOne=True)
    '''
    # dirDIP = '/home/jjhnenu/data/PPI/release/otherdata/DIP/'
    # dirout = 'file/8DIPPredict/data/'
    for eachdir in os.listdir(dirDIP):
        if eachdir not in ['Ecoli', 'Mus', 'Human', 'SC', 'HP']:continue
        currentdir = os.path.join(dirDIP,eachdir)
        for eachfile in os.listdir(currentdir):
            if 'id_pair' not in eachfile:continue
            fin = os.path.join(currentdir,eachfile)
            foutdir = os.path.join(dirout,eachdir)
            check_path(foutdir)
            handlePair(foutdir,sep='\t',fin=fin,jumpStep=[5],keepOne=True)
def _2negaPair(dirDIP,fpositive,outdir):
    '''
    nega tmp and nontmp

    dirout_related = os.path.join(foutdir, '4pairInfo_subcell_differ_related')

    '''
    # dirDIP = 'file/8DIPPredict/data/'
    # fpositive = 'file/3cluster/4posi.tsv'
    # outdir = 'file/8DIPPredict/data_nega/'
    for eachdir in os.listdir(dirDIP):
        currentdir = os.path.join(dirDIP, eachdir)

        ftmp = os.path.join(currentdir, '2tmp.list')
        fnontmp = os.path.join(currentdir, '2nontmp.list')
        fposi = os.path.join(currentdir, '2pair.tsv')

        foutdir = os.path.join(outdir, eachdir)

        f1pair = os.path.join(foutdir, '1pair.tsv')
        f2pair = os.path.join(foutdir, '2pair.tsv')
        f2pairInfo = os.path.join(foutdir, '2pairInfo.tsv')
        f3pairInfo = os.path.join(foutdir, '3pairInfo.tsv')
        f4pairInfo_subcell = os.path.join(foutdir, '4pairInfo_subcell.tsv')
        f4pairInfo_subcell_differ = os.path.join(foutdir, '4pairInfo_subcell_differ.tsv')
        dirout_related = os.path.join(foutdir, '4pairInfo_subcell_differ_related')
        check_path(dirout_related)

        df = pd.read_table(fposi, header=None)
        composeTMP_nonTMP(ftmp, fnontmp, f1pair, int(df.shape[0] * 1.5))

        dropPositiveAndRepeate(f1pair, fpositive, f2pair)
        dropPositiveAndRepeate(f1pair, fposi, f2pair)
        getPairInfo_TMP_nonTMP(f2pair, f2pairInfo, sep='\t', checkTMP=False, keepOne=True)
        saveQualified(f2pairInfo, f3pairInfo)
        handleRow(f3pairInfo, f4pairInfo_subcell, calcuSubcell)
        saveDifferSubcell(f4pairInfo_subcell, f4pairInfo_subcell_differ)
        saveRelated(f4pairInfo_subcell_differ, dirout_related)
def _3combine_posi_nega(eachdir):
    fposiInfo = 'file/8DIPPredict/data/%s/2tmp_nontmp_info_qualified.tsv' % eachdir
    fnegaInfo = 'file/8DIPPredict/data_nega/%s/4pairInfo_subcell_differ.tsv' % eachdir
    dirout = 'file/8DIPPredict/data_all/%s/' % eachdir
    _2_1combineFasta(fposiInfo, fnegaInfo, dirout)
def _4getFeature(fin_pair,fin_fasta,dir_feature_db,dirout_feature):
    # fin_pair = '%s/dirRelated/2pair.tsv'%dirout
    '''
    generate feature db
    '''
    print('generate feature db')
    fd = FastaDealer()
    fd.getNpy(fin_fasta, dir_feature_db)
    '''
    generate feature
    '''
    print('generate feature')
    BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)
def _5composeBanlancePair(eachdir):
    '''
    1. ComposeData
    0 group: load 1000 [0:1000] data from file/8DIPPredict/data/Ecoli/2pair.tsv
    0 group: load 1000 [0:1000] data from file/8DIPPredict/data_nega/Ecoli/4pairInfo_subcell_differ_related/2pair.tsv
    ...
    '''

    fin_p = 'file/8DIPPredict/data/%s/2pair.tsv' % eachdir
    fin_n = 'file/8DIPPredict/data_nega/%s/4pairInfo_subcell_differ_related/2pair.tsv' % eachdir

    f1out = 'file/8DIPPredict/predict/%s' % eachdir

    flist = [fin_p, fin_n]
    ratios_pn = [1, 1]
    # limit = 100
    # limit = 64939 * 2
    limit = 0

    ComposeData().save(f1out, flist, ratios_pn, limit, groupcount=-1, repeate=False,labels=[1,0])
def _6predict(fin_pair,dirout_feature,fin_model,dirout_result,limit=0):
    '''
    :param eachdir:
    :return:
    '''

    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % eachdir
    #
    # fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/_my_model.h5'
    # dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/testDIP/%s' % eachdir

    # f1out = 'file/8DIPPredict/predict/%s' % eachdir
    # fin_pair = os.path.join(f1out, '0/all.txt')

    '''
    testing on the model
    '''
    print('testing the model')
    savepredict(fin_pair, dirout_feature, fin_model, dirout_result, batch_size=500,limit=limit)