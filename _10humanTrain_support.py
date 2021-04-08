# Title     : _10humanTrain_support.py
# Created by: julse@qq.com
# Created on: 2021/3/20 14:50
# des : TODO

import os
import pandas as pd

from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from PairDealer import concatPAN, PairDealer
from _3handleCluster import cluster2Table, pairWithClusterLable, saveRelated_posi_nega
from _5statistic import findSpecies, relatedSpecies, mergeTwo
from common import check_path, countline, concatFile
from dao import getSingleInfo
from dataset import extractPairAndFasta, saveQualified, simplifyTable, getproteinlist, handleRow, calcuSubcell, \
    saveDifferSubcell


def _1posiSampleHumanPair(fin_1posiInfo,dirout):
    '''
    positive samples human-human
    '''
    # fin_1posiInfo = 'file/1positive/2tmp_nontmp_info_qualified.tsv'
    # dirout = 'file/10humanTrain/1positive'
    check_path(dirout)

    f8tmp_species = os.path.join(dirout, '8tmp_species.tsv')
    f8nontmp_species = os.path.join(dirout, '8nontmp_species.tsv')
    f8species = os.path.join(dirout, '8species.tsv')
    f8sameSpecies = os.path.join(dirout, '8sameSpecies.tsv')
    f8posiSpecies = os.path.join(dirout, '8posiSpecies.tsv')

    f9human_related = os.path.join(dirout, '9human_related.tsv')
    f9human_human = os.path.join(dirout, '9human_human.tsv')  # 44210 pairinfo

    findSpecies(fin_1posiInfo,f8species, f8tmp_species, f8nontmp_species, f8sameSpecies,f8posiSpecies,col=[1,8])

    species = 'HUMAN'
    relatedSpecies(f8posiSpecies, species, f9human_related, f9human_human, col=[0,7,14,15])

    fout_fasta = os.path.join(dirout, '2pair.fasta')
    fout_tmp_fasta = os.path.join(dirout, '2tmp.fasta')
    fout_nontmp_fasta = os.path.join(dirout, '2nontmp.fasta')
    f2positive = os.path.join(dirout, '2pair.tsv')
    f2tmp = os.path.join(dirout, '2tmp.list')
    f2nontmp = os.path.join(dirout, '2nontmp.list')
    f2all = os.path.join(dirout, '2all.list')
    f2tmp_info = os.path.join(dirout, '2tmp_info.tsv')
    f2nontmp_info = os.path.join(dirout, '2nontmp_info.tsv')
    f2all_info = os.path.join(dirout, '2all_info.tsv')

    # save 11995 protein fasta file/10humanTrain/1positive/2pair.fasta
    # save 3513 tmp  file/10humanTrain/1positive/2tmp.fasta
    # save 8482 nontmp  file/10humanTrain/1positive/2nontmp.fasta

    simplifyTable(f9human_human, f2positive)

    extractPairAndFasta(f9human_human, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
                        fout_nontmp_fasta=fout_nontmp_fasta)
    getproteinlist(f9human_human,
                   ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
                   ftmp_info=f2tmp_info, ftmp_nontmp_info=f2nontmp_info, fall_info=f2all_info)
def _2negaSampleHumanPair(fin_1nega,dirout):
    '''
    negative samples human-human qualified
    '''
    # fin_1nega = 'file/8humanPredict/3pair_norepeat.tsv'  # 74968183
    # ## P14060	P31946
    # dirout = 'file/10humanTrain/2negative'

    check_path(dirout)
    f1nega_pair = os.path.join(dirout, '1nega_pair.tsv')

    f1tmpinfo = os.path.join(dirout, '1tmpinfo.tsv')
    f1nontmpinfo = os.path.join(dirout, '1nontmpinfo.tsv')

    f2nega_info = os.path.join(dirout, '2nega_info.tsv')
    f2nega_info_1 = os.path.join(dirout, '2nega_info_reindex.tsv')

    dirout = os.path.join(dirout, '4subcellular')
    check_path(dirout)
    f41subcelluar = os.path.join(dirout, '1subcelluar.tsv')
    f42subcelluar_differ = os.path.join(dirout, '2subcellular_differ.tsv')

    dirRelated = os.path.join(dirout, 'dirRelated')

    '''
    sample part 
    number of positive 44210
    (66315, 2) file/10humanTrain/2negative/1nega_pair.tsv
    stop 2021-03-19 15:38:28
    time 24.403409481048584
    '''
    nrows = int(44210 * 1.5)
    df = pd.read_table(fin_1nega).sample(nrows)
    df.to_csv(f1nega_pair, header=None, index=None, sep='\t')
    print(df.shape, f1nega_pair)

    '''
    tmp 5088
    nontmp 14585
    time 22.182501316070557
    '''
    getSingleInfo(f1nega_pair, f1tmpinfo, fin_type='pair',col=0)
    getSingleInfo(f1nega_pair, f1nontmpinfo, fin_type='pair',col=1)
    '''
    fin_1nega
    merge nega info
    left,right,merge:  (74968183, 2) (5088, 7) (74968183, 8)
    left,right,merge:  (74968183, 8) (14585, 7) (74164718, 14)
    stop 2021-03-19 16:50:25
    time 2973.5031571388245

    left,right,merge:  (66315, 2) (5088, 7) (66315, 8)
    left,right,merge:  (66315, 8) (14585, 7) (66315, 14)
    stop 2021-03-19 19:42:05
    time 3.276724100112915
    '''
    mergeTwo(f1nega_pair, f1tmpinfo, f2nega_info, left=[0,7], right=[x for x in range(8)],keepright=[x for x in range(7)])
    mergeTwo(f2nega_info, f1nontmpinfo, f2nega_info, left=[0,7,1,2,3,4,5,6], right=[x+7 for x in range(8)],keepright=[x+7 for x in range(7)])
    '''
    reindex
    '''
    left = [0, 7, 1, 2, 3, 4, 5, 6]
    right = [x + 7 for x in range(1,7)]
    df = pd.read_table(f2nega_info,header=None)
    left.extend(right)
    df.columns = left
    df = df.reindex(columns=[x for x in range(14)])
    df.to_csv(f2nega_info_1,header=None,index=None,sep='\t')
    '''
    fin_1nega
    4.drop same subcellular
    (41141, 19)
    stop 2021-03-19 14:06:49
    time 168.97262406349182

    drop same subcellular
    (66315, 19)
    stop 2021-03-19 19:47:26
    time 269.01450657844543

    f42subcelluar_differ : (61323, 19)

    '''
    handleRow(f2nega_info_1, f41subcelluar, calcuSubcell)
    saveDifferSubcell(f41subcelluar, f42subcelluar_differ)
    '''
    save related
    save 19538 protein fasta file/10humanTrain/2negative/4subcellular/dirRelated/2pair.fasta
    save 5087 tmp  file/10humanTrain/2negative/4subcellular/dirRelated/2tmp.fasta
    save 14451 nontmp  file/10humanTrain/2negative/4subcellular/dirRelated/2nontmp.fasta
    '''
    saveRelated(f42subcelluar_differ, dirRelated)
def _2_1combineFasta(fposiInfo,fnegaInfo,dirout):
    '''

    :param fposiInfo:
    :param fnegaInfo:
    :param dirout:
    :return:
    fin_pair = '%s/dirRelated/2pair.tsv'%dirout
    fasta = '%s/dirRelated/2pair.fasta'%dirout
    '''
    '''
    config path
    '''

    # fposiInfo = 'file/10humanTrain/1positive/9human_human.tsv'
    # fnegaInfo = 'file/10humanTrain/2negative/4subcellular/2subcellular_differ.tsv'
    #
    # dirout = 'file/10humanTrain/3cluster/all'
    check_path(dirout)

    fpairinfo = os.path.join(dirout, '1pairinfo.tsv')
    dirRelated = os.path.join(dirout, 'dirRelated')
    '''
    concat positive and negative with info
    '''
    df1 = pd.read_table(fposiInfo,header=None)
    df2 = pd.read_table(fnegaInfo,header=None)
    df3 = pd.concat([df1,df2])
    df3.to_csv(fpairinfo,header=None,index=None,sep='\t')
    '''
    save 19724 protein fasta file/10humanTrain/3cluster/all/dirRelated/2pair.fasta
    save 5089 tmp  file/10humanTrain/3cluster/all/dirRelated/2tmp.fasta
    save 14635 nontmp  file/10humanTrain/3cluster/all/dirRelated/2nontmp.fasta
    '''
    saveRelated(fpairinfo, dirRelated)
def _3clusters(fin_posi,fin_nega,fin_tmp,fin_nontmp,dirout):
    '''
    :param fin_posi:
    :param fin_nega:
    :param fin_tmp:
    :param fin_nontmp:
    :param dirout:
    :return:

    f4posi 'file/10humanTrain/3cluster/4posi.tsv'
    f4nega 'file/10humanTrain/3cluster/4nega.tsv'

    cd hit 0.4 cd-hit tool :http://weizhong-lab.ucsd.edu/cdhit_suite/cgi-bin/index.cgi?cmd=cd-hit
    get  *.clstr file
    '''
    # fin_posi = 'file/10humanTrain/1positive/2pair.tsv' # (44210, 2)
    # fin_nega = 'file/10humanTrain/2negative/4subcellular/dirRelated/2pair.tsv' # (61323, 2)
    # fin_tmp = 'file/10humanTrain/3cluster/1tmp.clstr'
    # fin_nontmp = 'file/10humanTrain/3cluster/1nontmp.clstr'
    # dirout = 'file/10humanTrain/3cluster/'

    check_path(dirout)
    f2pair = os.path.join(dirout,'1pair.tsv')
    f3out_tmp = os.path.join(dirout,'3tmp.tsv')
    f3out_nontmp = os.path.join(dirout,'3nontmp.tsv')
    f3pair = os.path.join(dirout,'3pair.tsv')
    f3pair_clstr = os.path.join(dirout,'3pair_clstr.tsv')
    f4pair = os.path.join(dirout,'4pair.tsv')
    f4posi = os.path.join(dirout,'4posi.tsv')
    f4nega = os.path.join(dirout,'4nega.tsv')  # (60697, 3)

    '''
    concat positive and negative pair
    '''
    concatPAN(fin_posi, fin_nega, f2pair)

    cluster2Table(fin_tmp,f3out_tmp)
    cluster2Table(fin_nontmp, f3out_nontmp)

    pairWithClusterLable(f2pair,f3out_tmp,f3out_nontmp,fout_clus=f3pair_clstr,fout=f3pair)

    '''
    extract positive,negative
    '''
    saveRelated_posi_nega(f3pair,f4pair,f4posi,f4nega)
def _4getFeature(fin_p,fin_n,fin_fasta,dir_feature_db,dirout_feature):
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
    BaseFeature().base_compose(dirout_feature, fin_p, dir_feature_db, feature_type=Feature_type.SEQ_1D)
    # stop 2021-03-20 15:03:28
    # time 171.11109900474548
    BaseFeature().base_compose(dirout_feature, fin_n, dir_feature_db, feature_type=Feature_type.SEQ_1D)

def getMassPairInfo(fin_pair,f1allinfo,fpairInfo):
    '''
    :param fin_pair:
    :param f1allinfo:
    :param fpairInfo:
    :return:
    '''
    # fin_pair = 'file/1positive/1tmp_nontmp.tsv'
    # f1allinfo = 'file2.0/1positive/1allinfo.tsv'
    getSingleInfo(fin_pair, f1allinfo, fin_type='pair')
    mergeTwo(fin_pair, f1allinfo, fpairInfo, left=[0,7], right=[x for x in range(8)],keepright=[x for x in range(7)])
    mergeTwo(fpairInfo, f1allinfo, fpairInfo, left=[0,7,1,2,3,4,5,6], right=[x+7 for x in range(8)],keepright=[x+7 for x in range(7)])
def saveRelated(fin_info,dirout):
    '''
    :param fin_info:
    :param dirout:
    :return:
     # fin_info = os.path.join(dirout, '2subcellular_differ.tsv')
    '''
    check_path(dirout)
    print('save related to',dirout)
    fout_fasta = os.path.join(dirout, '2pair.fasta')
    fout_tmp_fasta = os.path.join(dirout, '2tmp.fasta')
    fout_nontmp_fasta = os.path.join(dirout, '2nontmp.fasta')
    f2positive = os.path.join(dirout, '2pair.tsv')
    f2tmp = os.path.join(dirout, '2tmp.list')
    f2nontmp = os.path.join(dirout, '2nontmp.list')
    f2all = os.path.join(dirout, '2all.list')
    f2tmp_info = os.path.join(dirout, '2tmp_info.tsv')
    f2nontmp_info = os.path.join(dirout, '2nontmp_info.tsv')
    f2all_info = os.path.join(dirout, '2all_info.tsv')

    simplifyTable(fin_info, f2positive)

    extractPairAndFasta(fin_info, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
                        fout_nontmp_fasta=fout_nontmp_fasta)
    getproteinlist(fin_info,
                   ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
                   ftmp_info=f2tmp_info, ftmp_nontmp_info=f2nontmp_info, fall_info=f2all_info)
