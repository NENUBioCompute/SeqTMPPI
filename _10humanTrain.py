# Title     : _10humanTrain.py
# Created by: julse@qq.com
# Created on: 2021/3/18 9:33
# des : TODO
import os
import time
import pandas as pd

from PairDealer import concatPAN
from _3handleCluster import cluster2Table, pairWithClusterLable, saveRelated
from _5statistic import findSpecies, relatedSpecies, mergeTwo
from common import check_path, countline, concatFile
from dataset import extractPairAndFasta, saveQualified, simplifyTable, getproteinlist

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()


    '''
    positive samples human-human
    '''
    fin_1posiInfo = 'file/1positive/2tmp_nontmp_info_qualified.tsv'
    dirout = 'file/10humanTrain/1positive'
    check_path(dirout)

    f8tmp_species = os.path.join(dirout, '8tmp_species.tsv')
    f8nontmp_species = os.path.join(dirout, '8nontmp_species.tsv')
    f8species = os.path.join(dirout, '8species.tsv')
    f8sameSpecies = os.path.join(dirout, '8sameSpecies.tsv')
    f8posiSpecies = os.path.join(dirout, '8posiSpecies.tsv')

    f9human_related = os.path.join(dirout, '9human_related.tsv')
    f9human_human = os.path.join(dirout, '9human_human.tsv') # 44210 pairinfo

    # findSpecies(fin_1posiInfo,f8species, f8tmp_species, f8nontmp_species, f8sameSpecies,f8posiSpecies,col=[1,8])

    # species = 'HUMAN'
    # relatedSpecies(f8posiSpecies, species, f9human_related, f9human_human, col=[0,7,14,15])

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

    # simplifyTable(f9human_human, f2positive)
    #
    # extractPairAndFasta(f9human_human, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
    #                     fout_nontmp_fasta=fout_nontmp_fasta)
    # getproteinlist(f9human_human,
    #                ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
    #                ftmp_info=f2tmp_info, ftmp_nontmp_info=f2nontmp_info, fall_info=f2all_info)
    '''
    negative samples human-human qualified
    '''
    fin_1nega = 'file/8humanPredict/3pair_norepeat.tsv' # 74968183

    dirout = 'file/8humanPredict'
    fin_1tmpInfo = os.path.join(dirout,'2tmpInfo.tsv')
    fin_1nontmpInfo = os.path.join(dirout,'2nontmpInfo.tsv')


    dirout = 'file/10humanTrain/2negative'
    check_path(dirout)
    f1qualified_info = os.path.join(dirout, '1qualified_info.tsv')
    f2qualified_info_sample = os.path.join(dirout, '1qualified_info_sample.tsv')
    f2qualified_info_sample_1 = os.path.join(dirout, '1qualified_info_sample_1.tsv')

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

    # left, right, merge: (74968183, 2)(5088, 7)(74968183, 8)
    # left, right, merge: (74968183, 8)(14743, 7)(74968183, 14)
    # time 3091.329487323761

    # mergeTwo(fin_1nega, fin_1tmpInfo, f1qualified_info, left=[0,7], right=[x for x in range(8)],keepright=[x for x in range(7)])
    # mergeTwo(f1qualified_info, fin_1nontmpInfo, f1qualified_info, left=[0,7,1,2,3,4,5,6], right=[x+7 for x in range(8)],keepright=[x+7 for x in range(7)])


    # sample 1.2 multiple than positive
    # time 1260.461941242218

    # nrows = int(44210 * 1.2)
    # df = pd.read_table(f1qualified_info).sample(nrows)
    # df.to_csv(f2qualified_info_sample,header=None,index=None,sep='\t')
    # print(df.shape,f2qualified_info_sample)




    # reindex

    # left = [0, 7, 1, 2, 3, 4, 5, 6]
    # right = [x + 7 for x in range(1,7)]
    # df = pd.read_table(f2qualified_info_sample,header=None)
    # left.extend(right)
    # df.columns = left
    # df = df.reindex(columns=[x for x in range(14)])
    # df.to_csv(f2qualified_info_sample_1,header=None,index=None,sep='\t')


    # save 19456 protein fasta file/10humanTrain/2negative/2pair.fasta
    # save 5088 tmp  file/10humanTrain/2negative/2tmp.fasta
    # save 14368 nontmp  file/10humanTrain/2negative/2nontmp.fasta

    # simplifyTable(f2qualified_info_sample_1, f2positive)
    #
    # extractPairAndFasta(f2qualified_info_sample_1, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
    #                     fout_nontmp_fasta=fout_nontmp_fasta)
    # getproteinlist(f2qualified_info_sample_1,
    #                ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
    #                ftmp_info=f2tmp_info, ftmp_nontmp_info=f2nontmp_info, fall_info=f2all_info)

    '''
    clusters cd-hit
    '''
    dirout = 'file/10humanTrain/1positive'
    check_path(dirout)
    f1List_tmp_fasta = ['file/10humanTrain/1positive/2tmp.fasta', 'file/10humanTrain/2negative/2tmp.fasta']
    f1List_nontmp_fasta = ['file/10humanTrain/1positive/2nontmp.fasta','file/10humanTrain/2negative/2nontmp.fasta']
    fin_posi = 'file/10humanTrain/1positive/2pair.tsv'
    fin_nega = 'file/10humanTrain/2negative/2pair.tsv'

    foutdir = 'file/10humanTrain/3cluster/'
    check_path(foutdir)
    f2pair = 'file/10humanTrain/3cluster/1all.tsv'
    f3out_tmp = 'file/10humanTrain/3cluster/3tmp.tsv'
    f3out_nontmp = 'file/10humanTrain/3cluster/3nontmp.tsv'
    f3pair = 'file/10humanTrain/3cluster/3pair.tsv'
    f3pair_clstr = 'file/10humanTrain/3cluster/3pair_clstr.tsv'
    f4pair = 'file/10humanTrain/3cluster/4pair.tsv'
    f4posi = 'file/10humanTrain/3cluster/4posi.tsv'
    f4nega = 'file/10humanTrain/3cluster/4nega.tsv'

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
    concat 2.0
    '''
    f9human_human = 'file/10humanTrain/1positive/9human_human.tsv'
    f2qualified_info_sample_1 ='file/10humanTrain/2negative/1qualified_info_sample_1.tsv'

    dirout = 'file/10humanTrain/3cluster/all'
    check_path(dirout)

    fpairinfo = os.path.join(dirout, '1pairinfo.tsv')

    df1 = pd.read_table(f9human_human,header=None)
    df2 = pd.read_table(f2qualified_info_sample_1,header=None)
    df3 = pd.concat([df1,df2])
    df3.to_csv(fpairinfo,header=None,index=None,sep='\t')

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


    # save 19678 protein fasta file/10humanTrain/3cluster/all/2pair.fasta
    # save 5089 tmp  file/10humanTrain/3cluster/all/2tmp.fasta
    # save 14589 nontmp  file/10humanTrain/3cluster/all/2nontmp.fasta

    # simplifyTable(fpairinfo, f2positive)
    #
    # extractPairAndFasta(fpairinfo, fout_fasta, fout_tmp_fasta=fout_tmp_fasta,
    #                     fout_nontmp_fasta=fout_nontmp_fasta)
    # getproteinlist(fpairinfo,
    #                ftmp=f2tmp, fnontmp=f2nontmp, fall=f2all,
    #                ftmp_info=f2tmp_info, ftmp_nontmp_info=f2nontmp_info, fall_info=f2all_info)





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

    # '''
    # combine P and N
    # '''
    #
    # dirin = 'file/3cluster/'
    # fin_p = os.path.join(dirin, '4posi.tsv')
    # fin_n = os.path.join(dirin, '4nega.tsv')
    # fin_fasta = os.path.join(dirin, '1all.fasta')
    # flist = [fin_p, fin_n]
    # ratios_pn = [1, 1]
    # limit = 64939 * 2
    # # limit = 64939*2/5
    #
    # f1out = 'file/4train/'
    # f2outdir = os.path.join(f1out, str(0))
    # fin_pair = os.path.join(f2outdir, 'all.txt')
    #
    # train = os.path.join(f2outdir, 'train.txt')
    # validate = os.path.join(f2outdir, 'validate.txt')
    # test = os.path.join(f2outdir, 'test.txt')
    # ratios_tvt = [0.8, 0.1, 0.1]
    # f3outs = [train, validate, test]
    #
    # dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/129878/'
    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'

    '''
    group train
    train based on the last model
    '''




    # countline(f3pair)
    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

