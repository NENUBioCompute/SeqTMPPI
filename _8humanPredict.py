# Title     : _8humanPredict.py
# Created by: julse@qq.com
# Created on: 2021/3/3 10:00
# des : TODO
import os
import time
import pandas as pd
from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
# from common import check_path, concatFile
# from dao import generateHumanLists, composeTMP_nonTMP, fullyComposeTMP_nonTMP, getPairInfo, getSingleInfo
# from dataset import saveQualified, extractPairAndFasta, simplifyTable, saveFasta
# from mySupport import savepredict
# from negativeData import dropPositiveAndRepeate
from PairDealer import PairDealer
from _5statistic import subcelluCount, calculateRatio, dropNul, mergeThree, getPDBPair, mergeTwo
from common import check_path
from dao import proteinPfam
from mySupport import savepredict

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    fpositive = 'file/1positive/2pair.tsv'
    dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/human/'
    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/human1/'
    fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/_my_model.h5'
    dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/testHuman01'

    fin_model_group0 = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/_my_model.h5'
    dirout_result_group0 = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/testHuman01'

    dirout = 'file/8humanPredict'
    f1tmp = os.path.join(dirout,'1tmp.list')
    f1nontmp = os.path.join(dirout,'1nontmp.list')

    f2tmpInfo = os.path.join(dirout,'2tmpInfo.tsv')
    f2nontmpInfo = os.path.join(dirout,'2nontmpInfo.tsv')

    f2tmp = os.path.join(dirout, '2tmp.list')
    f2nontmp = os.path.join(dirout, '2nontmp.list')

    f3pair = os.path.join(dirout,'3pair.tsv')

    f3pair_norepeat = os.path.join(dirout,'3pair_norepeat.tsv')

    f3pairInfo = os.path.join(dirout,'2pairInfo.tsv')
    f3tmp_fasta = os.path.join(dirout,'3tmp.fasta')
    f3nontmp_fasta = os.path.join(dirout,'3nontmp.fasta')
    f3all_fasta = os.path.join(dirout,'3all.fasta')

    dir3bathData = os.path.join(dirout,'3batchData')
    check_path(dir3bathData)
    f4sample1k = os.path.join(dir3bathData,'3batchData')


    '''
    get TMP, nonTMP list
    query 5208 tmp and 15186 nontmp
    '''
    # generateHumanLists(f1tmp, f1nontmp)
    '''
    get qualified single info
    '''
    # getSingleInfo(f1tmp,f1tmpInfo,fin_type='single')
    # getSingleInfo(f1nontmp,f1nontmpInfo,fin_type='single')
    '''
    get qualified protein
    '''
    # df = pd.read_table(f2tmpInfo, header=None)
    # df[0].to_csv(f2tmp,header=None,index=None,sep='\t')

    # df = pd.read_table(f2nontmpInfo, header=None)
    # df[0].to_csv(f2nontmp,header=None,index=None,sep='\t')
    '''
    get TMP_nonTMP pair
    time 367.5028109550476
    '''
    # fullyComposeTMP_nonTMP(f2tmp, f2nontmp, f3pair)
    '''
    drop repeat and positive
    '''
    # dropPositiveAndRepeate(f3pair, fpositive, f3pair_norepeat)

    '''
    get fasta and qualifeid pair
    '''
    # saveFasta(f2tmpInfo, f3tmp_fasta, AC=0, Seq=6)
    # saveFasta(f2nontmpInfo, f3nontmp_fasta, AC=0, Seq=6)
    # concatFile([f3tmp_fasta,f3nontmp_fasta],f3all_fasta)

    '''
    generate feature db
    '''
    print('generate feature db')
    # fd = FastaDealer()
    # fd.getNpy(f3all_fasta, dir_feature_db)
    '''
    generate feature
    '''
    print('generate feature')
    # pdealer = PairDealer()
    # pdealer.sample(f3pair_norepeat,10000,f4sample1k)
    # if os.path.exists(dirout_feature):os.removedirs(dirout_feature)
    # check_path(dirout_feature)
    # BaseFeature().base_compose(dirout_feature, f4sample1k, dir_feature_db, feature_type=Feature_type.SEQ_1D)
    print('end of delete feature')
    '''
    testing on the model
    '''
    # import os
    # import tensorflow as tf

    # gpu_id = '0,1,2,3'
    # gpu_id = '6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=tf_config)

    print('testing the model')
    # savepredict(f4sample1k, dirout_feature, fin_model, dirout_result,batch_size=500)
    '''
    predict human pair in group 0
    time 766.0980250835419
    '''
    # savepredict(f4sample1k, dirout_feature, fin_model_group0, dirout_result_group0,batch_size=500)


    # ##################### pdb on test human start ###########
    # fposi = 'file/8humanPredict/3batchData/3batchData'
    fposi = '/home/jjhnenu/SeqTMPPI20201226/file/8humanPredict/pdb/result_pdb_pair/result_queryPDB.csv'

    dirout = os.path.join(dirout,'pdb')
    check_path(dirout)
    f15TmpPDB = os.path.join(dirout, '15TmpPDB.tsv') #
    f15nonTmpPDB = os.path.join(dirout, '15nonTmpPDB.tsv') #
    f15TmpPDBCount = os.path.join(dirout, '15TmpPDBCount.tsv') # 11473
    f15nonTmpPDBCount = os.path.join(dirout, '15nonTmpPDBCount.tsv') # 47124
    f15TmpPDBRatio = os.path.join(dirout, '15TmpPDBRatio.tsv') # 11473
    f15nonTmpPDBRatio = os.path.join(dirout, '15nonTmpPDBRatio.tsv') # 47124

    f15TmpPDBnotNul = os.path.join(dirout, '15TmpPDBnotNul.tsv')  # (1873, 2)
    f15nonTmpPDBnotNul = os.path.join(dirout, '15nonTmpPDBnotNul.tsv')  # (6781, 2)

    f15Tmp_nonTMP_hasPDB = os.path.join(dirout, '15Tmp_nonTMP_hasPDB.tsv')  # (14544, 4)
    f15TmpPDB_nontmpPDB = os.path.join(dirout, '15TmpPDB_nontmpPDB.tsv')  # (6781, 2)
    f15TmpPDB_nontmpPDB_count = os.path.join(dirout, '15TmpPDB_nontmpPDB_count.tsv')  # (6781, 2)
    f15TmpPDB_nontmpPDB_ratio = os.path.join(dirout, '15TmpPDB_nontmpPDB_ratio.tsv')  # (6781, 2)

    dirout = os.path.join(dirout,'result_pdb_pair')
    check_path(dirout)

    fpdbpair_with_predict = os.path.join(dirout, 'pdbpair_with_predict.tsv')  # (6781, 2)
    fpdbpair_with_predict_sorted = os.path.join(dirout, 'pdbpair_with_predict_sorted.tsv')  # (6781, 2)



    # proteinPfam(f1tmp, f15TmpPDB, tophit=False, item='PDB')
    # proteinPfam(f1nontmp, f15nonTmpPDB, tophit=False, item='PDB')
    #
    # subcelluCount(f15TmpPDB, f15TmpPDBCount,1) # 11473
    # subcelluCount(f15nonTmpPDB, f15nonTmpPDBCount,1) # 47124
    #
    # calculateRatio(f15TmpPDBCount,f15TmpPDBRatio)
    # calculateRatio(f15nonTmpPDBCount,f15nonTmpPDBRatio)
    #
    # dropNul(f15TmpPDB, f15TmpPDBnotNul)     # (1873, 2)
    # dropNul(f15nonTmpPDB, f15nonTmpPDBnotNul) # (6781, 2)


    # mergeThree(fposi, f15TmpPDBnotNul, f15nonTmpPDBnotNul, f15Tmp_nonTMP_hasPDB,saveColumn=[0,1,4]) # (14544, 4)

    # getPDBPair(f15Tmp_nonTMP_hasPDB, f15TmpPDB_nontmpPDB)
    # mergeTwo(f15TmpPDB_nontmpPDB,fposi,fpdbpair_with_predict,left=[0,1,2,3],right=[0,1,2,3,4],keepleft=[0,1,2,3],keepright=[0,1,4])

    df = pd.read_table(fpdbpair_with_predict, header=None)
    df = df.sort_values(by=4,ascending=False)
    df.to_csv(fpdbpair_with_predict_sorted,header=None,index=None,sep='\t')
    # ##################### pdb on test human end ###########
    # df = pd.read_table(f15Tmp_nonTMP_hasPDB, header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


