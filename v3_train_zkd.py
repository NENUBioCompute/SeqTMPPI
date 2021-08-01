# Title     : train.py
# Created by: julse@qq.com
# Created on: 2021/6/10 16:51
# des :  human ppi + Blso_pca for training
# 中科大服务器：replace /home/19jjhnenu/Data/Phsi_Blos/ to /mnt/data/sunshiwei/Phsi_Blos/
# scp -r Code/SeqTMPPI20201226/file/10humanTrain/4train/10cross/ sunshiwei@10.64.0.109:/home/sunshiwei/Code/SeqTMPPI20201226/file/10humanTrain/4train/
# scp Code/SeqTMPPI20201226/myModel.py sunshiwei@10.64.0.109:/home/sunshiwei/Code/SeqTMPPI20201226/myModel.py
# stop 2021-07-06 12:44:04
# time 33034.55173945427
# nohup python v3_train_zkd.py >0708_10humantrain
import time
import numpy as np
import os

from common import check_path
from entry import entry, entry_kfold
from myModel import Param
from mySupport import savepredict, calculateResults
from v2_FastaDealear import FastaDealer
from v2_FeatureDealer import BaseFeature, Feature_type


def getFeature(fin_pair,fin_fasta,dir_feature_db,dirout_feature):
    # fin_pair = '%s/dirRelated/2pair.tsv'%dirout
    '''
    generate feature db
    '''
    print('generate feature db')
    fd = FastaDealer()
    fd.getPhsi_Blos(fin_fasta, dir_feature_db)
    '''
    generate feature
    '''
    print('generate feature')
    BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.PHSI_BLOS,check_data=False)

def crossTrain_fold(fin_pair,dir_in,dirout,fold = 5,model_type=Param.CNN1D_OH):
        onehot = False
        entry_kfold(dirout, fin_pair, dir_in, model_type=model_type, limit=0, onehot=onehot, kernel_size=90,
                    epochs=80,
                  # epochs=2,
                  filters=300, batch_size=500, n_splits=fold,fin_model=None)
        print(dirout)
        calculateResults(dirout, dirout, resultfilename='result.csv')

        # eachdir = 'benchmark_human/train'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
        # print('testing the model on test dataset')
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # fin_test = 'file/4train/5CV/data/test.txt'
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_test/%d'%cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test,batch_size=500,limit=2000,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/predict
        '''
        # fin_test = 'file/8DIPPredict/predict/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/data
        '''
        # fin_test = 'file/8DIPPredict/data/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)


        '''
        testing on  all.txt in Imex
        '''
        # fin_test = 'file/8ImexPredict/4pair.tsv'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_IMEx_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        IMEx + - 
        '''
        # fin_test = 'file/8ImexPredict/predict/0/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_IMEx/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/data/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/data/%s/2pair.tsv'%eachfile
        #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP_%s/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/predict/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/predict/%s/0/all.txt'%eachfile
        #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP_%s_+-/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

def crossTrain(f2out,dirout_feature,f2resultOut,fold = 5,filename = '.txt'):

    '''
    cross train
    '''
    for cv in range(fold):
        oldfile = '-1'
        f2dirout = os.path.join(f2resultOut, str(cv))
        f3dirout=''
        for elem in range(fold):
            if cv == elem:continue
            f3dirout = os.path.join(f2dirout,str(elem))
            fin_model = os.path.join(f2dirout,oldfile,'_my_model.h5')
            train = os.path.join(f2out, '%d%s'%(elem,filename))
            validate = {}
            validate['fin_pair'] = os.path.join(f2out, '%d%s'%(cv,filename))
            validate['dir_in'] = dirout_feature
            onehot = False

            entry(f3dirout, train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
                  epochs=80,
                  # epochs=2,
                  filters=300, batch_size=500, validate=validate,
                  )

        #
        #
            oldfile = str(elem)
            print(f3dirout)
        calculateResults(f2dirout, f2dirout, resultfilename='result.csv')

        # eachdir = 'benchmark_human/train'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
        # print('testing the model on test dataset')
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # fin_test = 'file/4train/5CV/data/test.txt'
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_test/%d'%cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test,batch_size=500,limit=2000,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/predict
        '''
        # fin_test = 'file/8DIPPredict/predict/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/data
        '''
        # fin_test = 'file/8DIPPredict/data/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)


        '''
        testing on  all.txt in Imex
        '''
        # fin_test = 'file/8ImexPredict/4pair.tsv'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_IMEx_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        IMEx + - 
        '''
        # fin_test = 'file/8ImexPredict/predict/0/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_IMEx/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/data/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/data/%s/2pair.tsv'%eachfile
        #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP_%s/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/predict/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/predict/%s/0/all.txt'%eachfile
        #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/benchmark_human/test/5CV_1_DIP_%s_+-/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    '''
    data
    '''

    '''
    feature
    '''
    # fin_pair = 'file/10humanTrain/4train/cross/all.txt'
    # fin_pair = 'file/10humanTrain/3cluster/all/dirRelated/2pair.tsv'
    # fin_fasta = 'file/10humanTrain/3cluster/all/dirRelated/2pair.fasta'
    # eachdir ='benchmark_human'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    # fin_pair = 'file/4train/0/all.txt'
    # fin_fasta = 'file/3cluster/1all.fasta'
    # eachdir ='benchmark'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    # fin_pair = 'file/8DIPPredict/data/all.txt'
    # fin_fasta = 'file/8DIPPredict/data_all/all.fasta'
    # eachdir ='DIP'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    # fin_pair = 'file/8DIPPredict/predict/all.txt'
    # fin_fasta = 'file/8DIPPredict/data_all/all.fasta'
    # eachdir ='DIP'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    # for ef in ['Ecoli', 'Human', 'Mus', 'SC']:
    #     fin_pair = 'file/8DIPPredict/data_nega/%s/4pairInfo_subcell_differ_related/2pair.tsv'%ef
    #     fin_fasta = 'file/8DIPPredict/data_all/all.fasta'
    #     eachdir ='DIP'
    #     dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    #     getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    #
    # fin_pair = 'file/8ImexPredict/data_nega/2pair.tsv'
    # fin_fasta = 'file/8ImexPredict/data_nega/4pairInfo_subcell_differ_related/2pair.fasta'
    # eachdir = 'IMEx'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    #
    # fin_pair = 'file/8ImexPredict/data/2pair.tsv'
    # fin_fasta = 'file/8ImexPredict/data/2pair.fasta'
    # eachdir = 'IMEx'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    '''
    train
    '''
    # eachdir ='benchmark_human'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # dir_in = dirout_feature
    # eachdir = 'benchmark_human'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/' % eachdir
    # check_path(dirout)

    # onehot = True
    #
    import os
    import tensorflow as tf

    # gpu_id = '0,1,2,3'
    gpu_id = '1,2,3'
    # gpu_id = '4,5,6,7'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tf.compat.v1.Session(config=tf_config)


    # fin_pair = 'file/4train/0/train.txt'
    # # fin_pair = 'file/4train/0/test.txt'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/' % 'testall'
    # print('training model')
    # entry(dirout, fin_pair, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=3,
    #       epochs=80,
    #       # epochs=2,
    #       filters=300, batch_size=50, validate=validate,
    #       fin_model=None)

    print('cross train')
    '''
    5 fold cross train
    '''
    # f2out = 'file/10humanTrain/4train/cross/group/'
    # eachdir = 'benchmark_human'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % eachdir
    # crossTrain(f2out,dirout_feature, dirout, modelreuse=True)

    '''
    10 fold cross train
    '''

    # f2out = 'file/10humanTrain/4train/10cross/'
    # eachdir = 'benchmark_human_10'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % eachdir
    # check_path(dirout)
    # crossTrain(f2out,dirout_feature, dirout, modelreuse = True, fold = 10, filename = '/all.txt')

    '''
    10 fold cross train not repeat model
    '''

    # f2out = 'file/10humanTrain/4train/10cross/'
    # eachdir = 'benchmark_human_10'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % eachdir
    # check_path(dirout)
    # crossTrain(f2out,dirout_feature, dirout, modelreuse = True, fold = 10, filename = '/all.txt')

    # eachdir = 'benchmark_human_10_'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % eachdir
    # calculateResults(dirout, dirout, filename='9/_evaluate.txt')
    # 手动把9/_evaluate.txt 合并到result中


    '''
    scikit learn 5 fold 
    '''
    # fin_pair = 'file/10humanTrain/4train/10cross_1/train_validate.txt'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % 'benchmark_human_5_sklearn'
    # crossTrain_fold(fin_pair, dirout_feature, dirout, fold=5)

    '''
    scikit learn 10 fold 
    '''
    # fin_pair = 'file/10humanTrain/4train/10cross_1/train_validate.txt'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % 'benchmark_human_10_sklearn'
    # crossTrain_fold(fin_pair, dirout_feature, dirout, fold=10)

    '''
    transformer OOM when allocating tensor with shape[500,2,4000,4000]
    '''
    # fin_pair = 'file/10humanTrain/4train/10cross_1/train_validate.txt'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train' % 'benchmark_human_10_sklearn_transformer'
    # crossTrain_fold(fin_pair, dirout_feature, dirout, fold=10,model_type=Param.TRANSFORMER)

    '''
    testing on the 10 fold model independent test
    '''
    # fold = 10
    # fin_pair = 'file/10humanTrain/4train/10cross_1/test.txt'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # for idx in range(1,fold+1):
    #     fin_model = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/%d/_my_model.h5' % ('benchmark_human_10_sklearn', idx)
    #     dirout_result = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s/%d/' % ('benchmark_human_10_sklearn', 'independent',idx)
    #     savepredict(fin_pair, dirout_feature, fin_model, dirout_result,batch_size=500,onehot = False)
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s/' % (
    # 'benchmark_human_10_sklearn', 'independent')
    # calculateResults(dirout, dirout, filename='log.txt', row=2)
    '''
    testing on the 10 fold model independent DIP + 
    '''
    # fold = 10
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
    # for idx in range(1,fold+1):
    #     fin_pair = 'file/8DIPPredict/data/2all_noHP.txt'
    #     fin_model = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/%d/_my_model.h5' % ('benchmark_human_10_sklearn', idx)
    #     dirout_result = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s/%d/' % ('benchmark_human_10_sklearn', 'DIP',idx)
    #     savepredict(fin_pair, dirout_feature, fin_model, dirout_result,batch_size=500,onehot = False)
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s' % ('benchmark_human_10_sklearn', 'DIP')
    # calculateResults(dirout, dirout, filename='log.txt', row=2)

    '''
    testing on the 10 fold model independent DIP + each species
    '''
    # fold = 10
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
    # for idx in range(1,fold+1):
    #     for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
    #         fin_pair = 'file/8DIPPredict/data/%s/3pair.tsv'%eachfile
    #         fin_model = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/%d/_my_model.h5' % ('benchmark_human_10_sklearn', idx)
    #         dirout_result = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s_%s/%d/' % ('benchmark_human_10_sklearn', 'DIP',eachfile,idx)
    #         savepredict(fin_pair, dirout_feature, fin_model, dirout_result,batch_size=500,onehot = False)
    # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
    #     dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s_%s/' % (
    #         'benchmark_human_10_sklearn', 'DIP',eachfile)
    #     calculateResults(dirout, dirout, filename='log.txt', row=2)
    '''
    把目前得到的结果写到钉钉里面，计划也写到钉钉里面
    然后drugkb的项目搞一搞
    '''
    # fold = 10
    # fin_pair = 'file/10humanTrain/4train/10cross_1/test.txt'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_human'
    # for idx in range(fold):
    #     fin_model = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/%d/_my_model.h5' % ('benchmark_human_10_sklearn', idx)
    #     dirout_result = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/test/%s/%d/' % ('benchmark_human_10_sklearn', 'independent',idx)
    #     savepredict(fin_pair, dirout_feature, fin_model, dirout_result,batch_size=500,onehot = False)
    '''
    testing on the model
    '''
    print('testing the model')
    # fin_pair = 'file/4train/0/test.txt'
    # dir_in = dirout_feature
    # fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/_my_model.h5'
    # dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/test'
    # check_path(dirout_result)
    # savepredict(fin_pair, dir_in, fin_model, dirout_result,batch_size=500)

    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_test'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_IMEx_posi'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP_posi'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP'

    # dirA = ['5CV_1_test','5CV_1_IMEx_posi','5CV_1_DIP_posi']
    # dirB = ['5CV_1_DIP_%s'%x for x in ['Ecoli', 'Human', 'Mus', 'SC']]
    # dirA.extend(dirB)
    # for eachfile in dirA:
    #     dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s'%eachfile
    #     calculateResults(dirout,dirout,filename='log.txt',row=2)

    # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
    #     dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP_%s_+-/' % (eachfile)
    #     calculateResults(dirout, dirout, filename='log.txt', row=2)

    #
    # eachfile = '5CV_1_IMEx'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s' % eachfile
    # calculateResults(dirout,dirout,filename='log.txt',row=2)

    # data = np.load('/mnt/data/sunshiwei/Phsi_Blos/featuredb/benchmark/A0A0B4J1F4.npy')
    # data = np.load('/mnt/data/sunshiwei/Phsi_Blos/feature/benchmark/Q9UYS2_P58552.npy')
    # data = np.load('/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/Q9UYS2_P58552.npy')

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


