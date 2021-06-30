# Title     : train.py
# Created by: julse@qq.com
# Created on: 2021/6/10 16:51
# des : TODO


# 五折交叉验证需要花费的时间
# start 2021-06-22 23:14:11
# stop 2021-06-25 04:34:07
# time 191995.84152054787 约 2.2221643518518515 天



import time
import numpy as np
import os

from common import check_path
from entry import entry
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

def crossTrain(dirout_feature,f2resultOut,modelreuse=True):

    '''
    cross train and test
    '''
    f1out = 'file/4train/'
    f2outdir = os.path.join(f1out, str(0))
    fin_pair = os.path.join(f2outdir, 'all.txt')

    f2outdir = os.path.join(f1out, '5CV','data')
    check_path(f2outdir)
    train = os.path.join(f2outdir, 'train_vali.txt')
    test = os.path.join(f2outdir, 'test.txt')
    ratios_tvt = [5,1]
    f3outs = [train, test]
    # PairDealer().part(fin_pair,ratios_tvt,f3outs)
    '''
    train model
    '''
    f2outdir = os.path.join(f1out, '5CV','data')
    check_path(f2outdir)
    train = os.path.join(f2outdir, 'train_vali.txt')
    f2out = 'file/4train/5CV/elem'
    ratios_tvt = [1] * 5
    f3outs = [os.path.join(f2out,'%d.txt'%x) for x in range(5)]
    # PairDealer().part(train,ratios_tvt,f3outs)
    limit = 0
    # eachdir = 'benchmark'
    # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    # f2resultOut = '/home/19jjhnenu/Data/SeqTMPPI2W/result/5CV_1'
    '''
    cross train
    '''
    for cv in range(5):
        oldfile = '-1'
        f2dirout = os.path.join(f2resultOut, str(cv))
        fin_model = ''
        f3dirout=''
        for elem in range(5):
            if cv == elem:continue
            f3dirout = os.path.join(f2dirout,str(elem))
            fin_model = os.path.join(f2dirout,oldfile,'_my_model.h5')
            if not os.access(fin_model,os.F_OK) or not modelreuse:fin_model=None
            train = os.path.join(f2out, '%d.txt'%elem)
            validate = {}
            validate['fin_pair'] = os.path.join(f2out, '%d.txt'%cv)
            validate['dir_in'] = dirout_feature
            onehot = False

            # entry(f3dirout, train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
            #       epochs=80,
            #       # epochs=2,
            #       filters=300, batch_size=500, validate=validate,
            #       fin_model=fin_model)

            entry(f3dirout, train, dirout_feature, model_type=Param.TRANSFORMER, limit=10, onehot=onehot, kernel_size=90,
                  # epochs=80,
                  epochs=2,
                  filters=300, batch_size=500, validate=validate,
                  fin_model=fin_model)
        #
        #
            oldfile = str(elem)
        #     # print(f3dirout)
        # calculateResults(f2dirout, f2dirout, resultfilename='result.csv')

        # eachdir = 'benchmark'
        # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
        # print('testing the model on test dataset')
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # fin_test = 'file/4train/5CV/data/test.txt'
        # dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_test/%d'%cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test,batch_size=500,limit=2000,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/predict
        '''
        # fin_test = 'file/8DIPPredict/predict/all.txt'
        # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/data
        '''
        # fin_test = 'file/8DIPPredict/data/all.txt'
        # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)


        '''
        testing on  all.txt in Imex
        '''
        # fin_test = 'file/8ImexPredict/4pair.tsv'
        # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_IMEx_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        IMEx + - 
        '''
        # fin_test = 'file/8ImexPredict/predict/0/all.txt'
        # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_IMEx/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/data/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/data/%s/2pair.tsv'%eachfile
        #     dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_%s/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/predict/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/predict/%s/0/all.txt'%eachfile
        #     dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_%s_+-/%d' % (eachfile,cv)
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
    # fin_pair = 'file/4train/0/all.txt'
    # fin_fasta = 'file/3cluster/1all.fasta'
    # eachdir ='benchmark'
    # dir_feature_db = '/home/19jjhnenu/Data/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    fin_pair = 'file/8DIPPredict/data/all.txt'
    fin_fasta = 'file/8DIPPredict/data_all/all.fasta'
    eachdir ='DIP'
    dir_feature_db = '/home/19jjhnenu/Data/Phsi_Blos/featuredb/%s/' % eachdir
    dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    fin_pair = 'file/8DIPPredict/predict/all.txt'
    fin_fasta = 'file/8DIPPredict/data_all/all.fasta'
    eachdir ='DIP'
    dir_feature_db = '/home/19jjhnenu/Data/Phsi_Blos/featuredb/%s/' % eachdir
    dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    # for ef in ['Ecoli', 'Human', 'Mus', 'SC']:
    #     fin_pair = 'file/8DIPPredict/data_nega/%s/4pairInfo_subcell_differ_related/2pair.tsv'%ef
    #     fin_fasta = 'file/8DIPPredict/data_all/all.fasta'
    #     eachdir ='DIP'
    #     dir_feature_db = '/home/19jjhnenu/Data/Phsi_Blos/featuredb/%s/' % eachdir
    #     dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    #     getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    #
    # fin_pair = 'file/8ImexPredict/data_nega/2pair.tsv'
    # fin_fasta = 'file/8ImexPredict/data_nega/4pairInfo_subcell_differ_related/2pair.fasta'
    # eachdir = 'IMEx'
    # dir_feature_db = '/home/19jjhnenu/Data/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    # # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    #
    # fin_pair = 'file/8ImexPredict/data/2pair.tsv'
    # fin_fasta = 'file/8ImexPredict/data/2pair.fasta'
    # eachdir = 'IMEx'
    # dir_feature_db = '/home/19jjhnenu/Data/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    # # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    '''
    train
    '''
    eachdir ='benchmark'
    dirout_feature = '/home/19jjhnenu/Data/Phsi_Blos/feature/%s/' % eachdir
    dir_in = dirout_feature
    eachdir = 'benchmark'
    dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/%s/' % eachdir
    check_path(dirout)
    validate = {}
    validate['fin_pair'] = 'file/4train/0/validate.txt'
    validate['dir_in'] = dir_in
    onehot = False

    import os
    import tensorflow as tf

    # gpu_id = '0,1,2,3'
    gpu_id = '4,5,6,7'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tf.compat.v1.Session(config=tf_config)


    # fin_pair = 'file/4train/0/train.txt'
    # fin_pair = 'file/4train/0/test.txt'

    print('training model')
    # entry(dirout, fin_pair, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=3,
    #       epochs=80,
    #       # epochs=2,
    #       filters=300, batch_size=50, validate=validate,
    #       fin_model=None)

    print('cross train')
    crossTrain(dirout_feature, dirout, modelreuse=True)
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

    # dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_test'
    # dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_IMEx_posi'
    # dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_posi'
    # dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP'

    # dirA = ['5CV_1_test','5CV_1_IMEx_posi','5CV_1_DIP_posi']
    # dirB = ['5CV_1_DIP_%s'%x for x in ['Ecoli', 'Human', 'Mus', 'SC']]
    # dirA.extend(dirB)
    # for eachfile in dirA:
    #     dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/%s'%eachfile
    #     calculateResults(dirout,dirout,filename='log.txt',row=2)

    # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
    #     dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_%s_+-/' % (eachfile)
    #     calculateResults(dirout, dirout, filename='log.txt', row=2)

    #
    # eachfile = '5CV_1_IMEx'
    # dirout = '/home/19jjhnenu/Data/Phsi_Blos/result/%s' % eachfile
    # calculateResults(dirout,dirout,filename='log.txt',row=2)

    # data = np.load('/home/19jjhnenu/Data/Phsi_Blos/featuredb/benchmark/A0A0B4J1F4.npy')
    # data = np.load('/home/19jjhnenu/Data/Phsi_Blos/feature/benchmark/Q9UYS2_P58552.npy')
    # data = np.load('/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/Q9UYS2_P58552.npy')

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

