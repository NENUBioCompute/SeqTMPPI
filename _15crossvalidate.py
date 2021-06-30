# Title     : _15crossvalidate.py
# Created by: julse@qq.com
# Created on: 2021/4/30 15:52
# des : nohup /usr/bin/python _15crossvalidate.py >0506_15corssvalidate
import numpy as np


import time
import os

from PairDealer import PairDealer
from common import concatFile, check_path, countline
from entry import entry
from myData import BaseData
from myModel import Param, MyModel
from mySupport import calculateResults, savepredict


def crossTrain(modelreuse=True):

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
    dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    f2resultOut = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1'
    '''
    cross train
    '''
    for cv in range(5):
        oldfile = '-1'
        f2dirout = os.path.join(f2resultOut, str(cv))
        fin_model = ''
        f3dirout=''
        for elem in range(5):
            if elem == cv:continue
            f3dirout = os.path.join(f2dirout,str(elem))
            fin_model = os.path.join(f2dirout,oldfile,'_my_model.h5')
            if not os.access(fin_model,os.F_OK) or not modelreuse:fin_model=None
            train = os.path.join(f2out, '%d.txt'%elem)
            validate = {}
            validate['fin_pair'] = os.path.join(f2out, '%d.txt'%cv)
            validate['dir_in'] = dirout_feature
            onehot = True

            # entry(f3dirout, train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
            #       epochs=80,
            #       # epochs=2,
            #       filters=300, batch_size=500, validate=validate,
            #       fin_model=fin_model)


            oldfile = str(elem)
            # print(f3dirout)
        # calculateResults(f2dirout, f2dirout, resultfilename='result.csv')

        # print('testing the model on test dataset')
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # fin_test = 'file/4train/5CV/data/test.txt'
        # dirout_result_test = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1_test/%d'%cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test,batch_size=500)
        '''
        testing on DIP all.txt in DIP/predict
        '''

        # fin_test = 'file/8DIPPredict/predict/all.txt'
        # dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1_DIP/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500)

        '''
        testing on DIP all.txt in DIP/data
        '''
        # fin_test = 'file/8DIPPredict/data/all.txt'
        # dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1_DIP_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500)


        '''
        testing on  all.txt in Imex
        '''
        # fin_test = 'file/8ImexPredict/3pair.tsv'
        # dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1_IMEx_posi_5/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500)

        '''
        testing on  all.txt in Imex + - 
        '''
        # fin_test = 'file/8ImexPredict/predict/0/all.txt'
        # dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1_IMEx/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500)


def cvtrain(cv):
    modelreuse = True
    f2out = 'file/4train/5CV/elem'
    dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    f2resultOut = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV'

    # f2out = 'file/10humanTrain/4train/cross/group'
    # dirout_feature = '/home/jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    # f2resultOut = '/home/jjhnenu/Data/SeqTMPPI2W/result/10humanTrain_80epoch/group_reusemodel_5CV'
    '''
    cross train
    '''

    oldfile = '-1'
    for elem in range(5):
        if elem == cv: continue
        f2dirout = os.path.join(f2resultOut, str(cv))
        f3dirout = os.path.join(f2dirout, str(elem))
        fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        if not os.access(fin_model, os.F_OK) or not modelreuse: fin_model = None
        train = os.path.join(f2out, '%d.txt' % elem)
        validate = {}
        validate['fin_pair'] = os.path.join(f2out, '%d.txt' % cv)
        validate['dir_in'] = dirout_feature
        onehot = True

        entry(f3dirout, train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
              epochs=80,
              # epochs=2,
              filters=300, batch_size=500, validate=validate,
              fin_model=fin_model)

        oldfile = str(elem)


if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    # import os
    # import tensorflow as tf

    # gpu_id = '0,1,2,3'
    # gpu_id = '4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=tf_config)
    # crossTrain()
    # dirout = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1_IMEx'
    # calculateResults(dirout,dirout,filename='log.txt',row=2)

    # dirout = '/home/jjhnenu/Data/SeqTMPPI2W/result/5CV_1'
    # calculateResults(dirout, dirout, filename='4/_evaluate.txt')
    # 手动把4/_evaluate.txt 合并到result中
    pass



    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


