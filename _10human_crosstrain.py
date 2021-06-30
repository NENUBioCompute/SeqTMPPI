# Title     : _10human_crosstrain.py
# Created by: julse@qq.com
# Created on: 2021/5/2 22:05
# des : nohup /usr/bin/python _10human_crosstrain.py >0502_10human_cross
import os
import time


# from FastaDealear import FastaDealer
# from FeatureDealer import BaseFeature, Feature_type
# from PairDealer import ComposeData, PairDealer
from PairDealer import PairDealer
from _10humanTrain_suppGPU import _5train
# from _10humanTrain_support import _4getFeature
# from common import check_path
# from entry import entry
# from myModel import Param
#
# # def humanTrain(fin_p,fin_n,fin_fasta,limit,f1out,f2resultOut,dir_feature_db,dirout_feature):
# from mySupport import plot_result
from common import check_path, concatFile
from entry import entry
from myModel import Param
from mySupport import calculateResults


def crosshumanTrain(modelreuse=False):
    f2all = 'file/10humanTrain/3cluster/4pair.tsv'

    f1out = 'file/10humanTrain/4train/group'
    f2out = 'file/10humanTrain/4train/cross/group'
    f3out = 'file/10humanTrain/4train/cross'
    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    # f2resultOut = '/home/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain_80epoch/group_reusemodel_5CV'
    dirout_feature = '/root/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    f2resultOut = '/root/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain_80epoch/group_reusemodel_5CV'

    check_path(f2resultOut)
    check_path(f2out)

    # flist = [os.path.join(f1out,x,'all.txt') for x in os.listdir(f1out)]
    # concatFile(flist,f2all)
    '''
    train:test = 5:1
    '''
    train = os.path.join(f3out, 'train_vali.txt')
    test = os.path.join(f3out, 'test.txt')
    ratios_tvt = [5,1]
    f3outs = [train, test]
    # PairDealer().part(os.path.join(f3out,'all.txt'),ratios_tvt,f3outs)
    # PairDealer().part(f2all,ratios_tvt,f3outs)

    '''
    5cv
    '''
    ratios_tvt = [1]*5
    f3outs = [os.path.join(f2out,'%d.txt'%x) for x in range(5)]
    # PairDealer().part(os.path.join(f3out,'all.txt'),ratios_tvt,f3outs)
    # PairDealer().part(os.path.join(f3out,'train_vali.txt'),ratios_tvt,f3outs)
    '''
    cross train
    '''
    for cv in range(5):
        # oldfile = '-1'
        oldfile = '2'
        for elem in range(5):
            if elem == cv:continue
            if cv==0 and elem <3:continue

            f2dirout = os.path.join(f2resultOut,str(cv))
            f3dirout = os.path.join(f2dirout,str(elem))
            fin_model = os.path.join(f2dirout,oldfile,'_my_model.h5')
            if not os.access(fin_model,os.F_OK) or not modelreuse:fin_model=None
            train = os.path.join(f2out, '%d.txt'%elem)
            validate = {}
            validate['fin_pair'] = os.path.join(f2out, '%d.txt'%cv)
            validate['dir_in'] = dirout_feature
            onehot = True

            entry(f3dirout, train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
                  epochs=80,
                  # epochs=2,
                  filters=300, batch_size=500, validate=validate,
                  fin_model=fin_model)
            oldfile = str(elem)


def cvhumantrain(cv,f2out,dirout_feature,f2resultOut):
    modelreuse = True
    # f2out = 'file/10humanTrain/4train/cross/group'
    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    # f2resultOut = '/home/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain_80epoch/group_reusemodel_5CV'
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
import time

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    # import os
    # import tensorflow as tf
    # gpu_id = '1,2,3,4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=tf_config)

    crosshumanTrain(modelreuse=True)

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

