# Title     : v3_predict.py
# Created by: julse@qq.com
# Created on: 2021/7/14 8:52
# des : 输入tmp和nontmp的fasta序列，随机组队
# tmp 5088 nontmp 14743

import numpy as np
import time
from Bio import SeqIO
from tensorflow.keras import models
# from keras import models

from ProteinDealer import Protein
from common import check_path
from myEvaluate import MyEvaluate
from v2_FastaDealear import FastaDealer
from v2_FeatureDealer import BaseFeature
def loaddata(ftmp_fa,fnontmp_fa):
    '''

    :param ftmp_fa:  tmp fasta file
    :param fnontmp_fa: nontmp fasta file
    :return: idpair, feature

    # ftmp_fa = 'file/8humanPredict/3tmp.fasta'
    # fnontmp_fa = 'file/8humanPredict/3nontmp.fasta'
    '''

    fd = FastaDealer()
    bf = BaseFeature()
    p = Protein()
    for pa in SeqIO.parse(ftmp_fa,'fasta'):
        a = fd.phsi_blos(pa.seq)
        if not p.checkProtein(pa.seq, 50, 2000, uncomm=True):continue
        for pb in SeqIO.parse(fnontmp_fa,'fasta'):
            b = fd.phsi_blos(pb.seq)
            if not p.checkProtein(pb.seq, 50, 2000, uncomm=True): continue
            c = bf.padding_PSSM(a,b,vstack=True,shape=(2000,25))
            yield '%s-%s'%(pa.id,pb.id),c
def loadmodel(fmodel):
    '''
    :param fmodel: well trained model
    :return:
    fmodel = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/5/_my_model.h5' % 'benchmark_human_10_sklearn'

    '''
    model = models.load_model(fmodel, custom_objects=MyEvaluate.metric_json)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=MyEvaluate.metric)
    model.summary()
    return model

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    # import os
    # import tensorflow as tf
    #
    # gpu_id = '0,1,2,3'
    # # gpu_id = '4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    #
    # tf.compat.v1.Session(config=tf_config)

    '''
    load the best model in 10 fold
    '''
    fmodel = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/5/_my_model.h5' % 'benchmark_human_10_sklearn'
    fout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/predict/5/human_predict.tsv' % 'benchmark_human_10_sklearn'
    f2out = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/predict/5/2human_predict.tsv' % 'benchmark_human_10_sklearn'
    # fmodel = '/home/jjhnenu/Data/zkd_phsi_blos_10_fold/benchmark_human_10_sklearn/train/5/_my_model.h5'
    # fout = '/home/jjhnenu/Data/zkd_phsi_blos_10_fold/benchmark_human_10_sklearn/predict/5/human_predict.tsv'
    check_path(fout[:fout.rindex('/')])
    ftmp_fa = 'file/8humanPredict/3tmp.fasta'
    fnontmp_fa = 'file/8humanPredict/3nontmp.fasta'


    limit = 500
    x_test = []
    ids = []
    model = loadmodel(fmodel)
    batch_size = 500
    with open(fout,'a') as fo:
        for idpair, feature in loaddata(ftmp_fa, fnontmp_fa):
            if len(ids)>=limit:
                # result_predict = model.predict(x_test)
                x_test = np.array(x_test)
                result_predict = model.predict(x_test, batch_size=batch_size)
                result_class = (result_predict > 0.5).astype("int32")
                result_predict = result_predict.reshape(-1)
                result_class = result_class.reshape(-1)
                for i in range(len(ids)):
                    fo.write('%s\t%d\t%f\n'%(ids[i],result_class[i],result_predict[i]))
                    fo.flush()
                x_test = []
                ids = []
            x_test.append(feature)
            ids.append((idpair))

    pass

    '''
    fo.write('%s\t%d\t%f\n'%(idpair,result_class[i],result_predict[i]))
    这一行 idpair 改为ids[i]
    '''
    with open(f2out,'w') as fo:
        with open(fout,'r') as fi:
            for idpair, feature in loaddata(ftmp_fa, fnontmp_fa):
                line = fi.readline()
                if not line:break
                _, result_class, result_predict = line[:-1].split('\t')
                fo.write('%s\t%s\t%s\n' % (idpair, result_class, result_predict))
                fo.flush()
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


