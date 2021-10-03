# Title     : v3_predict.py
# Created by: julse@qq.com
# Created on: 2021/7/14 8:52
# des : 输入tmp和nontmp的fasta序列，随机组队
# tmp 5088 nontmp 14743

import numpy as np
import pandas as pd
import time
from Bio import SeqIO
from tensorflow.keras import models
# from keras import models

from ProteinDealer import Protein
from common import check_path, concatFile
from myEvaluate import MyEvaluate
from v2_FastaDealear import FastaDealer
from v2_FeatureDealer import BaseFeature
# from _2negativeSample import dropPositiveAndRepeate

def dropPositiveAndRepeate(fin,fbase,fout,saverepeate=False,col=[0,1]):
    df = pd.read_csv(fin, sep='\t', header=None)[col]
    df_base = pd.read_csv(fbase, sep='\t', header=None)[col].drop_duplicates()
    # df_all = pd.concat([df_base,df]).drop_duplicates()
    df_concat = pd.concat([df_base,df])
    df_all = df_concat.drop_duplicates() if not saverepeate else df_concat[df_concat.duplicated()]
    if not saverepeate:
        df_all =df_concat.drop_duplicates() if not saverepeate else df_concat[df_concat.duplicated()]
        df_save = df_all.iloc[df_base.shape[0]:,:]
    else:
        df_save = df_concat[df_concat.duplicated()]
    df_save.to_csv(fout,header=None,index=None,sep='\t')
    print('origin %d,%s'%(df.shape[0],fin))
    print('reperate %d,%s'%(df.shape[0]-df_save.shape[0],fbase))
    print('save %d,%s'%(df_save.shape[0],fout))
    print()
    return df_all
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
    # flag = False
    # flagA = False
    # flagB = False
    for pa in SeqIO.parse(ftmp_fa,'fasta'):
        # if pa.id == 'Q8TCW7':flagA = True  # 程序中断之后，需要用这个代码控制起始预测位置
        # if not flagA:continue
        if not p.checkProtein(pa.seq, 50, 2000, uncomm=True):continue
        a = fd.phsi_blos(pa.seq)
        for pb in SeqIO.parse(fnontmp_fa,'fasta'):
            # if pb.id == 'Q15942':flagB = True
            # if not flagB:continue
            if not p.checkProtein(pb.seq, 50, 2000, uncomm=True): continue
            b = fd.phsi_blos(pb.seq)
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
def predict(fout,ftmp_fa,fnontmp_fa,fmodel):
    import os
    import tensorflow as tf

    gpu_id = '2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tf.compat.v1.Session(config=tf_config)

    # limit = 10000
    limit = 1
    x_test = []
    ids = []
    model = loadmodel(fmodel)
    batch_size = 500
    with open(fout,'a') as fo:
        for idpair, feature in loaddata(ftmp_fa, fnontmp_fa):
            x_test.append(feature)
            ids.append((idpair))
            if len(ids)>=limit:
                x_test = np.array(x_test)
                result_predict = model.predict(x_test, batch_size=batch_size)
                result_class = (result_predict > 0.5).astype("int32")
                result_predict = result_predict.reshape(-1)
                result_class = result_class.reshape(-1)
                for i in range(len(ids)):
                    fo.write('%s\t%d\t%f\n'%(ids[i],result_class[i],result_predict[i]))
                    fo.flush()
                del x_test,result_predict,result_class,ids
                x_test = []
                ids = []
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    '''
    predict human pair from fasta file when
    loading the best model(5th) in 10 fold 
    '''
    # fmodel = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/train/5/_my_model.h5' % 'benchmark_human_10_sklearn'
    # fout = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/predict/5/secondtry/4human_predict.tsv' % 'benchmark_human_10_sklearn'
    # # fmodel = '/home/jjhnenu/Data/zkd_phsi_blos_10_fold/benchmark_human_10_sklearn/train/5/_my_model.h5'
    # # fout = '/home/jjhnenu/Data/zkd_phsi_blos_10_fold/benchmark_human_10_sklearn/predict/5/human_predict.tsv'
    # check_path(fout[:fout.rindex('/')])
    # ftmp_fa = 'file/8humanPredict/3tmp.fasta'
    # fnontmp_fa = 'file/8humanPredict/3nontmp.fasta'
    # predict(fout,ftmp_fa,fnontmp_fa,fmodel)

    '''
    cat *.tsv >allsorted.tsv 合并文件
    sorted human protein pair
    '''
    # import pandas as pd
    # basedir = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/predict/5/secondtry/' % 'benchmark_human_10_sklearn'
    # fin = basedir+'human_predict_all.tsv'
    # f2out = basedir+'human_predict_all_sorted.tsv'
    # df = pd.read_table(fin, header=None)
    # df.sort_values(by=2, ascending=False, inplace=True)
    # df.to_csv(f2out,header=None,index=None,sep='\t')

    '''
    删除三个重复的 蛋白断点
    '''
    # import pandas as pd
    # basedir = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/predict/5/secondtry/' % 'benchmark_human_10_sklearn'
    # fin = basedir+'human_predict_all_sorted.tsv'
    # fout = basedir+'human_predict_all_sorted_2.tsv'
    # df = pd.read_table(fin, header=None) # (75012387, 3)
    # df.drop(df[df[0] == 'A6NF34-Q9BWW8'].index[0],inplace=True)  # (75012386, 3)
    # df.drop(df[df[0] == 'Q8NG11-Q96N22'].index[0], inplace=True) # (75012385, 3)
    # df.drop(df[df[0] == 'Q8TCW7-P20226'].index[0], inplace=True) # (75012384, 3)
    # df.to_csv(fout, sep='\t', header=None, index=None)

    '''
    删除在正样本中重复的蛋白质 todo 为什么会没有重复的呢？
    是因为蛋白表述的问题 一个是A\tB 一个是 A-B 
    P03372-P0CG48	1	0.999997
    
    '''
    # f1pair = '/mnt/data/sunshiwei/Phsi_Blos/result/%s/predict/5/secondtry/' % 'benchmark_human_10_sklearn'
    f1pair = '/home/jiangjiuhong/Data/Phsi_Blos/result/benchmark_human_10_sklearn/predict/5/secondtry/human_predict_all_sorted_2.tsv'
    fpositive = 'file/10humanTrain/3cluster/4posi_join.tsv'
    f2pair = '/home/jiangjiuhong/Data/Phsi_Blos/result/benchmark_human_10_sklearn/predict/5/secondtry/human_predict_all_sorted_3.tsv'
    dropPositiveAndRepeate(f1pair, fpositive, f2pair)


    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


