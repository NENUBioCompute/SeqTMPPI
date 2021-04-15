# encoding: utf-8
"""
@author: julse@qq.com
@time: 2021/4/15 16:56
@desc:
"""
from keras import models

from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from _8DIPPredict_support import _6predict
from common import check_path
from myData import BaseData
from myEvaluate import MyEvaluate
import pandas as pd
import os
import numpy as np
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

def savepredict(fin_pair,dir_in,fin_model,dirout_result,batch_size=90,limit=0):
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/1/0/test.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/test'
    check_path(dirout_result)
    print('predict ',fin_pair,'...')
    print('save result in ',dirout_result)
    df = pd.read_table(fin_pair, header=None)
    if df.shape[1]!=3:df[2]=0
    onehot = True
    dataarray = BaseData().loadTest(fin_pair, dir_in,onehot=onehot,is_shuffle=False,limit=limit)
    x_test, y_test =dataarray
    print('load model...')
    # model = load_model(fin_model, custom_objects=MyEvaluate.metric_json)
    # model = models.load_model(fin_model, custom_objects=MyEvaluate.metric_json)
    model = models.load_model(fin_model)

    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=MyEvaluate.metric)
    result = model.evaluate(x_test, y_test, verbose=1,batch_size=batch_size)

    result_predict = model.predict(x_test,batch_size=batch_size)
    result_class = np.argmax(result_predict, axis=-1)
    result_predict = result_predict.reshape(-1)

    # result_class = model.predict_classes(x_test,batch_size=batch_size)
    # UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01.

    result_class = result_class.reshape(-1)

    # y_test = y_test.reshape(-1)

    print('Loss:%f,ACC:%f' % (result[0], result[1]))


    if limit!=0:df = df[:limit]

    df.columns = ['tmp', 'nontmp','real_label']
    # df.rename(columns={0: 'tmp', 1: 'nontmp'}, inplace=True)
    # df['real_label'] = list(y_test)
    df['predict_label'] = result_class
    df['predict'] = result_predict
    df.sort_values(by=['predict'],ascending=False).to_csv(os.path.join(dirout_result,'result.csv'),index=False)

    # result_manual = MyEvaluate().evaluate_manual(y_test, result_predict)
    # print('[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]')
    # print(result_manual)
    # print('[acc,precision,sensitivity,f1,mcc,aps,aucResults,specificity]')
    # result_manual2 =calculate_performance(len(x_test), y_test, result_class, result_predict)
    # print(result_manual2)

    with open(os.path.join(dirout_result,'log.txt'),'w') as fo:
        fo.write('test dataset %s\n'%fin_pair)
        fo.write('Loss:%f,ACC:%f\n' % (result[0], result[1]))
        fo.write('evaluate result:'+str(result)+'\n')
        fo.write('manual result:[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]\n')
        # fo.write('manual result:' + str(result_manual) + '\n')
        # fo.write('manual result2:[acc,precision,sensitivity,f1,mcc,aps,aucResults,specificity]\n')
        # fo.write('manual result2:'+str(result_manual2)+'\n')
        fo.flush()
def main():
    _4getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    savepredict(fin_pair, dirout_feature, fin_model, dirout_result, batch_size=500, limit=20)
if __name__ == '__main__':
    fin_model = '../2model/0/_my_model.h5'
    fin_fasta = '../3dataset/DIP_mus/2pair.fasta'
    fin_pair = '../3dataset/DIP_mus/0/all.txt'


    dirout_result = 'result'
    dir_feature_db = 'result/feadb'
    dirout_feature = 'result/feature'

    main()


