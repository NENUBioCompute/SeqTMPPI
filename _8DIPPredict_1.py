# Title     : _8DIPPredict.py
# Created by: julse@qq.com
# Created on: 2021/3/19 9:12
# des : TODO
import os
import time
from PairDealer import ComposeData, PairDealer
from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from common import check_path, concatFile
from entry import entry
from myModel import Param

from mySupport import savepredict, calculateResults


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
def _5composeBanlancePair(eachdir):
    '''
    1. ComposeData
    0 group: load 1000 [0:1000] data from file/8DIPPredict/data/Ecoli/2pair.tsv
    0 group: load 1000 [0:1000] data from file/8DIPPredict/data_nega/Ecoli/4pairInfo_subcell_differ_related/2pair.tsv
    ...
    '''

    fin_p = 'file/8DIPPredict/data/%s/2pair.tsv' % eachdir
    fin_n = 'file/8DIPPredict/data_nega/%s/4pairInfo_subcell_differ_related/2pair.tsv' % eachdir

    f1out = 'file/8DIPPredict/predict/%s' % eachdir

    flist = [fin_p, fin_n]
    ratios_pn = [1, 1]
    # limit = 100
    # limit = 64939 * 2
    limit = 0

    ComposeData().save(f1out, flist, ratios_pn, limit, groupcount=-1, repeate=False,labels=[1,0])
def _6predict(fin_pair,dirout_feature,fin_model,dirout_result,limit=0):
    '''
    Ecoli 数据集测试起来效果很差
    :param eachdir:
    :return:
    '''

    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % eachdir
    #
    # fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/_my_model.h5'
    # dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/testDIP/%s' % eachdir

    # f1out = 'file/8DIPPredict/predict/%s' % eachdir
    # fin_pair = os.path.join(f1out, '0/all.txt')

    '''
    testing on the model
    '''
    print('testing the model')
    savepredict(fin_pair, dirout_feature, fin_model, dirout_result, batch_size=500,limit=limit)


def _7dividedTrainAndTest(dir_data):
    '''
    config path
    '''
    fin_pair = os.path.join(dir_data, 'all.txt')

    train = os.path.join(dir_data, 'train.txt')
    validate = os.path.join(dir_data, 'validate.txt')
    test = os.path.join(dir_data, 'test.txt')
    ratios_tvt = [0.8, 0.1, 0.1]
    f3outs = [train, validate, test]

    '''
    2. divided dataset
    divide data to train and test

    '''
    print('divided dataset')
    PairDealer().part(fin_pair,ratios_tvt,f3outs)
def _7trainAndTest(dirout_feature,fin_train,fin_validate,dirout):
    # time 664909.4274818897 ~ 7.6 day

    '''
    training the model
    '''
    print('training on the model')
    check_path(dirout)
    validate = {}
    validate['fin_pair'] = fin_validate
    validate['dir_in'] = dirout_feature
    onehot = True
    entry(dirout, fin_train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
          epochs=80,
          # epochs=30,
          filters=300, batch_size=50, validate=validate)
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





if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    # import tensorflow as tf
    # # gpu_id = '0,1,2,3'
    # gpu_id = '6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=tf_config)


    # for eachdir in ['Ecoli', 'Mus', 'Human', 'SC', 'HP']:
    # for eachdir in ['Mus', 'Human']:
    #
    #     # '''
    #     # fin
    #     # '''
    #     # fin_model = '/home/jjhnenu/data/PPI/release/2deployment0325/result/model/group0/_my_model.h5'
    #     # fin_fasta = 'file/8DIPPredict/data_all/%s/dirRelated/2pair.fasta' % eachdir
    #     # fin_pair = 'file/8DIPPredict/predict/%s/0/all.txt' % eachdir
    #     # '''
    #     # fout
    #     # '''
    #     # dirout_feature = '/home/jjhnenu/data/PPI/release/2deployment0325/feature/%s/' % eachdir
    #     # dir_feature_db = '/home/jjhnenu/data/PPI/release/2deployment0325/featuredb/%s/' % eachdir
    #     # dirout_result = '/home/jjhnenu/data/PPI/release/2deployment0325/result/model/group0/testDIP_PAN/%s' % eachdir
    #     # 模型加载有问题
    #
    #     # '''
    #     # fin
    #     # '''
    #     # fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/_my_model.h5'
    #     # # fin_model_group0 = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/_my_model.h5'
    #     # fin_fasta = 'file/8DIPPredict/data_all/%s/dirRelated/2pair.fasta' % eachdir
    #     # fin_pair = 'file/8DIPPredict/predict/%s/0/all.txt' % eachdir
    #     # '''
    #     # fout
    #     # '''
    #     # dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/%s/' % eachdir
    #     dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % eachdir
    #     # dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/testDIP/%s' % eachdir
    #     # # dirout_result_group0 = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/testDIP/%s' % eachdir
    #     # # _4getFeature(fin_pair,fin_fasta,dir_feature_db,dirout_feature)
    #     # _6predict(fin_pair, dirout_feature, fin_model, dirout_result, limit=0)
    #
    #     # dir_data = 'file/8DIPPredict/predict/%s/0/' % eachdir
    #     # _7dividedTrainAndTest(dir_data)
    #
    #     fin_train = 'file/8DIPPredict/predict/%s/0/train.txt' % eachdir
    #     fin_validate = 'file/8DIPPredict/predict/%s/0/validate.txt' % eachdir
    #     dirout = '/home/19jjhnenu/Data/SeqTMPPI2W/result/DIP/%s' % eachdir
    #     _7trainAndTest(dirout_feature,fin_train,fin_validate,dirout)

    # table = []
    # for eachdir in ['Ecoli', 'Mus', 'Human', 'SC', 'HP']:
    #     dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/testDIP/%s' % eachdir
    #     fin_pair = os.path.join(dirout_result, 'result.csv')
    #     group_dirout = 'file/8DIPPredict/statistic/group.tsv'
    #     row = []
    #     df = pd.read_csv(fin_pair, header=None)[2]
    #     row.extend([eachdir])
    #     row.extend(list(df.value_counts()))
    #     row.extend([len(df)])
    #     table.append(row)
    # pd.DataFrame(table).to_csv(group_dirout, header=None, index=None, sep='\t')



    '''
    testing on the model
    '''

    print('testing the model')
    # for eachdir in ['Ecoli', 'Mus', 'Human', 'SC', 'HP']:
    #     fin_pair = 'file/8DIPPredict/data/%s/2pair.tsv'%eachdir
    #     dir_in = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % eachdir
    #     fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/_my_model.h5'
    #     dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/testDIP1/%s'%eachdir
    #     check_path(dirout_result)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result,batch_size=500,posi=True)

    # dirout = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group/0/testDIP1/'
    # calculateResults(dirout, dirout, filename='log.txt', row=2, resultfilename='result.csv')
    pass

    print('testing the model')
    # for eachdir in ['Ecoli', 'Mus', 'Human', 'SC', 'HP']:
    #
    #     fin_pair = 'file/8DIPPredict/data_all/%s/dirRelated/2pair.tsv'%eachdir
    #     dir_in = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % eachdir
    #     fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/5CV_1/2/4/_my_model.h5'
    #     dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/5CV_1/2/4/testDIP_all/%s'%eachdir
    #     check_path(dirout_result)
    #
    #     fin_fasta = 'file/8DIPPredict/data_all/%s/dirRelated/2pair.fasta' % eachdir
    #
    #     dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/%s/' % eachdir
    #     dirout_feature = dir_in
    #     _4getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result,batch_size=500,posi=True)
    #
    # dirout = '/home/19jjhnenu/Data/SeqTMPPI2W/result/5CV_1/2/4/testDIP_all/'
    # calculateResults(dirout, dirout, filename='log.txt', row=2, resultfilename='result.csv')






    # fin = 'file/8DIPPredict/data/Ecoli/2pair.tsv'
    # countline(fin)
    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')

    '''
    concat all kinds of species 正负样本 1：1
    HP 数据太少，只有26个正样本，抛弃
    '''
    # print('concat all kinds of species 正负样本 1：1')
    # dirin = 'file/8DIPPredict/predict'
    # fileList = [os.path.join(dirin,eachfile,'0/all.txt') for eachfile in ['Ecoli', 'Mus', 'Human', 'SC']]
    # fout = os.path.join(dirin,'all.txt')
    # concatFile(fileList, fout)

    '''concat fasta'''
    # print('concat all kinds of species  fasta')
    # dirin = 'file/8DIPPredict/data_all'
    # fileList = [os.path.join(dirin,eachfile,'dirRelated/2pair.fasta') for eachfile in ['Ecoli', 'Mus', 'Human', 'SC','HP']]
    # fout = os.path.join(dirin,'all.fasta')
    # concatFile(fileList, fout)

    '''
    feature 
    '''
    # fin_pair = 'file/8DIPPredict/predict/all.txt'
    # fin_fasta= 'file/8DIPPredict/data_all/all.fasta'
    # dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/%s/' % 'DIP'
    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/%s/' % 'DIP'
    # _4getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    '''
    concat DIP posi exclude HP
    '''
    # dirin = 'file/8DIPPredict/data'
    # fileList = [os.path.join(dirin,eachfile,'2pair.tsv') for eachfile in ['Ecoli', 'Mus', 'Human', 'SC']]
    # fout = os.path.join(dirin,'all.txt')
    # concatFile(fileList, fout)

    # dir_data = 'file/8DIPPredict/predict'
    # _7dividedTrainAndTest(dir_data)

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

