import os
import time

from keras import models

from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from PairDealer import ComposeData, PairDealer
from common import check_path
from entry import entry
from myData import BaseData
from myEvaluate import MyEvaluate
from myModel import Param
from mySupport import savepredict, plot_result, calculateResults

def train():
    # time 664909.4274818897 ~ 7.6 day
    '''
    config path
    '''
    dirin = 'file/3cluster/'
    fin_p = os.path.join(dirin, '4posi.tsv')
    fin_n = os.path.join(dirin, '4nega.tsv')
    fin_fasta = os.path.join(dirin, '1all.fasta')
    flist = [fin_p, fin_n]
    ratios_pn = [1, 1]
    # limit = 100
    limit = 64939*2
    # limit = 64939*2/5


    f1out = 'file/4train/'
    f2outdir = os.path.join(f1out, str(0))
    fin_pair = os.path.join(f2outdir, 'all.txt')

    train = os.path.join(f2outdir, 'train.txt')
    validate = os.path.join(f2outdir, 'validate.txt')
    test = os.path.join(f2outdir, 'test.txt')
    ratios_tvt = [0.8, 0.1, 0.1]
    f3outs = [train, validate, test]

    dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/129878/'
    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'

    '''
    1. ComposeData 64939 * 2
    '''
    # ComposeData().save(f1out, flist, ratios_pn, limit,groupcount=1)
    '''
    2. divided dataset
    divide data to train and test
    save 103902 pair to file/4train/0/train.txt
    save 12987 pair to file/4train/0/validate.txt
    save 12987 pair to file/4train/0/test.txt
    '''
    # PairDealer().part(fin_pair,ratios_tvt,f3outs)

    '''
    generate feature db
    '''
    # print('generate feature db')
    # fd = FastaDealer()
    # fd.getNpy(fin_fasta, dir_feature_db)
    '''
    generate feature
    '''
    # print('generate feature')
    # BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)

    '''
    training the model
    '''
    print('training on the model')

    dir_in = dirout_feature
    dirout = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878'
    check_path(dirout)
    validate = {}
    validate['fin_pair'] = 'file/4train/0/validate.txt'
    validate['dir_in'] = dir_in
    onehot = True
    # entry(dirout, train, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
    #       # epochs=80,
    #       epochs=30,
    #       filters=300, batch_size=500, validate=validate)
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


def groupTrain():
    '''
    config path
    '''
    dirin = 'file/3cluster/'
    fin_p = os.path.join(dirin, '4posi.tsv')
    fin_n = os.path.join(dirin, '4nega.tsv')
    fin_fasta = os.path.join(dirin, '1all.fasta')
    flist = [fin_p, fin_n]
    ratios_pn = [1, 1]
    # limit = 100
    # limit = 64939 * 2
    limit = 64939*2/5

    f1out = 'file/4train/group'
    f2resultOut = '/home/19jjhnenu/Data/SeqTMPPI2W/result/group'

    dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/129878/'
    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'



    '''
    generate feature db
    '''
    # print('generate feature db')
    # fd = FastaDealer()
    # fd.getNpy(fin_fasta, dir_feature_db)
    '''
    generate feature
    '''
    # print('generate feature')
    # BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)
    # time 276.1376268863678
    # BaseFeature().base_compose(dirout_feature, fin_n, dir_feature_db, feature_type=Feature_type.SEQ_1D)

    '''
    1. ComposeData 64939 * 2 / 5 
    
    load 12987 [0:12987] data from file/3cluster/4posi.tsv
    load 12987 [0:12987] data from file/3cluster/4nega.tsv
    ...
    load 12987 [64935:77922] data from file/3cluster/4posi.tsv
    load 12987 [64935:77922] data from file/3cluster/4nega.tsv
    '''
    # ComposeData().save(f1out, flist, ratios_pn, limit,groupcount=-1,repeate=False)

    '''
    2. divided dataset
    divide data to train and test
    
    save 20779 pair to file/4train/group/0/train.txt
    save 2597 pair to file/4train/group/0/validate.txt
    save 2597 pair to file/4train/group/0/test.txt
    ...
    save 20779 pair to file/4train/group/4/train.txt
    save 2597 pair to file/4train/group/4/validate.txt
    save 2597 pair to file/4train/group/4/test.txt
    '''
    # for eachfile in os.listdir(f1out):
    #     f2outdir = os.path.join(f1out, eachfile)
    #     fin_pair = os.path.join(f2outdir, 'all.txt')
    #     train = os.path.join(f2outdir, 'train.txt')
    #     validate = os.path.join(f2outdir, 'validate.txt')
    #     test = os.path.join(f2outdir, 'test.txt')
    #     ratios_tvt = [0.8, 0.1, 0.1]
    #     f3outs = [train, validate, test]
    #     # PairDealer().part(fin_pair,ratios_tvt,f3outs)
    #     '''
    #     training the model
    #     '''
    #     print('training on the model')
    #
    #     dir_in = dirout_feature
    #     dirout = os.path.join(f2resultOut,eachfile)
    #     check_path(dirout)
    #     validate = {}
    #     validate['fin_pair'] = os.path.join(f1out,eachfile,'validate.txt')
    #     validate['dir_in'] = dir_in
    #     onehot = True
    #     entry(dirout, train, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
    #           epochs=80,
    #           # epochs=30,
    #           filters=300, batch_size=500, validate=validate)
    '''
    testing on the model
    time 1616.8053002357483
    '''
    print('testing the model')
    # for eachfile in os.listdir(f1out):
    #     fin_pair = os.path.join(f1out, eachfile,'test.txt')
    #     dir_in = dirout_feature
    #     dirout = os.path.join(f2resultOut,eachfile)
    #     fin_model = os.path.join(dirout,'_my_model.h5')
    #     dirout_result = os.path.join(dirout,'test')
    #     check_path(dirout_result)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)

    '''
    calculateResults
    '''
    # calculateResults(f2resultOut, f2resultOut, filename='test/log.txt', row=2, resultfilename='result.csv')
    '''
    plot result 
    '''
    # for idx in range(5):
    #     fin = os.path.join(f1out,'%d/_history_dict.txt'%idx)
    #     outdir = os.path.join(f2resultOut, idx)
    #     with open(fin) as fi:
    #         line = fi.readline()
    #         mydict = eval(line)
    #         plot_result(mydict, outdir)

# def exploreEpoch(fin_pair,dir_in,fin_model,batch_size):
#     check_path(dirout_result)
#     onehot = True
#     dataarray = BaseData().loadTest(fin_pair, dir_in, onehot=onehot, is_shuffle=False, limit=limit)
#     x_test, y_test = dataarray
#     model = models.load_model(fin_model, custom_objects=MyEvaluate.metric_json)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=MyEvaluate.metric)

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    # import os
    # import tensorflow as tf
    #
    # # gpu_id = '0,1,2,3'
    # gpu_id = '6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=tf_config)



    # train()

    # fin = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/_history_dict.txt'
    # outdir = '/home/19jjhnenu/Data/SeqTMPPI2W/result/129878/'
    # with open(fin) as fi:
    #     line = fi.readline()
    #     mydict = eval(line)
    #     plot_result(mydict, outdir)



    groupTrain()





    # import pandas as pd
    # group_dirin = 'file/4train/group/'
    # group_dirout = 'file/4train/statistic/group.tsv'
    # table= []
    # for eachfile in os.listdir(group_dirin):
    #     dirin = os.path.join(group_dirin,eachfile)
    #     for ef in ['train','validate','test','all']:
    #         row = []
    #         fin = os.path.join(dirin,'%s.txt'%ef)
    #         df = pd.read_table(fin,header=None)[2]
    #         row.extend([eachfile,ef])
    #         row.extend(list(df.value_counts()))
    #         row.extend([len(df)])
    #         table.append(row)
    # pd.DataFrame(table).to_csv(group_dirout,header=None,index=None,sep='\t')

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


