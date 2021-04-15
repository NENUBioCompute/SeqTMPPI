# Title     : _10humanTrain.py
# Created by: julse@qq.com
# Created on: 2021/3/18 9:33
# des : TODO
import os
import time


# from FastaDealear import FastaDealer
# from FeatureDealer import BaseFeature, Feature_type
# from PairDealer import ComposeData, PairDealer
from _10humanTrain_suppGPU import _5train
# from _10humanTrain_support import _4getFeature
# from common import check_path
# from entry import entry
# from myModel import Param
#
# # def humanTrain(fin_p,fin_n,fin_fasta,limit,f1out,f2resultOut,dir_feature_db,dirout_feature):
# from mySupport import plot_result


def humanTrain(modelreuse=False):
    '''
    config path
    '''
    fin_p = 'file/10humanTrain/3cluster/4posi.tsv'
    fin_n = 'file/10humanTrain/3cluster/4nega.tsv'
    fin_fasta = 'file/10humanTrain/3cluster/all/dirRelated/2pair.fasta'

    dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/129878/'
    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'

    # _4getFeature(fin_p, fin_n, fin_fasta, dir_feature_db, dirout_feature)

    flist = [fin_p, fin_n]
    ratios_pn = [1, 1]
    # limit = 100
    # limit = 64939 * 2
    limit = 44210 * 2 / 6

    f1out = 'file/10humanTrain/4train/group'
    f2resultOut = '/home/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain/group_reusemodel'
    '''
    1. ComposeData 
    0 group: load 7368 [0:7368] data from file/10humanTrain/3cluster/4posi.tsv
    0 group: load 7368 [0:7368] data from file/10humanTrain/3cluster/4nega.tsv
    ...
    4 group: load 7368 [29472:36840] data from file/10humanTrain/3cluster/4posi.tsv
    4 group: load 7368 [29472:36840] data from file/10humanTrain/3cluster/4nega.tsv
    '''
    # ComposeData().save(f1out, flist, ratios_pn, limit,groupcount=-1,repeate=False)

    '''
    2. divided dataset
    divide data to train and test

    save 11788 pair to file/10humanTrain/4train/group/0/train.txt
    save 1473 pair to file/10humanTrain/4train/group/0/validate.txt
    save 1473 pair to file/10humanTrain/4train/group/0/test.txt
    ...
    save 11788 pair to file/10humanTrain/4train/group/4/train.txt
    save 1473 pair to file/10humanTrain/4train/group/4/validate.txt
    save 1473 pair to file/10humanTrain/4train/group/4/test.txt
    '''
    oldfile = '0'
    for eachfile in os.listdir(f1out):
        if eachfile =='0':continue
        f2outdir = os.path.join(f1out, eachfile)
        fin_pair = os.path.join(f2outdir, 'all.txt')
        train = os.path.join(f2outdir, 'train.txt')
        validate = os.path.join(f2outdir, 'validate.txt')
        test = os.path.join(f2outdir, 'test.txt')
        ratios_tvt = [0.8, 0.1, 0.1]
        f3outs = [train, validate, test]
        # PairDealer().part(fin_pair,ratios_tvt,f3outs)
        fin_model = os.path.join(f2resultOut,oldfile,'_my_model.h5')
        if not os.access(fin_model,os.F_OK) or not modelreuse:fin_model=None
        _5train(f1out, eachfile,train,dirout_feature, f2resultOut, fin_model=fin_model)

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

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    import os
    import tensorflow as tf

    # gpu_id = '0,1,2,3'
    # # gpu_id = '6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=tf_config)
    #
    humanTrain(modelreuse=True)

    # fin = '/home/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain/group/1/_history_dict.txt'
    # outdir = '/home/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain/group/1/'
    # with open(fin) as fi:
    #     line = fi.readline()
    #     mydict = eval(line)
    #     plot_result(mydict, outdir)
    pass
    # countline(f2nega_info)
    # fin = 'file/5statistic/positive/11TmpSubcellularCount.tsv
    # df = pd.read_table(fin,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

