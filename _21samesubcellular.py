# Title     : _21samesubcellular.py
# Created by: julse@qq.com
# Created on: 2021/7/28 15:06
# des : TODO



import time
import pandas as pd

from PairDealer import ComposeData, PairDealer
from negativeData import dropPositiveAndRepeate
from v3_train_zkd import crossTrain_fold, getFeature

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    '''
    1 delete none overlap in posi
    '''
    # fin_1subcellular = 'file/1positive/3subcellular/1subcellular.tsv' # (73784, 19)
    # f1posi_info = 'file/21samesubcellular/1posi_info.tsv' # (13049, 19)
    # f1posi = 'file/21samesubcellular/1posi.tsv' # (13049, 3)
    #
    # df = pd.read_table(fin_1subcellular,header=None)
    # df1 = df[df[18]==False]
    # df1.to_csv(f1posi_info,header=None,index=None,sep='\t')
    # df1[[0,7]].to_csv(f1posi,header=None,index=None,sep='\t')

    '''
    2 save pair in 3cluster drop similiar pair
    origin 13049,file/21samesubcellular/1posi.tsv
    delete reperate 2322,file/3cluster/4posi.tsv
    save 10727,file/21samesubcellular/2posi.tsv
    
    origin 86219,file/2negative/4subcellular/2pair.tsv
    delete reperate 1493,file/3cluster/4nega.tsv
    save 84726,file/21samesubcellular/2nega.tsv
    '''
    # f1posi = 'file/21samesubcellular/1posi.tsv'
    # f1nega = 'file/2negative/4subcellular/2pair.tsv'
    #
    # fclaster_nega = 'file/3cluster/4nega.tsv'
    # fclaster_posi = 'file/3cluster/4posi.tsv'
    # fout_nega = 'file/21samesubcellular/2nega.tsv'
    # fout_posi = 'file/21samesubcellular/2posi.tsv'
    # dropPositiveAndRepeate(f1posi, fclaster_posi, fout_posi,saverepeate=True)
    # dropPositiveAndRepeate(f1nega, fclaster_nega, fout_nega,saverepeate=True)
    '''
    3 compose posi and nega
    0 group: load 10727 data from file/21samesubcellular/2posi.tsv
    0 group: load 10727 [0:10727] data from file/21samesubcellular/2nega.tsv
    '''
    # fin_nega = 'file/21samesubcellular/2nega.tsv'
    # f1posi = 'file/21samesubcellular/2posi.tsv'
    # f2all = 'file/21samesubcellular/'
    # flist = [f1posi, fin_nega]
    # ratios = [0.5,0.5]
    # labels = [1, 0]
    # limit = 0
    # ComposeData().save(f2all, flist, ratios, limit, labels=labels)

    '''
    4. divided dataset
    divide data to vali_train and test
    save 19503 pair to file/21samesubcellular/0/vali_train.txt
    save 1950 pair to file/21samesubcellular/0/test.txt
    '''
    # f1_all = 'file/21samesubcellular/0/all.txt'
    # ratios_tvt = [10,1]
    # f2_vali = 'file/21samesubcellular/0/vali_train.txt'
    # f3_test = 'file/21samesubcellular/0/test.txt'
    # f3outs = [f2_vali,f3_test]
    # PairDealer().part(f1_all,ratios_tvt,f3outs)

    '''
    5 feature
    '''
    # fin_pair = 'file/21samesubcellular/0/all.txt'
    # fin_fasta = 'file/3cluster/1all.fasta'
    # eachdir ='benchmark_subcellu'
    # dir_feature_db = '/mnt/data/sunshiwei/Phsi_Blos/featuredb/%s/' % eachdir
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # getFeature(fin_pair, fin_fasta, dir_feature_db, dirout_feature)

    '''
    scikit learn 5 fold 
    '''
    # import os
    # import tensorflow as tf
    #
    # # gpu_id = '0,1,2,3'
    # gpu_id = '2,3'
    # # gpu_id = '4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    #
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    #
    # tf.compat.v1.Session(config=tf_config)
    #
    # fin_pair = 'file/21samesubcellular/0/vali_train.txt'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'benchmark_subcellu'
    # dirout = '/mnt/data/sunshiwei/Phsi_Blos_subcellu/result/%s/train' % 'benchmark_10_sklearn'
    # crossTrain_fold(fin_pair, dirout_feature, dirout, fold=5)

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

