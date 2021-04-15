# Title     : _8DIPPredict.py
# Created by: julse@qq.com
# Created on: 2021/3/19 9:12
# des : TODO

import os
import time
import pandas as pd

from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from _8DIPPredict_support import _6predict, _4getFeature

from mySupport import savepredict






def _5trainAndTest():

    pass





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


    for eachdir in ['Ecoli', 'Mus', 'Human', 'SC', 'HP']:
        '''
        fin
        '''
        fin_model = '/home/jjhnenu/data/PPI/release/2deployment0325/result/model/group0/_my_model.h5'
        fin_fasta = 'file/8DIPPredict/data_all/%s/dirRelated/2pair.fasta' % eachdir
        fin_pair = 'file/8DIPPredict/predict/%s/0/all.txt' % eachdir
        '''
        fout
        '''
        dirout_feature = '/home/jjhnenu/data/PPI/release/2deployment0325/feature/%s/' % eachdir
        dir_feature_db = '/home/jjhnenu/data/PPI/release/2deployment0325/featuredb/%s/' % eachdir
        dirout_result = '/home/jjhnenu/data/PPI/release/2deployment0325/result/model/group0/testDIP_PAN/%s' % eachdir

        # _4getFeature(fin_pair,fin_fasta,dir_feature_db,dirout_feature)

        # _6predict(fin_pair, dirout_feature, fin_model, dirout_result, limit=0)
        break

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
    pass
    # fin = 'file/8DIPPredict/data/Ecoli/2pair.tsv'
    # countline(fin)
    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

