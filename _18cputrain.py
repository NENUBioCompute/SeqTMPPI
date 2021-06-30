# Title     : _18cputrain.py
# Created by: julse@qq.com
# Created on: 2021/5/4 18:52
# des :

# nohup python3 _18cputrain.py >0504_feature


import time
import os
from FastaDealear import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    dirin = 'file/3cluster/'
    fin_fasta = os.path.join(dirin, '1all.fasta')

    dir_feature_db = '/root/19jjhnenu/Data/SeqTMPPI2W/featuredb/129878/'
    dirout_feature = '/root/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'

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


    f2out = 'file/10humanTrain/4train/cross/group'
    dirout_feature = '/root/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    f2resultOut = '/root/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain_80epoch/group_reusemodel_5CV'
    cvhumantrain(cv, f2out, dirout_feature, f2resultOut)
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


