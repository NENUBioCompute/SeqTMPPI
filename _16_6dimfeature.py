# Title     : _16_6dimfeature.py
# Created by: julse@qq.com
# Created on: 2021/4/30 22:24
# des : TODO

import time

from FeatureDealer import BaseFeature, Feature_type
import numpy as np
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    # dirout_feature = ''
    # dir_feature_db = ''
    # fin_pair = ''
    # BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)

    fin = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/O14880_Q5T1C6.npy'
    pd = np.load(fin)
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

