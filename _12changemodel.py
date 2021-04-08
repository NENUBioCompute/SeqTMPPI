# Title     : _12changemodel.py
# Created by: julse@qq.com
# Created on: 2021/3/26 17:02
# des : TODO

import time
import os

from entry import entry
from myModel import Param

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    import os
    import tensorflow as tf

    gpu_id = '0,1,2,3'
    # gpu_id = '6,7'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=tf_config)

    print('training on the model')

    dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    dirout = '/home/19jjhnenu/Data/SeqTMPPI2W/result/changemodel/lstm'
    fin_model = None
    train = 'file/10humanTrain/4train/group/0/train.txt'
    validate = {}
    validate['fin_pair']  = 'file/10humanTrain/4train/group/0/validate.txt'
    validate['dir_in'] = dirout_feature
    onehot = True

    entry(dirout, train, dirout_feature, model_type=Param.LSTM, limit=0, onehot=onehot, kernel_size=90,
          epochs=80,
          # epochs=30,
          filters=300, batch_size=500, validate=validate,
          fin_model=fin_model)

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

