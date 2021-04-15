# Title     : _10humanTrain_suppGPU.py
# Created by: julse@qq.com
# Created on: 2021/3/22 16:22
# des : TODO
import os
from common import check_path
from entry import entry
from myModel import Param

def _5train(f1out,eachfile,train,dirout_feature,f2resultOut,fin_model=None):

    '''
    training the model
    '''
    print('training on the model')

    dir_in = dirout_feature
    dirout = os.path.join(f2resultOut, eachfile)
    check_path(dirout)

    validate = {}
    validate['fin_pair'] = os.path.join(f1out, eachfile, 'validate.txt')
    validate['dir_in'] = dir_in
    onehot = True

    entry(dirout, train, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
          epochs=80,
          # epochs=30,
          filters=300, batch_size=500, validate=validate,
          fin_model=fin_model)
