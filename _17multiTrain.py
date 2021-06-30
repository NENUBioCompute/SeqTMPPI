# Title     : _17multiTrain.py
# Created by: julse@qq.com
# Created on: 2021/5/3 17:50
# des :  nohup /usr/bin/python _17multiTrain.py >0503_multiTrain
# /root/19jjhnenu/Data/SeqTMPPI2W/result/5CV
# des :  nohup /usr/bin/python _17multiTrain.py >0503_multiTrain_func2

# cpu root to root

import time
from multiprocessing import Process

from _10human_crosstrain import cvhumantrain
from _15crossvalidate import cvtrain


def fun1(cv):
    print('train cv %s' % cv)
    import os
    import tensorflow as tf

    gpu_id = cv%4
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf_config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=tf_config)
    cvtrain(cv)
def fun2(cv):
    print('train cv %s' % cv)
    import os
    import tensorflow as tf

    gpu_id = '6,7'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf_config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=tf_config)

    f2out = 'file/10humanTrain/4train/cross/group'
    dirout_feature = '/root/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    f2resultOut = '/root/19jjhnenu/Data/SeqTMPPI2W/result/10humanTrain_80epoch/group_reusemodel_5CV'
    cvhumantrain(cv, f2out, dirout_feature, f2resultOut)

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    process_list = []
    for i in range(5):  # 开启5个子进程执行fun1函数
        p = Process(target=fun1, args=(i,))  # 实例化进程对象
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('结束测试')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


