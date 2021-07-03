# Title     : v2_train_zkd_single.py
# Created by: julse@qq.com
# Created on: 2021/7/3 17:27
# des : TODO


############# save package ################

import os
from functools import wraps
from tensorflow.keras import models, Input, Model, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv2D, Conv1D, Embedding, GlobalAveragePooling2D, Flatten, \
    Dropout, Activation, MaxPooling2D, MaxPooling1D, LSTM, concatenate, Concatenate, GlobalMaxPooling1D
from tensorflow import keras
###########################################
import time

from keras import models
import os

import numpy as np
from sklearn.model_selection import train_test_split
from myModel import MyModel, Param
# from mySupport import calculateResults, savepredict, plot_result
import pandas as pd



import os

from myModel import Param
from mySupport import savepredict, calculateResults
from v2_FastaDealear import FastaDealer
from v2_FeatureDealer import BaseFeature, Feature_type

def check_path(in_dir):
    if '.' in in_dir:
        in_dir,_ = os.path.split(in_dir)
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
        print('make dir ',in_dir)
class BaseData:
    def __init__(self):
        self.positive = []
        self.negative = []

    def load(self,fin_pair,dir_in,test_size=0.1, random_state=123,limit=0,onehot=False):
        # (positive, negative)
        self.loadPair(fin_pair,dir_in,limit=limit)
        return self.process(test_size=test_size, random_state=random_state,onehot=onehot)
    def loadPAN(self,dir_positive,dir_negative,test_size=0.1, random_state=123,limit=0):
        self.positive = self.loadFile(dir_positive,limit=limit)
        self.negative = self.loadFile(dir_negative,limit=limit)
        return self.process(test_size=test_size, random_state=random_state)

    def process(self,test_size=0.1, random_state=123,onehot=False):
        data = np.vstack([self.positive, self.negative]) if self.negative!=[] else self.positive
        label = np.hstack([np.ones(len(self.positive)), np.zeros(len(self.negative))]) if self.negative!=[] else np.ones(len(self.positive))
        return self.subprocess(data,label,test_size=test_size, random_state=random_state,onehot=onehot)

    def subprocess(self,data,label,test_size=0.1, random_state=123,onehot=False,is_shuffle=True):
        if is_shuffle:
            index = [x for x in range(len(label))]
            np.random.shuffle(index)
            data = data[index]
            label = label[index]
        if onehot:
            from keras.utils import to_categorical
            data = to_categorical(data)
        else:
            if len(data.shape) > 3:  # for 2D
                data = np.reshape(data, data.shape + (1,))
        if test_size == 0:
            return data, label
        else:
            print('end of the process')
            x_train, x_test, y_train, y_test = \
                train_test_split(data, label, test_size=test_size, random_state=random_state)
            return (x_train, y_train), (x_test, y_test)
    def loadFile(self,dir_in,limit=0):
        mylist = []
        row = 0
        for eachfile in os.listdir(dir_in):
            mylist.append(np.load(os.path.join(dir_in,eachfile)))
            row = row + 1
            if limit!=0 and row ==limit:break
        return np.stack(mylist)
    def loadPair(self,fin_pair,dir_in,limit=0):
        positive = []
        negative = []
        row = 0
        for proteins in getPairs(fin_pair,title=False):
            eachfile = os.path.join(dir_in,'%s_%s.npy'%(proteins[0],proteins[1]))
            try:
                elem = np.load(os.path.join(dir_in, eachfile))
                # loading test dataset or positive dataset
                if len(proteins) < 3 or proteins[2] == '1':
                    positive.append(elem)
                else:
                    negative.append(elem)
                row = row + 1
                if row ==limit:break
            except:
                print('not find feature of this pair',str(proteins))
        # if limit!=0 and limit<min(len(positive),len(negative)):positive,negative = positive[:limit],negative[:limit]
        print('positive : ',len(positive))
        print('negative : ',len(negative))
        positive = np.stack(positive)
        negative = np.stack(negative) if negative!=[] else []
        self.positive = positive
        self.negative = negative

    def loadTest(self,fin_pair,dir_in,onehot=False,is_shuffle=False,limit=0):
        """

        :param fin_pair:
        :param dir_in:
        :param limit:
        :param onehot:
        :return: data,label
        """
        x_test = []
        y_test = []
        count = 0
        for proteins in getPairs(fin_pair, title=False):
            count = count +1
            xelem,yelem = self.loadPpair(dir_in, proteins)
            x_test.append(xelem)
            y_test.append(yelem)
            # eachfile = os.path.join(dir_in, '%s_%s.npy' % (proteins[0], proteins[1]))
            # # print(count,eachfile)
            # try:
            #     # elem = np.load(os.path.join(dir_in, eachfile))
            #     elem = np.load(eachfile)
            #     x_test.append(elem)
            #     # loading test dataset or positive dataset
            #     if len(proteins) < 3 or proteins[2] == '1':
            #         y_test.append(1)
            #     else:
            #         y_test.append(0)
            # except:
            #     print('not find feature of this pair', str(proteins))
            if count == limit:break
        data = np.array(x_test)
        label = np.array(y_test)
        return self.subprocess(data,label,test_size=0, random_state=123,onehot=onehot,is_shuffle=is_shuffle)
    def loadPpair(self,dir_in,proteins):
        eachfile = os.path.join(dir_in, '%s_%s.npy' % (proteins[0], proteins[1]))
        # print(count,eachfile)
        try:
            # elem = np.load(os.path.join(dir_in, eachfile))
            xelem = np.load(eachfile)
            # loading test dataset or positive dataset
            if len(proteins) < 3 or proteins[2] == '1':
                yelem=1
            else:
                yelem=0
            return xelem, yelem
        except:
            print('not find feature of this pair', str(proteins))
    # def loadDataWithLable(self,fin_pair,dir_in,onehot=False,is_shuffle=False):
    #     return self.subprocess(data,label,test_size=0, random_state=123,onehot=onehot,is_shuffle=is_shuffle)
    # def loadFeature(self, fin_pair,dir_in):
    #     # dir_in = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/129878/'
    #     # fin_pair = 'file/4train/0/validate.txt'
    #     df = pd.read_table(fin_pair, header=None)[1:10]
    #     data,label= df.apply(lambda x: self.func(x,dir_in), axis=1).values
    # def func(self,x,dir_in):
    #     eachfile = os.path.join(dir_in, '%s_%s.npy' % (x[0], x[1]))
    #     print(eachfile)
    #     return np.load(eachfile),x[2]

def processPair(line,sep):
    return line[:-1].split(sep)
def processTXTbyLine(fin,func,sep,title=True):
    """
    func = processPair
    genre = processTXTbyLine(finPair, func, '\t', title=False)
    :param fin: ID pair
    :param func: process pair
    :param sep:
    :param title:
    :return: [proteins for proteins in genre]
    """
    with open(fin,'r')as fo:
        if title:fo.readline()
        line = fo.readline()
        while (line):
            yield func(line,sep)
            line = fo.readline()
def getPairs(fin,sep='\t',title=False):
    '''
    :param fin: ID1\tID2\n
    :param sep:
    :param title:
    :return: [ID1,ID2]

    '''
    func = processPair
    return processTXTbyLine(fin, func, sep, title=title)
def getFeature(fin_pair,fin_fasta,dir_feature_db,dirout_feature):
    # fin_pair = '%s/dirRelated/2pair.tsv'%dirout
    '''
    generate feature db
    '''
    print('generate feature db')
    fd = FastaDealer()
    fd.getPhsi_Blos(fin_fasta, dir_feature_db)
    '''
    generate feature
    '''
    print('generate feature')
    BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.PHSI_BLOS,check_data=False)


#################### model #############################

class Param:
    metrics=MyEvaluate.metric

    CNN1D = 'CNN1D'
    CNN1D_OH = 'CNN1D_OH' # onehot
    CNN2D = 'CNN2D'
    LSTM = 'LSTM'
    DNN = 'DNN'
    CNN_LSTM = 'CNN_LSTM'
    CNN1D_MAX_OH = 'CNN1D_MAX_OH'
    CNN1D_6DIM = 'CNN1D_6DIM'
    TRANSFORMER = 'TRANSFORMER'

class MyModel(object):
    def __init__(self,
                input_shape = (160,),
                filters = 250,
                kernel_size = 3,
                pool_size = 2,
                hidden_dims = 250,
                batch_size=100,
                epochs = 60,
                metrics = None,
                model_type = Param.CNN1D
                 ):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type
        if metrics == None:self.metrics = Param.metrics
    def __call__(self, func,x_train,y_train,validation_data=None,*args, **kwargs):
        @wraps(func) # todo
        def wrapper(x_train,y_train,validation_data=validation_data,*args, **kwargs):
            # history = self.process(x_train,y_train,validation_data=validation_data)
            # return func(history, *args, **kwargs)
            pass
        return wrapper

    def process(self,fout,x_train, y_train, x_test,y_test,fin_model=None):
        if self.model_type == Param.CNN_LSTM:
            fixlen = int(self.input_shape[0]/2)
            x_train = [x_train[:,:fixlen],x_train[:,fixlen:]]
            x_test = [x_test[:,:fixlen],x_test[:,fixlen:]]
        print('x_train.shape,x_test.shape',x_train[0].shape,x_test[0].shape)
        if fin_model:self.loadExistModel(fin_model)
        else:self.loadModel()
        self.complie()
        self.fit(x_train,y_train,validation_data=(x_test, y_test))
        self.save_model(fout)
        self.save_result(fout,x_test,y_test)
        # plot_result(self.history.history,fout)

    def process_re_emerge(self,fin_model,x_test,y_test):
        self.loadExistModel(fin_model)
        self.complie()
        print(self.evaluate(x_test, y_test))

    # support process
    def loadModel(self):
        model =None
        if self.model_type==Param.CNN1D:model =self.CNN1D()
        elif self.model_type==Param.CNN1D_OH:model =self.CNN1D_OH()
        elif self.model_type==Param.CNN1D_6DIM:model =self.CNN1D_OH()
        elif self.model_type == Param.CNN1D_MAX_OH:model = self.CNN1D_MAX_OH()
        elif self.model_type==Param.CNN2D:model =self.CNN2D()
        elif self.model_type==Param.LSTM:model =self.LSTM()
        elif self.model_type==Param.DNN:model =self.DNN()
        elif self.model_type == Param.CNN_LSTM:model = self.CNN_LSTM()
        elif self.model_type == Param.TRANSFORMER:model = self.TRANSFORMER()
        else:assert 'no such model'
        self.model = model

    def complie(self):
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=self.metrics)
        self.model.summary()
    def fit(self,x_train,y_train,validation_data=None,log_dir=None):
        """
        :param x_train:
        :param y_train:
        :param validation_data: validation_data=(x_test, y_test)
        :return:history
        """

        # from keras.callbacks import Tensorboard
        # tensorboard = Tensorboard(log_dir=log_dir)
        # callback_lists = [tensorboard]  # 因为callback是list型,必须转化为list

        self.history  = self.model.fit(x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=validation_data,
                            # callbacks=[EarlyStopping(monitor='loss', patience=3,min_delta=0.00003)]
                            callbacks=[EarlyStopping(monitor='loss', patience=10,min_delta=0.000003)]
                                       )

    def save_model(self,fout):
        # plot_model(self.model, to_file=os.path.join(fout,'_model.png'), show_shapes=True, show_layer_names='False', rankdir='TB')
        self.model.save(os.path.join(fout,'_my_model.h5'))  # creates a HDF5 file 'my_model.h5'
        json_string = self.model.to_json()
        with open(os.path.join(fout,'_my_model.json'), 'w') as fo:
            fo.write(json_string)
            fo.flush()

    def save_result(self,fout,x_test,y_test):
        history_dict = self.history.history
        with open(os.path.join(fout,'_history_dict.txt'), 'w') as fo:
            fo.write(str(history_dict))
            fo.flush()
        with open(os.path.join(fout , '_evaluate.txt'), 'w') as fi:
            fi.write('evaluate:' + str(self.evaluate(x_test, y_test)) + '\n')
            fi.write('history.params:' + str(self.history.params) + '\n')

    def CNN1D(self):
        model = Sequential()
        model.add(Embedding(21, 160, input_shape=self.input_shape))
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model
    def CNN1D_OH(self):
        model = Sequential()
        model.add(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def CNN1D_MAX_OH(self):
        model = Sequential()
        model.add(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D())
        # model.add(Flatten())
        # model.add(
        #     Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def CNN2D(self):
        model = Sequential()
        model.add(Conv2D(32, self.kernel_size, padding='valid',
                         input_shape=self.input_shape, data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, self.kernel_size, padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))

        return model


    def LSTM(self):

        model = Sequential()
        model.add(
            LSTM(128,
                 input_shape=self.input_shape,
                 activation='relu',
                 return_sequences=True))
            # LSTM(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def DNN(self):
        pass

    def CNN_LSTM(self):
        # main_input_a = Input(shape = self.input_shape/2)
        _shape = (int(self.input_shape[0]/2),)
        print('_shape',_shape)

        input_a = Input(shape=_shape)
        embedding_a = Embedding(21, 128)(input_a)
        conv1d_a = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(embedding_a)
        pool_a = MaxPooling1D()(conv1d_a)
        lstm_a = LSTM(80)(pool_a)


        input_b = Input(shape=_shape)
        embedding_b = Embedding(21, 128)(input_b)
        conv1d_b = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(embedding_b)
        pool_b = MaxPooling1D()(conv1d_b)
        lstm_b = LSTM(80)(pool_b)

        concat = concatenate([lstm_a,lstm_b],axis=-1)
        predictions = Dense(1, activation='sigmoid')(concat)

        model = Model(inputs=[input_a,input_b],outputs=predictions)
        model.evaluate()
        return model
    def TRANSFORMER(self):
        model = Sequential()
        model.add(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape))

        # embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        # ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs = layers.Input(shape=self.input_shape)
        # embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        # x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.input_shape[-1], num_heads, self.input_shape[-1])
        x = transformer_block(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    # support process_re_emerge
    def loadExistModel(self,fin_model):
        self.model = models.load_model(
            fin_model,
            custom_objects=MyEvaluate.metric_json)
        self.model.summary()
    # support save_result
    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test, y_test, verbose=False,batch_size=self.batch_size)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


################## evaluate #############################
import tensorflow as tf
import keras.backend as K



#精确率评价指标
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP+ K.epsilon())
    return precision
#召回率评价指标
def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN+ K.epsilon())
    return recall
#F1-score评价指标
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP+ K.epsilon())
    recall=TP/(TP+FN+ K.epsilon())
    F1score=2*precision*recall/(precision+recall+ K.epsilon())
    return F1score
# MCC
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
    # return MCC(y_true, y_pred)

'''
old model need this
'''
def MCC(y_true,y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    myMCC = (TP*TN - FP*FN)*1.0/(tf.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+K.epsilon())
    return myMCC
def metric_ACC(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    acc=(TP+TN)/(TP+FP+TN+FN+K.epsilon())
    return acc
class MyEvaluate:
    metric = ['acc',metric_precision, metric_recall, metric_F1score, matthews_correlation]
    metric_json = {
        'acc':'acc',
        'metric_precision': metric_precision,
        'metric_recall': metric_recall,
        'metric_F1score': metric_F1score,
        'MCC': MCC,
        'matthews_correlation': matthews_correlation
    }
    def evaluate_manual(self,y_true,y_pred):
        # y_true = tf.constant(list(y_true), dtype=float)
        # y_pred = tf.constant(list(y_pred),dtype=float)
        y_true = tf.constant(list(y_true),dtype=float)
        y_pred = tf.constant(y_pred,dtype=float)
        return [K.eval(metric_ACC(y_true, y_pred)),
                K.eval(metric_precision(y_true, y_pred)),
                K.eval(metric_recall(y_true, y_pred)),
                K.eval(metric_F1score(y_true, y_pred)),
                K.eval(matthews_correlation(y_true, y_pred))]


def entry(dirout,fin_pair,dir_in,model_type = Param.CNN1D,limit=0,onehot=False,kernel_size = 3,epochs=60,filters = 250,batch_size = 100,validate=None,fin_model=None):
    '''
    :param dirout:
    :param fin_pair:
    :param dir_in:
    :param model_type:
    :param limit: used when validate ==None
    :param onehot:
    :param kernel_size:
    :param epochs:
    :param filters:
    :param batch_size:
    :param validate:
    :param fin_model: if not none , reload model and train
    :return:
    '''
    # model_type = Param.CNN1D
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/pssm400_feature_1D/p_fp_fw_2_1_1/all/'
    # fout = '/home/jjhnenu/data/PPI/release/result/pssm400_feature_1D/p_fp_fw_2_1_1/all/'
    # if 'all' not in dirout:return
    check_path(dirout)
    print('dirout:',dirout)
    # dir_in = dirout.replace(des, 'feature')
    print('dir_in:',dir_in)
    print('fin_pair',fin_pair)
    bd = BaseData()
    print('load dataset...')
    if validate == None:
        (x_train, y_train), (x_test, y_test) = bd.load(fin_pair,dir_in,test_size=0.1,limit=limit,onehot=onehot)
    else:
        x_test, y_test = bd.loadTest(validate['fin_pair'],validate['dir_in'],onehot=onehot,is_shuffle=True,limit=limit)
        x_train, y_train= bd.loadTest(fin_pair,dir_in,onehot=onehot,is_shuffle=True,limit=limit)

    print('Build and fit model...')
    print('x_train.shape',x_train.shape)
    mm = MyModel(model_type = model_type,
                    input_shape = x_train.shape[1:],
                    filters = filters,
                    kernel_size = kernel_size,
                    pool_size = 2,
                    hidden_dims = 250,
                    batch_size = batch_size,
                    epochs = epochs,)
    mm.process(dirout,x_train, y_train, x_test, y_test,fin_model=fin_model)
    print('save result to %s'%dirout)
    del x_test,x_train,y_test,y_train,mm


def crossTrain(dirout_feature,f2resultOut,modelreuse=True):

    '''
    cross train and test
    '''
    f1out = 'file/4train/'
    f2outdir = os.path.join(f1out, str(0))
    fin_pair = os.path.join(f2outdir, 'all.txt')

    f2outdir = os.path.join(f1out, '5CV','data')
    check_path(f2outdir)
    train = os.path.join(f2outdir, 'train_vali.txt')
    test = os.path.join(f2outdir, 'test.txt')
    ratios_tvt = [5,1]
    f3outs = [train, test]
    # PairDealer().part(fin_pair,ratios_tvt,f3outs)
    '''
    train model
    '''
    f2outdir = os.path.join(f1out, '5CV','data')
    check_path(f2outdir)
    train = os.path.join(f2outdir, 'train_vali.txt')
    f2out = 'file/4train/5CV/elem'
    ratios_tvt = [1] * 5
    f3outs = [os.path.join(f2out,'%d.txt'%x) for x in range(5)]
    # PairDealer().part(train,ratios_tvt,f3outs)
    limit = 0
    # eachdir = 'benchmark'
    # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
    # f2resultOut = '/mnt/data/sunshiwei/SeqTMPPI2W/result/5CV_1'
    '''
    cross train
    '''
    for cv in range(5):
        oldfile = '-1'
        f2dirout = os.path.join(f2resultOut, str(cv))
        fin_model = ''
        f3dirout=''
        for elem in range(5):
            if cv == elem:continue
            f3dirout = os.path.join(f2dirout,str(elem))
            fin_model = os.path.join(f2dirout,oldfile,'_my_model.h5')
            if not os.access(fin_model,os.F_OK) or not modelreuse:fin_model=None
            train = os.path.join(f2out, '%d.txt'%elem)
            validate = {}
            validate['fin_pair'] = os.path.join(f2out, '%d.txt'%cv)
            validate['dir_in'] = dirout_feature
            onehot = False

            entry(f3dirout, train, dirout_feature, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
                  epochs=80,
                  # epochs=2,
                  filters=300, batch_size=500, validate=validate,
                  fin_model=fin_model)

            # entry(f3dirout, train, dirout_feature, model_type=Param.TRANSFORMER, limit=10, onehot=onehot, kernel_size=90,
            #       # epochs=80,
            #       epochs=2,
            #       filters=300, batch_size=500, validate=validate,
            #       fin_model=fin_model)
        #
        #
            oldfile = str(elem)
        #     # print(f3dirout)
        calculateResults(f2dirout, f2dirout, resultfilename='result.csv')

        # eachdir = 'benchmark'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % eachdir
        # print('testing the model on test dataset')
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # fin_test = 'file/4train/5CV/data/test.txt'
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_test/%d'%cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test,batch_size=500,limit=2000,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/predict
        '''
        # fin_test = 'file/8DIPPredict/predict/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/data
        '''
        # fin_test = 'file/8DIPPredict/data/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)


        '''
        testing on  all.txt in Imex
        '''
        # fin_test = 'file/8ImexPredict/4pair.tsv'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_IMEx_posi/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        IMEx + - 
        '''
        # fin_test = 'file/8ImexPredict/predict/0/all.txt'
        # dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'IMEx'
        # fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        # dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_IMEx/%d' % cv
        # check_path(dirout_result_test)
        # savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)
        '''
        testing on DIP all.txt in DIP/data/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/data/%s/2pair.tsv'%eachfile
        #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP_%s/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)

        '''
        testing on DIP all.txt in DIP/predict/Human
        '''
        # for eachfile in ['Ecoli', 'Human', 'Mus', 'SC']:
        #     fin_test = 'file/8DIPPredict/predict/%s/0/all.txt'%eachfile
        #     dirout_feature = '/mnt/data/sunshiwei/Phsi_Blos/feature/%s/' % 'DIP'
        #     fin_model = os.path.join(f2dirout, oldfile, '_my_model.h5')
        #     dirout_result_test = '/mnt/data/sunshiwei/Phsi_Blos/result/5CV_1_DIP_%s_+-/%d' % (eachfile,cv)
        #     check_path(dirout_result_test)
        #     savepredict(fin_test, dirout_feature, fin_model, dirout_result_test, batch_size=500,onehot = onehot)
