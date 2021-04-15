# encoding: utf-8
"""
@author: julse@qq.com
@time: 2021/4/15 23:45
@desc:
"""

from keras import models
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import os
from Bio import SeqIO
import numpy as np
import pandas as pd


######################## tools ###############################
def check_path(in_dir):
    if '.' in in_dir:
        in_dir,_ = os.path.split(in_dir)
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
        print('make dir ',in_dir)
def getPairs(fin,sep='\t',title=False):
    '''
    :param fin: ID1\tID2\n
    :param sep:
    :param title:
    :return: [ID1,ID2]

    '''
    func = processPair
    return processTXTbyLine(fin, func, sep, title=title)
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
def processPair(line,sep):
    return line[:-1].split(sep)
######################## data process ########################
class Protein:
    def checkProtein(self,seq, min, max,uncomm=True):
        '''

        :param seq:
        :param min:
        :param max:
        :param uncomm:
        :return:
            True: qualified
            False: not qualified
        '''
        if uncomm:
            return self.checkUncomm(seq) and self.checkLength(seq, min=min, max=max)
        else:return self.checkLength(seq, min=min, max=max)
    def checkUncomm(self,seq):
        '''
        if protein seq contains uncommon amino acid,return False
        :param seq:
        :return:
            True: qualified
            False: not qualified
        '''
        uncommList = list('XOUBZJ')
        for u in uncommList:
            if u in seq:
                print('protein seq contains uncommon amino acid', u)
                return False
        return True
    def checkLength(self,seq, min=50, max=2000):
        length = len(seq)
        return self.checkLengthRange(length,min=min,max=max)
    def checkLengthRange(self,length,min=50,max=2000):
        '''

        :param length:
        :param min:
        :param max:
        :return:
            True: qualified
            False: not qualified
        '''
        if min == max: return True  # no checking
        if length >= min and length <= max:
            return True
        print('%s length of protein seq not [50-2000]' % length)
        return False
class FastaDealer:
    def getNpy(self,fin_fasta,out_dir,multi=True,checkprotein = True):
        check_path(out_dir)
        p = Protein()
        for ID, seq in self.getYield(fin_fasta,multi=multi):
            if not checkprotein or p.checkProtein(seq, 50, 2000, uncomm=True):
                filename = os.path.join(out_dir,"%s.npy"%ID)
                result =self.seq2num(seq)
                if len(result)!=0:
                    np.save(filename,result)
    # support
    def getYield(self,fin_fasta,multi=True):
        """
        :param fin_fasta:
        :param multi:
            True
                >db|ID
                seqpart
                seqpart

                >db|ID xxx
                seqpart
                seqpart

                >ID
                seqpart
                seqpart


            False
                >db|ID
                seq

                >db|ID xxx
                seq

                >ID
                seq

        :return:
        yied like
        A6NIM6 MSVTGFTITDEKVHLYHSIEKEKTVRHIGDLCSSHSVKKIQVGICLLLVELCERFTFFEVVCNMIPFCTIKLGYHNCQAAILNLCFIGTSILTPVFVRWLTDVYLGRNKLVYICLFLHFLGTALLSVVAFPLEDFYLGTYHAVNNIPKTEQHRLFYVALLTICLGIGGVRAIVCPLGAFGLQEYGSQKTMSFFNWFYWLMNLNATIVFLGISYIQHSQAWALVLLIPFMSMLMAVITLHMIYYNLIYQSEKRCSLLTGVGVLVSALKTCHPQYCHLGRDVTSQLDHAKEKNGGCYSELHVEDTTFFLTLLPLFIFQLLYRMCIMQIPSGYYLQTMNSNLNLDGFLLPIAVMNAISSLPLLILAPFLEYFSTCLFPSKRVGSFLSTCIIAGNLFAALSVMIAGFFEIHRKHFPAVEQPLSGKVLTVSSMPCFYLILQYVLLGVAETLVNPALSVISYRFVPSNVRGTSMNFLTLFNGFGCFTGALLVKLVYLISDGNWFPNTLNKGNLESFFFFLASLTLLNVLGFCSVSQRYCNLNHFNAQNIRGSNLEETLLLHEKSLKFYGSIQEFSSSIDLWETAL
        Q14160 MLKCIPLWRCNRHVESVDKRHCSLQAVPEEIYRYSRSLEELLLDANQLRELPKPFFRLLNLRKLGLSDNEIQRLPPEVANFMQLVELDVSRNDIPEIPESIKFCKALEIADFSGNPLSRLPDGFTQLRSLAHLALNDVSLQALPGDVGNLANLVTLELRENLLKSLPASLSFLVKLEQLDLGGNDLEVLPDTLGALPNLRELWLDRNQLSALPPELGNLRRLVCLDVSENRLEELPAELGGLVLLTDLLLSQNLLRRLPDGIGQLKQLSILKVDQNRLCEVTEAIGDCENLSELILTENLLMALPRSLGKLTKLTNLNVDRNHLEALPPEIGGCVALSVLSLRDNRLAVLPPELAHTTELHVLDVAGNRLQSLPFALTHLNLKALWLAENQAQPMLRFQTEDDARTGEKVLTCYLLPQQPPPSLEDAGQQGSLSETWSDAPPSRVSVIQFLEAPIGDEDAEEAAAEKRGLQRRATPHPSELKVMKRSIEGRRSEACPCQPDSGSPLPAEEEKRLSAESGLSEDSRPSASTVSEAEPEGPSAEAQGGSQQEATTAGGEEDAEEDYQEPTVHFAEDALLPGDDREIEEGQPEAPWTLPGGRQRLIRKDTPHYKKHFKISKLPQPEAVVALLQGMQPDGEGPVAPGGWHNGPHAPWAPRAQKEEEEEEEGSPQEEEVEEEEENRAEEEEASTEEEDKEGAVVSAPSVKGVSFDQANNLLIEPARIEEEELTLTILRQTGGLGISIAGGKGSTPYKGDDEGIFISRVSEEGPAARAGVRVGDKLLEVNGVALQGAEHHEAVEALRGAGTAVQMRVWRERMVEPENAVTITPLRPEDDYSPRERRGGGLRLPLLPPESPGPLRQRHVACLARSERGLGFSIAGGKGSTPYRAGDAGIFVSRIAEGGAAHRAGTLQVGDRVLSINGVDVTEARHDHAVSLLTAASPTIALLLEREAGGPLPPSPLPHSSPPTAAVATTSITTATPGVPGLPSLAPSLLAAALEGPYPVEEIRLPRAGGPLGLSIVGGSDHSSHPFGVQEPGVFISKVLPRGLAARSGLRVGDRILAVNGQDVRDATHQEAVSALLRPCLELSLLVRRDPAPPGLRELCIQKAPGERLGISIRGGARGHAGNPRDPTDEGIFISKVSPTGAAGRDGRLRVGLRLLEVNQQSLLGLTHGEAVQLLRSVGDTLTVLVCDGFEASTDAALEVSPGVIANPFAAGIGHRNSLESISSIDRELSPEGPGKEKELPGQTLHWGPEATEAAGRGLQPLKLDYRALAAVPSAGSVQRVPSGAAGGKMAESPCSPSGQQPPSPPSPDELPANVKQAYRAFAAVPTSHPPEDAPAQPPTPGPAASPEQLSFRERQKYFELEVRVPQAEGPPKRVSLVGADDLRKMQEEEARKLQQKRAQMLREAAEAGAEARLALDGETLGEEEQEDEQPPWASPSPTSRQSPASPPPLGGGAPVRTAKAERRHQERLRVQSPEPPAPERALSPAELRALEAEKRALWRAARMKSLEQDALRAQMVLSRSQEGRGTRGPLERLAEAPSPAPTPSPTPVEDLGPQTSTSPGRLSPDFAEELRSLEPSPSPGPQEEDGEVALVLLGRPSPGAVGPEDVALCSSRRPVRPGRRGLGPVPS
        """
        if multi:
            for record in SeqIO.parse(fin_fasta, 'fasta'):
                ID = record.id
                ID = ID.split(' ')[0]
                if '|' in ID:ID = ID.split('|')[1]
                seq = str(record.seq)
                yield ID, seq
        else:
            with open(fin_fasta, 'r')as fo:
                line = fo.readline()
                while (line):
                    ID = line[1:-1]
                    ID = ID.split(' ')[0]
                    if '|' in ID: ID = ID.split('|')[1]
                    line = fo.readline()
                    seq = line[:-1]
                    line = fo.readline()
                    yield ID,seq
    def seq2num(self,seq):
        """
        FastaDealer().seq2num('GAVLIPFWMYSTCNQDEKRH')
        Out[4]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        :param seq:
        :return:
        """
        # list1 = 'GAVLIPFWMYSTCNQDEKRHXOUBZJ'
        list1 = 'GAVLIPFWMYSTCNQDEKRH'
        numList = []
        for e in seq:
            if e not in list1:return []
            numList.append(list1.index(e)+1)
        return numList
class BaseFeature:
    def base_compose(self,dirout_feature,fin_pair,dir_feature_db,fout_pair=''):
        check_path(dirout_feature)
        fo  = open(fout_pair,'w') if fout_pair!='' else None
        row = 0
        for pairs in getPairs(fin_pair):
            a = pairs[0]
            b = pairs[1]
            # print(pairs)  # ['O35668', 'P00516']
            fa = os.path.join(dir_feature_db, a + '.npy')
            fb = os.path.join(dir_feature_db, b + '.npy')
            row = row + 1
            print('loading %d th feature pair'%row)
            if not (os.access(fa, os.F_OK) and os.access(fb, os.F_OK)):
                print('===============features of pairs not found %s %s================' % (a, b), os.access(fa, os.F_OK),
                      os.access(fb, os.F_OK))
                continue
            pa = np.load(fa,allow_pickle=True)
            pb = np.load(fb,allow_pickle=True)
            if (len(pa)<50 or len(pa)>2000 or max(pa)>20) or (len(pb)<50 or len(pb)>2000 or max(pb)>20):
                print('wrong length or x')
                continue
            if fo!=None:
                fo.write('%s\t%s\n'%(a,b))
                fo.flush()
            pc = self.padding_seq1D(pa,pb,vstack=False)
            fout = os.path.join(dirout_feature, "%s_%s.npy" % (a, b))
            np.save(fout, pc)
            del pc, pa, pb
        if fo != None:
            fo.close()
    def padding_seq1D(self,pa,pb,vstack=True,shape=(2000,)):
        # data.shape = (4000,)
        # warring padding number not appear in origin data
        pa_pad_col = np.pad(pa, ((0, shape[0]-pa.shape[0])), 'constant', constant_values=0)
        pb_pad_col = np.pad(pb, ((0, shape[0]-pb.shape[0])), 'constant', constant_values=0)
        pc = np.vstack([pa_pad_col, pb_pad_col]) if vstack else np.hstack([pa_pad_col, pb_pad_col])
        return pc
class BaseData:
    def __init__(self):
        self.positive = []
        self.negative = []

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
            eachfile = os.path.join(dir_in, '%s_%s.npy' % (proteins[0], proteins[1]))
            # print(count,eachfile)
            try:
                # elem = np.load(os.path.join(dir_in, eachfile))
                elem = np.load(eachfile)
                x_test.append(elem)
                # loading test dataset or positive dataset
                if len(proteins) < 3 or proteins[2] == '1':
                    y_test.append(1)
                else:
                    y_test.append(0)
            except:
                print('not find feature of this pair', str(proteins))
            if count == limit:break
        data = np.array(x_test)
        label = np.array(y_test)
        return self.subprocess(data,label,test_size=0, random_state=123,onehot=onehot,is_shuffle=is_shuffle)

    def subprocess(self,data,label,test_size=0.1, random_state=123,onehot=False,is_shuffle=True):
        if is_shuffle:
            index = [x for x in range(len(label))]
            np.random.shuffle(index)
            data = data[index]
            label = label[index]
        if onehot:
            data = to_categorical(data)
        else:
            if len(data.shape) > 2:  # for 2D
                data = np.reshape(data, data.shape + (1,))
        if test_size == 0:
            return data, label
        else:
            print('end of the process')
            x_train, x_test, y_train, y_test = \
                train_test_split(data, label, test_size=test_size, random_state=random_state)
            return (x_train, y_train), (x_test, y_test)

######################## predicting ##########################
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
def MCC(y_true,y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    myMCC = (TP*TN - FP*FN)*1.0/(tf.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+K.epsilon())
    return myMCC
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
def getFeature(fin_pair,fin_fasta,dir_feature_db,dirout_feature):
    # fin_pair = '%s/dirRelated/2pair.tsv'%dirout
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
    BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db)
def savepredict(fin_pair,dir_in,fin_model,dirout_result,batch_size=90,limit=0):
    check_path(dirout_result)
    print('predict ',fin_pair,'...')
    print('save result in ',dirout_result)
    df = pd.read_table(fin_pair, header=None)
    if df.shape[1]!=3:df[2]=0
    onehot = True
    dataarray = BaseData().loadTest(fin_pair, dir_in,onehot=onehot,is_shuffle=False,limit=limit)
    x_test, y_test =dataarray
    print('load model...')
    # model = load_model(fin_model, custom_objects=MyEvaluate.metric_json)
    model = models.load_model(fin_model, custom_objects=MyEvaluate.metric_json)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=MyEvaluate.metric)
    result = model.evaluate(x_test, y_test, verbose=1,batch_size=batch_size)

    result_predict = model.predict(x_test,batch_size=batch_size)
    result_class = (result_predict > 0.5).astype("int32")
    result_predict = result_predict.reshape(-1)

    # result_class = model.predict_classes(x_test,batch_size=batch_size)
    # UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01.

    result_class = result_class.reshape(-1)

    print('manual result:[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]\n')
    print('evaluate result:' + str(result) + '\n')
    # y_test = y_test.reshape(-1)

    if limit!=0:df = df[:limit]

    df.columns = ['tmp', 'nontmp','real_label']
    # df.rename(columns={0: 'tmp', 1: 'nontmp'}, inplace=True)
    # df['real_label'] = list(y_test)
    df['predict_label'] = result_class
    df['predict'] = result_predict
    df.sort_values(by=['predict'],ascending=False).to_csv(os.path.join(dirout_result,'result.csv'),index=False)

    with open(os.path.join(dirout_result,'log.txt'),'w') as fo:
        fo.write('test dataset %s\n'%fin_pair)
        fo.write('manual result:[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]\n')
        fo.write('evaluate result:'+str(result)+'\n')
        fo.flush()
