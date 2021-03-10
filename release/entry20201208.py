# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/8 21:04
@desc:
"""
from FastaDealer import FastaDealer
from common import check_path

if __name__ == '__main__':
    ''' 
    input file : 
        positive pair and negative pair
        AC-pair of TMP-nonTMP
        fasta 
    generate feature
    fasta 
    '''
    print()

    '''
   fasta to feature
   '''
    # >ID\nseq\n
    fin_fasta = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_DIP_fasta20170301_simple.seq'
    dir_feature_db = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_featuredb'
    check_path(dir_feature_db)
    fd = FastaDealer()
    fd.getNpy(fin_fasta,dir_feature_db)
    '''
    generate feature pair
    50-2000 no X
    '''
    dir_feature_db = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_featuredb'
    dir_feature = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_feature/'
    # ID pair proteinA\tproteinB\n
    dir_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/'
    dirout_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP_qualified/'
    check_path(dir_feature)
    check_path(dirout_pair)
    for eachfile in os.listdir(dir_pair):
        print(eachfile)
        fin_pair = os.path.join(dir_pair, eachfile)
        fout_pair = os.path.join(dirout_pair,eachfile)
        BaseFeature().base_compose(dir_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D,fout_pair=fout_pair)
        countpair(fout_pair)

    # ##########################################################
    '''
    test single dataset on model
    '''
    # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/Ecoli_TMP_nonTMP_1156.txt'
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_cpu/9/1/_my_model.h5'
    # onehot = True
    # (x_train, y_train), (x_test, y_test) = BaseData().load(fin_pair, dir_feature, limit=0, onehot=onehot)
    # model = models.load_model(fin_model,custom_objects=MyEvaluate.metric_json)
    # result = model.evaluate(x_train, y_train, verbose=False)
    # print('Loss:%f,ACC:%f'%(result[0],result[1]))
    # result_predict = model.predict(x_train)
    # print(result_predict)