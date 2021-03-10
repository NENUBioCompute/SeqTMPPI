# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/20 16:46
@desc:
"""
# from PairDealer import PairDealer, ComposeData
import os

# from FastaDealer import FastaDealer
# from FeatureDealer import BaseFeature, Feature_type
import time

# from FastaDealer import FastaDealer
# from PairDealer import combinePAN
# from common import check_path, getCol, concatFile, countline
# from entry import entry
# from myModel import Param
# from mySupport import savepredict
from common import check_path
from mySupport import savepredict

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    import os
    import tensorflow as tf

    gpu_id = '6,7'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=tf_config)

    # '''
    # compose tmp_sp from swissprot
    # '''
    # ftmp = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\KW\KW-0812.list'
    # fsp = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\splist.list'
    # fpositive = r'E:\githubCode\SeqTMPPI20201207\file\5_TMP_nonTMP_AC_pair.txt'
    # fnegative = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\negative.txt'
    # # randomeComposeWithTwo(readIDlist(ftmp),readIDlist(fsp),fpositive,fnegative,sep='\t', title=False,num=23157)
    #
    # '''
    # get all fasta
    # '''
    # fin = fnegative
    # fout_fnegative_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\negative.fasta'
    # description = 'positiveV1_fswissprot_Composi_5' #7781
    # # pair2Fasta(fin, fout_fnegative_fasta, description)
    #
    #
    # fin = fpositive
    # fout_fpositive_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\positive.fasta'
    # description = 'positive'
    # # pair2Fasta(fin, fout_fpositive_fasta, description)
    #`
    '''
    get tmp list and nontmp list from positive/negative pair
    '''
    # finPair_negative = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\negative_qualified.txt'
    # finPair_positive = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified.txt'
    # fileList = [finPair_negative,finPair_positive]
    # fallqualified =  r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified.txt'
    # concatFile(fileList, fallqualified)
    #
    # finPair = fallqualified
    # fallqualified_tmp = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_tmp.list'
    # getCol(finPair, fallqualified_tmp, col=0, repeat=False)
    # finPair = fallqualified
    # fallqualified_nontmp = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_nontmp.list'
    # getCol(finPair, fallqualified_nontmp, col=1, repeat=False)

    # finFasta_negative = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\negative.fasta'
    # finFasta_positive = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive.fasta'
    # fileList = [finFasta_negative,finFasta_positive]
    # fallqualified =  r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified.fasta'
    # concatFile(fileList, fallqualified)


    # countline(fallqualified_tmp)
    # countline(fallqualified_nontmp)
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157'
    # for eachfile in os.listdir(dirin):
    #     countline(os.path.join(dirin,eachfile))

    # finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified.txt'
    # fallqualified_tmp = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified_tmp.list'
    # getCol(finPair, fallqualified_tmp, col=0, repeat=False)
    #
    # finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified.txt'
    # fallqualified_nontmp = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified_nontmp.list'
    # getCol(finPair, fallqualified_nontmp, col=1, repeat=False)

    '''
    extract fasta from list
    '''

    # fd = FastaDealer()
    #
    # fin_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified.fasta'
    # fin_idlist = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_tmp.list'
    # fout_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_tmp.fasta'
    # fd.extractFasta(fin_fasta, fin_idlist, fout_fasta, in_multi=True, out_multi=True)
    #
    # fin_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified.fasta'
    # fin_idlist = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_nontmp.list'
    # fout_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_nontmp.fasta'
    # fd.extractFasta(fin_fasta, fin_idlist, fout_fasta, in_multi=True, out_multi=True)

    # fin_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified.fasta'
    # fin_idlist = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_nontmp.list'
    # fout_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\allqualified_nontmp.fasta'
    # fd.extractFasta(fin_fasta, fin_idlist, fout_fasta, in_multi=True, out_multi=True)

    '''
    check pair 
    handle pair in PiarDealer
    get train.txt ,validate.txt,test.txt all.txt
    '''

    # fin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\p_fw_13349\0\all.txt'
    # fout_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\p_fw_13349\0\all.fasta'
    # description = 'p_fw_13349' #7781
    # pair2Fasta(fin, fout_fasta, description)


    # fin = fpositive
    # fout_fpositive_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\positive.fasta'
    # description = 'positive'
    # pair2Fasta(fin, fout_fpositive_fasta, description)
    '''
    generate feature db
    '''
    # print('generate feature db')
    # fd = FastaDealer()
    # fin_fasta = '/home/19jjhnenu/Data/SeqTMPPI2W/pair/p_fw_13349/0/all.fasta'
    # dir_feature_db = '/home/19jjhnenu/Data/SeqTMPPI2W/featuredb/p_fw_13349/0'
    # fd.getNpy(fin_fasta, dir_feature_db)
    '''
    generate feature
    '''
    # print('generate feature')
    # fin_pair = '/home/19jjhnenu/Data/SeqTMPPI2W/pair/p_fw_13349/0/all.txt'
    # dirout_feature = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/p_fw_13349/0'
    # BaseFeature().base_compose(dirout_feature, fin_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)

    # '''
    # training the model
    # '''
    # print('training on the model')
    #
    #
    # fin_pair = '/home/19jjhnenu/Data/SeqTMPPI2W/pair/p_fw_13349/0/train.txt'
    # dir_in = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/p_fw_13349/0'
    # dirout = '/home/19jjhnenu/Data/SeqTMPPI2W/result/p_fw_13349/0'
    # check_path(dirout)
    # validate = {}
    # validate['fin_pair'] = '/home/19jjhnenu/Data/SeqTMPPI2W/pair/p_fw_13349/0/validate.txt'
    # validate['dir_in'] = dir_in
    # onehot = True
    # print(dirout)
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
    #       epochs=80,
    #       filters=300, batch_size=500, validate=validate)
    '''
    testing on the model
    '''
    print('testing the model')
    # fin_pair = '/home/19jjhnenu/Data/SeqTMPPI2W/pair/p_fw_13349/0/test.txt'
    # dir_in = '/home/19jjhnenu/Data/SeqTMPPI2W/feature/p_fw_13349/0'
    # fin_model = '/home/19jjhnenu/Data/SeqTMPPI2W/result/p_fw_13349/0/_my_model.h5'
    # dirout_result = '/home/19jjhnenu/Data/SeqTMPPI2W/result/p_fw_13349/0/test'
    # check_path(dirout_result)
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)

    # fin_posi = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified.txt'
    # fin_nega = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\negative_qualified.txt'
    # fout_pair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\all_qualified_label.txt'
    # combinePAN(fin_posi, fin_nega, fout_pair)

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


