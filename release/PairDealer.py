# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/17 17:52
@desc:
"""
import os

import pandas as pd

from FastaDealer import FastaDealer
from ProteinDealer import Protein
import numpy as np

from common import getPairs, check_path

myprotein = Protein()
class ComposeData:
    def save(self,dirout,flist, ratios,limit,labels=None,sep='\t',filename = 'all.txt'):
        """
        same length of flist,ratios,labels if not None
        :param fout:
        :param flist:
        :param ratios:
        :param limit:
        :param labels:
        :param sep:
        :return:
        case
        dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\data'
        fin_p = r'%s\positive_2049.txt' % dirin
        fin_fp = r'%s\negative_fpositive_10245.txt' % dirin
        fin_fw = r'%s\negative_fswissprot_7781.txt' % dirin
        fout = r'%s\p_fp_fw_2_1_1\all.txt' % dirin
        flist = [fin_p, fin_fp,fin_fw]
        ratios = [0.5,0.25,0.25]
        labels = [1,0,0]
        limit = 2049
        ComposeData().save(fout, flist, ratios, limit, labels=labels)
        """
        for idx,elem in enumerate(self.compose(flist, ratios, limit,labels=labels)):
            data = pd.concat(elem)
            data = data.sample(frac=1)
            fout = os.path.join(dirout,str(idx))
            check_path(fout)
            fout = os.path.join(fout,filename)
            quick_save(data, fout, sep=sep)
        # data = pd.concat(self.compose(flist, ratios, limit,labels=labels))
        # data = data.sample(frac=1)
        # quick_save(data,fout,sep=sep)
    '''
    2049:7177
    1:2 可以得到two group of data
    '''
    def compose(self,flist,ratios,limit,labels=None):
        ratios = calcu_ratio(ratios)
        src_datalist, _min,_nums = self.load(flist)
        np_ratios = np.array(ratios[1:])
        np_nums = np.array(_nums[1:])
        max_group = int(min(np_nums/(limit*np_ratios)))
        group_datasets = []
        for group in range(max_group):
            des_datalist = []
            for idx,each in enumerate(src_datalist):
                _num = min(int(limit * ratios[idx]), _nums[idx])
                if idx ==0: # positive
                    data = each.sample(_num)
                    print('load %d data from %s'%(_num,flist[idx]))
                else:
                    # data = each.sample(_num)
                    data = each[group * _num:(group+1) * _num]
                    print('load %d [%d:%d] data from %s'%(_num,group * _num,(group+1) * _num,flist[idx]))
                if labels != None:
                    data['label'] = labels[idx]
                    # data['label'] = [labels[idx]]*_num
                des_datalist.append(data)
            group_datasets.append(des_datalist)
        return group_datasets
    def load(self,flist):
        """
        load and calculate the minist length
        :param flist: first file is positive
        :return:
        """
        datalist = []
        _num = []
        for eachfile in flist:
            data = pd.read_table(eachfile, header=None)
            data = data.sample(frac=1)
            _num.append(data.shape[0])
            datalist.append(data)
        return datalist, min(_num),_num
class PairDealer:
    def __init__(self,title = False,sep = '\t'):
        self.title = title
        self.sep = sep
    def part(self,fin,ratios,fouts):
        calcu_ratio(ratios)
        data = pd.read_table(fin, header=None)
        data = data.sample(frac=1)
        _num = data.shape[0]
        start = 0
        for idx,fout in enumerate(fouts):
            end = start + int(ratios[idx] * _num)
            quick_save(data[start:end], fout)
            print('save %d pair to %s'%(end-start,fout))
            start = end
    def checkPair(self,fin_pair,fin_fasta,fout,min,max,fin_positive='',fin_subcelluar='',uncomm=False,multi=True):
        """
        at elast check min max
        :param fin_pair:
        :param fin_fasta:
        :param min:
        :param max:
        :param fin_positive:
        :param fin_subcelluar:
        :param uncomm:
        :return:
        """
        mydict = FastaDealer().getDict(fin_fasta,multi=multi)
        with open(fout,'w') as fo:
            for a,b in getPairs(fin_pair,self.sep):
                drop = True
                if not (a in mydict.keys() and b in mydict.keys()): continue
                # check length and uncom
                if myprotein.checkProtein(mydict[a], min, max,uncomm=uncomm) and myprotein.checkProtein(mydict[b], min, max,uncomm=uncomm):
                    drop = False
                # check tmp and sp
                # check subcelluar location
                # todo
                if not drop:
                    fo.write('%s\t%s\n'%(a,b))
                    fo.flush()

class GroupDealder():
    def __init__(self,dirin,fin_p,fin_fp,fin_fw):
        fin_list = [fin_p,fin_fp,fin_fw]
        self.dirin = dirin
        self.type=['p_fp_fw_2_1_1','p_fp_1_1','p_fw_1_1']
        self.fin = (fin_list,fin_list[0:2],[fin_list[0],fin_list[-1]])
        self.ratio = ([2,1,1],[1,1],[1,1])
        self.label = ([1,0,0],[1,0],[1,0])
        self.limit = 2049*2
    def save(self):
        for i in range(len(self.fin)):
            dirout = os.path.join(self.dirin, self.type[i])
            check_path(dirout)
            ComposeData().save(dirout, self.fin[i],self.ratio[i], self.limit, labels=self.label[i],filename='all.txt')
            print('divided to train and test')
            for eachdir in os.listdir(dirout):
                if '.' in eachdir:continue
                fin =os.path.join(dirout,eachdir,'all.txt')
                train =os.path.join(dirout,eachdir,'train.txt')
                test =os.path.join(dirout,eachdir,'test.txt')
                fouts = [train, test]
                ratios = [0.8, 0.2]
                PairDealer().part(fin,ratios,fouts)


def calcu_ratio(ratios):
    sum = 0
    for x in ratios:
        sum = sum +x
    return [float(x/sum) for x in ratios]
def quick_save(data,fout,sep='\t'):
    data.to_csv(fout, index=False, header=False, sep=sep)
class TMPSP:
    def __init__(self):
        pass
    def myfunc(self):
        # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
        # spf = '/home/jjhnenu/data/PPI/sp/splist.list'
        # finPair = '/home/jjhnenu/data/PPI/release/otherdata/S_C/3Scere20170205_id_pair.txt'
        # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/S_C/tmp_sp.txt'
        # getTmp_SpPair(tmpf, spf, finPair, foutPair)
        # 从列表中查询可能没有从数据库中查询效果好
        pass


if __name__ == '__main__':
    print()
    '''
    check data
    '''
    # fin_pair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\stage2\processPair2445\pair\positiveV1_fswissprot_Composi_5.txt'
    # fin_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\stage2\processPair2445\pair\positiveV1_fswissprot_Composi_5.fasta'
    # fout = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\data\negative_fswissprot_7177.txt'
    # min = 50
    # max = 2000
    # PairDealer().checkPair(fin_pair,fin_fasta,fout,min,max,fin_positive='',fin_subcelluar='',uncomm=True,multi=True)

    '''
    ComposeData
    '''
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\data'
    # fin_p = r'%s\positive_2049.txt' % dirin
    # fin_fp = r'%s\negative_fpositive_10245.txt' % dirin
    # fin_fw = r'%s\negative_fswissprot_7177.txt' % dirin
    # fout = r'%s\p_fp_fw_2_1_1\all.txt' % dirin
    # flist = [fin_p, fin_fp,fin_fw]
    # ratios = [2,1,1]
    # labels = [1,0,0]
    # limit = 2049*2
    # ComposeData().save(fout, flist, ratios, limit, labels=labels)
    '''
    divide data to train and test
    '''
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\data\p_fp_fw_2_1_1'
    # fin = r'%s\all.txt'% dirin
    # train = r'%s\train.txt' % dirin
    # test = r'%s\test.txt' % dirin
    # fouts = [train, test]
    # ratios = [0.8, 0.2]
    # PairDealer().part(fin,ratios,fouts)



    # dirin = '/home/jjhnenu/data/PPI/release/data'
    # fin_p = '%s/positive_2049.txt' % dirin
    # fin_fp = '%s/negative_fpositive_10245.txt' % dirin
    # fin_fw = '%s/negative_fswissprot_7177.txt' % dirin
    # fout = '%s/p_fw_1_1/all.txt' % dirin
    # check_path('%s/p_fw_1_1/' % dirin)
    # flist = [fin_p, fin_fw]
    # ratios = [1,1]
    # labels = [1,0]
    # limit = 2049*2
    # ComposeData().save(fout, flist, ratios, limit, labels=labels)

    # dirin = '/home/jjhnenu/data/PPI/release/data'
    # fin_p = '%s/positive_2049.txt' % dirin
    # fin_fp = '%s/negative_fpositive_10245.txt' % dirin
    # fin_fw = '%s/negative_fswissprot_7177.txt' % dirin
    # fout = '%s/p_fp_1_1/all.txt' % dirin
    # check_path('%s/p_fp_1_1/' % dirin)
    # flist = [fin_p, fin_fp]
    # ratios = [1, 1]
    # labels = [1, 0]
    # limit = 2049 * 2
    # ComposeData().save(fout, flist, ratios, limit, labels=labels)

    # dirin = '/home/jjhnenu/data/PPI/release/data/group/'
    # fin_p = '/home/jjhnenu/data/PPI/release/data/positive_2049.txt'
    # fin_fp = '/home/jjhnenu/data/PPI/release/data/negative_fpositive_10245.txt'
    # fin_fw = '/home/jjhnenu/data/PPI/release/data/negative_fswissprot_7177.txt'
    # GroupDealder(dirin,fin_p,fin_fp,fin_fw).save()


    '''
    different ratio
    '''
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\pairdata'
    # fin_p = r'%s\positive_2049.txt' % dirin
    # fin_fp = r'%s\negative_fpositive_10245.txt' % dirin
    # fin_fw = r'%s\negative_fswissprot_7177.txt' % dirin
    # for myratio in range(1,11):
    #     dirout = r'%s\p_fw\%s' % (dirin,myratio)
    #     flist = [fin_p,fin_fw]
    #     ratios = [1,myratio]
    #     labels = [1,0]
    #     limit = 2049*2
    #     ComposeData().save(dirout, flist, ratios, limit, labels=labels)

    '''
    divide data to train and test
    '''
    # root_dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\pairdata\p_fw'
    # for idx in os.listdir(root_dirin):
    #     dirin = r'%s\%s\0'%(root_dirin,idx)
    #     fin = r'%s\all.txt'% dirin
    #     train = r'%s\train0.txt' % dirin
    #     validate = r'%s\validate0.txt' % dirin
    #     test = r'%s\test0.txt' % dirin
    #     fouts = [train,validate,test]
    #     ratios = [0.8, 0.1,0.1]
    #     PairDealer().part(fin,ratios,fouts)

    '''
    different ratio
    '''
    # dirin = '/home/jjhnenu/data/PPI/release/pairdata/'
    # fin_p = '/home/jjhnenu/data/PPI/release/pairdata/positive_2049.txt'
    # fin_fw = '/home/jjhnenu/data/PPI/release/pairdata/negative_tmp_nontmp/negative_fswissprit_nontmp_v4_26798.txt'
    # for myratio in range(1,11):
    #     dirout = '%s/p_fw_v1/%s' % (dirin,myratio)
    #     flist = [fin_p,fin_fw]
    #     ratios = [1,myratio]
    #     labels = [1,0]
    #     limit = 2049*(myratio+1)
    #     ComposeData().save(dirout, flist, ratios, limit, labels=labels)
    '''
    divide data to train and test
    '''
    # root_dirin = '/home/jjhnenu/data/PPI/release/pairdata/p_fw_v1/'
    # for idx in os.listdir(root_dirin):
    #     dirin = r'%s/%s/0'%(root_dirin,idx)
    #     fin = r'%s/all.txt'% dirin
    #     train = r'%s/train.txt' % dirin
    #     validate = r'%s/validate.txt' % dirin
    #     test = r'%s/test.txt' % dirin
    #     fouts = [train,validate,test]
    #     ratios = [0.8, 0.1,0.1]
    #     PairDealer().part(fin,ratios,fouts)

    '''
    divided into validate and test for single dataset

    '''
    # dirin = '/home/jjhnenu/data/PPI/release/pairdata/p_fp_1_1'
    # fin = r'%s/all.txt'% dirin
    # train = r'%s/train.txt' % dirin
    # validate = r'%s/validate.txt' % dirin
    # test = r'%s/test.txt' % dirin
    # fouts = [train,validate,test]
    # ratios = [0.8, 0.1,0.1]
    # PairDealer().part(fin,ratios,fouts)
    '''
    result in papaer _2
    '''
    # for idx in range(0,5):
    #     # '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/3'
    #     fin = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/2/%d/all.txt'%idx
    #     train = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/2/%d/train.txt'%idx
    #     validate = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/2/%d/validate.txt'%idx
    #     test = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/2/%d/test.txt'%idx
    #     fouts = [train, validate, test]
    #     ratios = [0.8, 0.1, 0.1]
    #     PairDealer().part(fin, ratios, fouts)

    '''
    need a 1:3 dataset
    '''
    # dirin = '/home/jjhnenu/data/PPI/release/pairdata/'
    # fin_p = '/home/jjhnenu/data/PPI/release/pairdata/positive_2049.txt'
    # fin_fw = '/home/jjhnenu/data/PPI/release/pairdata/negative_tmp_nontmp/negative_fswissprit_nontmp_v4_26798.txt'
    # for myratio in range(1,11):
    #     if myratio!=3:continue
    #     dirout = '%s/p_fw_v1/3_add' % (dirin)
    #     flist = [fin_p,fin_fw]
    #     ratios = [1,myratio]
    #     labels = [1,0]
    #     limit = 2049*(myratio+1)
    #     ComposeData().save(dirout, flist, ratios, limit, labels=labels)

    '''
    5 subset of 1:3
    '''
    # for idx in range(1,6):
    #     base = '/home/jjhnenu/data/PPI/release/0deployment/TMPPI_8194'
    #     fin = '%s/%d/all.txt'%(base,idx)
    #     train = '%s/%d/train.txt'%(base,idx)
    #     validate = '%s/%d/validate.txt'%(base,idx)
    #     test = '%s/%d/test.txt'%(base,idx)
    #     fouts = [train, validate, test]
    #     ratios = [0.8, 0.1, 0.1]
    #     PairDealer().part(fin, ratios, fouts)

    '''
    5 subset of 1:3 
    move 3_add 0 to 3 4
    '''
    # for idx in range(0,5):
    #     base = '/home/jjhnenu/data/PPI/release/pairdata/p_fw_v1/3'
    #     fin = '%s/%d/all.txt'%(base,idx)
    #     train = '%s/%d/train.txt'%(base,idx)
    #     validate = '%s/%d/validate.txt'%(base,idx)
    #     test = '%s/%d/test.txt'%(base,idx)
    #     fouts = [train, validate, test]
    #     ratios = [0.8, 0.1, 0.1]
    #     PairDealer().part(fin, ratios, fouts)


    '''
    20201220
    '''
    fpositive = r'E:\githubCode\SeqTMPPI20201207\file\5_TMP_nonTMP_AC_pair.txt'
    fnegative = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\negative.txt'
    fout_fnegative_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\negative.fasta'
    fout_fpositive_fasta = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\\positive.fasta'

    # '''
    # check data
    # '''
    # fin_pair = fnegative
    # fin_fasta = fout_fnegative_fasta
    # fout = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\negative_qualified.txt'
    # min = 50
    # max = 2000
    # PairDealer().checkPair(fin_pair, fin_fasta, fout, min, max, fin_positive='', fin_subcelluar='', uncomm=True,
    #                        multi=True)
    #
    # '''
    # check data
    # '''
    # fin_pair = fpositive
    # fin_fasta = fout_fpositive_fasta
    # fout = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\positive_qualified.txt'
    # min = 50
    # max = 2000
    # PairDealer().checkPair(fin_pair, fin_fasta, fout, min, max, fin_positive='', fin_subcelluar='', uncomm=True,
    #                        multi=True)

    '''
    ComposeData
    '''
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157'
    # fin_p = r'%s\positive_qualified.txt' % dirin
    # fin_fw = r'%s\negative_qualified.txt' % dirin
    # fout = r'%s\p_fw_13349' % dirin
    # flist = [fin_p,fin_fw]
    # ratios = [1,1]
    # labels = [1,0]
    # limit = 13349*2
    # ComposeData().save(fout, flist, ratios, limit, labels=labels)
    '''
    divide data to train and test
    '''
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\intact_23157\p_fw_13349\0'
    # fin = r'%s\all.txt'% dirin
    # train = r'%s\train.txt' % dirin
    # validate = r'%s\validate.txt' % dirin
    # test = r'%s\test.txt' % dirin
    # fouts = [train,validate, test]
    # ratios = [0.8, 0.1,0.1]
    # PairDealer().part(fin,ratios,fouts)