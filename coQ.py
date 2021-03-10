# Title     : coQ.py
# Created by: julse@qq.com
# Created on: 2021/3/6 17:22
# des : TODO

import time
import pandas as pd
from DatabaseOperation2 import DataOperation

def readIDlist(filePath,by = '\n'):
    fo = open(filePath,'r')
    lines = fo.readlines()
    # 'P67999\n', 'P0ADV1\n', 'P0A6F5\n',
    list = []
    for line in lines:
        list.append(line.split(by)[0])
    fo.close()
    # 'P67999', 'P0ADV1', 'P0A6F5'
    return list
def saveList(inlist,outListFile,by = '\n',file_mode='w'):
    """
    将list保存为指定格式的txt文件
    :param inlist: 输入的list
    :param outListFile: 输出list文件的位置
    :param by:分割符号
    :return:
    eg
    idlist = [1,2,3]
    outListF = 'file/result/pfam/pfamJson/queryIdList.txt'
    saveList(idlist, outListF, by='\n')
    """
    with open(outListFile,file_mode) as fo:
        for e in inlist:
            fo.write(str(e)+by)
            fo.flush()
def generateCoQLists(fcoQ):
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':True,'accession':True}
    docs = do.QueryObj({"keyword.@id":'KW-0830'},projection=projection)
    coQlist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    print('query %d coQ'%(len(coQlist)))
    saveList(coQlist,fcoQ)
    return coQlist
def generateCoQInfo(finfo):
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':True,'accession':True}
    docs = do.QueryObj({"keyword.@id":'KW-0830'},projection=projection)
    coQinfo = [(x['accession'][0],x['_id'].split('_')[1]) if isinstance(x['accession'],list) else (x['accession'],x['_id'].split('_')[1]) for x in docs]
    print('query %d coQ info'%(len(coQinfo)))
    saveList(coQinfo,finfo)
    return coQinfo
def tuple2single(finfo,fout):
    df = pd.read_table(finfo, header=None)
    df[1] = df[0].apply(lambda x:eval(x)[0])
    df[2] = df[0].apply(lambda x:eval(x)[1])
    df[[1,2]].to_csv(fout,header=None,index=None,sep='\t')
def findpartner(fpair,fin,fout,col=0,pairSep = ','):
    '''
    根据nontmp在tmp_nonTmp表中找tmp
    :param fpair:
    :param fin:
    :param fout:
    :return:
    '''
    # fpair = 'file/2intAct_pair_norepeat_info.txt'
    # fin = 'file_kegg/all_protein.csv'
    # fout = 'file_kegg/protein_partner.csv'
    df = pd.read_csv(fin,sep='\t',header=None)
    df.columns=[col]
    df_pair = pd.read_csv(fpair,sep=pairSep,header=None)
    df1 = df_pair.merge(df)
    print(df1.shape)
    df1.to_csv(fout,header=None,index=None,sep='\t')
def getPartnerlist(fcoQ_partner,fpartner_coQ,f3partnerList):
    df1 = pd.read_table(fcoQ_partner, header=None)[1]
    df1.columns=[0]
    df2 = pd.read_table(fpartner_coQ, header=None)[0]
    df2 = df2.append(df1).drop_duplicates()
    df2.to_csv(f3partnerList,header=None,index=None,sep='\t')
    print(df2.shape)
########################## pfam ###################################
def proteinPfam(fin,fout,tophit=True,item='Pfam'):
    do = DataOperation('uniprot', 'uniprot_sprot')
    df = pd.read_csv(fin, header=None)
    df = pd.DataFrame(df)
    df2 = df.apply(lambda x: funcPfam(x,do,tophit=tophit,item=item), axis=1)
    df2.to_csv(fout, sep='\t',header=None, index=None)
def funcPfam(x,do,tophit=False,item='Pfam'):
    # ac = 'P03372'
    result = queryPfam(x[0], do, tophit=tophit,item=item)
    if result:
        print(result)
        if tophit:
            x[1] = result[0]
            x[2] = result[1]
        else:
            x[1] = result
    return x
def queryPfam(ac,do,tophit=False,item='Pfam'):
    dic = {'accession':ac}
    projection = {'dbReference':True,'_id':False}
    qa = do.QueryObj(dic,projection=projection)
    result = None
    for q in qa:
        dbReference = q['dbReference']
        # {'@type': 'Pfam', '@id': 'PF18424', 'property': [{'@type': 'entry name', '@value': 'a_DG1_N2'}
        #  {'@type': 'Pfam', '@id': 'PF12743', 'property': [{'@type': 'entry name', '@value': 'ESR1_C'}, {'@type': 'match status', '@value': '1'}]}
        if item == 'Pfam':result = getPfam(dbReference)
        return result if not tophit else result[0] if result else None
    return None
def getPfam(dbref):
    '''
    :param dbref:
    :return: list of (pfamId,pfamName)
    [('PF12743', 'ESR1_C'), ('PF00104', 'Hormone_recep'), ('PF02159', 'Oest_recep'), ('PF00105', 'zf-C4')]
    '''
    pfamInfo = []
    for dbr in dbref:
        if dbr['@type'] == 'Pfam':
            for pr in dbr['property']:
                if pr['@type'] == 'entry name':
                    pfamInfo.append((dbr['@id'],pr['@value']))
    return pfamInfo
######################## end of Pfam #############################

def checkTaN(keywordDic):
    '''
    :param keywordDic:
    :return:
        True: TMP
        False: nonTMP
    '''
    # check T and N
    if isinstance(keywordDic,list):
        for keyword in keywordDic:
            if keyword['@id'] == 'KW-0812':
                return True
    else:
        if keywordDic['@id'] == 'KW-0812':
            return True
    return False
def queryProtein(AC,do,projection=None):
    '''
    dao
    :param AC:
    :param do:
    :return:
    '''
    # one accession mapping several protein sequence
    protein = None
    docs = do.Query('accession',AC,projection=projection)
    count = 0
    for doc in docs:
        count = count +1
        if count >1:
        #    一个accession 查询到多个蛋白质
        # 保存这个列表
            print('%s is more than one entry'%AC)
            return None
        protein = doc
    return protein
def getTmp(fin,fout):
    '''
    :param fin: protein list
    :param fout:  tmp list
    :return:
    '''
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':True,'keyword.@id':True}
    aclist = readIDlist(fin)
    with open(fout,'w') as fo:
        for AC in aclist:
            qa = do.QueryOne('accession',AC,projection=projection)
            if qa==None:continue
            if checkTaN(qa['keyword']):
                print('%s TMP'%AC)
            pa_TMP_flag = False if 'keyword' not in qa.keys() else checkTaN(qa['keyword'])
            if pa_TMP_flag:
                fo.write('%s\n'%AC)
                fo.flush()
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    # '''
    # query coQ list
    # query 5926 coQ
    # '''
    fcoQ = 'file_coQ/coQ.list'
    #
    # generateCoQLists(fcoQ)
    # '''
    # find species
    # '''
    # finfo = 'file_coQ/coQ_info.tsv'
    # finfo_1 = 'file_coQ/coQ_info_1.tsv'
    #
    # generateCoQInfo(finfo)
    # tuple2single(finfo, finfo_1)
    # '''
    # find partner
    # '''
    # intActPair = '/home/jjhnenu/SeqTMPPI20201226/file/1intAct_pair_norepeat.txt'
    # fcoQ_partner = 'file_coQ/2coQ_partner.tsv' # (497, 2)
    # fpartner_coQ = 'file_coQ/2partner_coQ.tsv' # (812, 2)
    #
    # findpartner(intActPair,fcoQ,fcoQ_partner,col=0,pairSep=',')
    # findpartner(intActPair,fcoQ,fpartner_coQ,col=1,pairSep=',')
    #
    # '''
    # get partner list
    # '''
    # f3partnerList = 'file_coQ/3partner.list' # (746,)
    # getPartnerlist(fcoQ_partner, fpartner_coQ, f3partnerList)

    '''
    get pfam of coQ, only record tophit pfam
    time 16.931116104125977
    '''
    # f4pfamCoQ = 'file_coQ/4pfamCoQ.tsv'
    # proteinPfam(fcoQ, f4pfamCoQ, tophit=True, item='Pfam')
    '''
    get pfams of coQ
    '''
    # f4pfamsCoQ = 'file_coQ/4pfamsCoQ.tsv'
    # proteinPfam(fcoQ, f4pfamsCoQ, tophit=False, item='Pfam')
    '''
    get find tmp in coQ
    '''
    # ftmpcoQ = 'file_coQ/5tmpcoQ.tsv'
    # getTmp(fcoQ, ftmpcoQ)

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

