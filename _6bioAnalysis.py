# Title     : _6bioAnalysis.py
# Created by: julse@qq.com
# Created on: 2021/2/1 16:13
# des : TODO
# vs: time of top 10
# queryPathwayByKid(pathwayDic,kid,multi=1) 27 s
# queryPathwayByKid(kid) 11.8 s
# Rscript/gene_enrichment.R
import os

from Bio.KEGG import REST
import time
import pandas as pd

from _5statistic import subcelluCount, calculateRatio
from common import saveList, readIDlist, check_path


def mapKid_pathway(kid,df_kid_pid=None):
    pathwayIds = df_kid_pid[df_kid_pid[0]==kid]
    return [] if pathwayIds.empty else pathwayIds[1].values

def queryKid_pathway(fTmpKEGGCount,outListFile):
    df = pd.read_table(fTmpKEGGCount, header=None)
    # kid_pathway = []
    for id,item in df.iterrows():
        # kid_pathway.extend(queryPathwayByKid(item[0]))
        print(id,'query for ',item[0],item[1])
        result = queryPathwayByKid(item[0])
        if result == []:continue
        saveList(result, outListFile,file_mode='a')
def queryPathwayByKid(kid):
    kid_pathway = []
    # pathwayDic = {}
    # result = REST.kegg_get('ath:ArthCp053').read()
    try:
        result = REST.kegg_get(kid).read()
    except:
        print('not found %s'%kid)
        return kid_pathway
    if 'PATHWAY' not in result:return []
    current_section = None
    for line in result.rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == "":
            current_section = section
        if current_section == "PATHWAY":
            pid, pname = line[12:].split("  ")
            kid_pathway.append((kid,pid,pname))
    return kid_pathway
def handleList(func,elem,dropRepeat=True,*args,**kwargs):
    if elem == []: return []
    elem = eval(elem)
    resultList = []
    if isinstance(elem,list):
        for item in elem:
            resultList.extend(func(item,*args,**kwargs))
    else:resultList = func(elem,*args,**kwargs)
    if dropRepeat:
        resultList = list(set(resultList))
    return resultList




############################ query gene info in  hsa pathway ###################
def queryAllPathway(fpathway=None,fpathwayInfo=None,hsa='hsa'):
    human_pathways = REST.kegg_list("pathway", hsa).read()
    repair_pathways = []
    repair_pathways_info = []
    for line in human_pathways.rstrip().split("\n"):
        entry, description = line.split("\t")
        entry = entry.split(':')[1]
        repair_pathways.append(entry)
        repair_pathways_info.append((entry, description))
    if fpathway:saveList(repair_pathways, fpathway)
    if fpathwayInfo:saveList(repair_pathways_info,fpathwayInfo)
    return repair_pathways
def extractGeneFromPathway(pathway):
    # pathway = 'path:hsa00230'
    print('query http://rest.kegg.jp/get/path:%s' % pathway)
    pathway_file = REST.kegg_get(pathway).read()  # query and read each pathway
    # iterate through each KEGG pathway file, keeping track of which section
    # of the file we're in, only read the gene in each pathway
    return extractGeneFromLocalPathway(pathway_file)
def extractGeneFromLocalPathway(pathway_file,item='GENE'):
    current_section = None
    for line in pathway_file.rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == "":
            current_section = section
        if current_section == item:
            if item == "GENE":yield from extractGeneItem(line[12:].strip())
            if item == "DISEASE":yield from extractDiseaseItem(line[12:].strip())
def extractGeneItem(line):
    gene_id = gene_symbol = KO = EC = ''
    if "; " in line:
        gene_identifiers, gene_description = line.split("; ")
        gene_id, gene_symbol = gene_identifiers.split()
    else:
        items = line.split("  ")
        gene_id = items[-2]
        gene_description = items[-1]
    if "[KO:" in gene_description:
        KO = gene_description[gene_description.index('[KO:') + 4:gene_description.index(']')].split(' ')
    if "[EC:" in gene_description:
        EC = gene_description[gene_description.index('[EC:') + 4:gene_description.rindex(']')].split(' ')
    # geneID_geneName_KO_EC.append((pathway.split(':')[0],gene_id, gene_symbol, KO, EC))
    if gene_id != '': yield gene_id, gene_symbol, KO, EC
def extractDiseaseItem(line):
    items = line.split("  ")
    gene_id = items[-2]
    gene_description = items[-1]
    yield gene_id, gene_description
def queryPathway_Gene(foutPathway_Gene,fpathway=None,fpathwayInfo=None,hsa='hsa'):
    '''

    :param foutPathway_Gene:
        0     hsa00010  pathway id
        1         3101  gene id
        2          HK3  gene symbol
        3     [K00844]  KO id
        4    [2.7.1.1]  EC number
    :param fpathway:
    :param fpathwayInfo:
    :param hsa:
    :return:
    '''
    if os.access(fpathway,os.F_OK):
        repair_pathways = readIDlist(fpathway)
    else:
        repair_pathways = queryAllPathway(fpathway=fpathway, fpathwayInfo=fpathwayInfo,hsa=hsa)
    for idx,pathway in enumerate(repair_pathways):
        geneID_geneName_KO_EC = []
        print(idx,end='.')
        if idx<65:continue
        for gene_id, gene_symbol, KO, EC in extractGeneFromPathway(pathway):
            geneID_geneName_KO_EC.append((pathway,gene_id, gene_symbol, KO, EC))
        saveList(geneID_geneName_KO_EC,foutPathway_Gene,file_mode='a')
def funcMapPid_Gid(x,df):
    if pd.isna(x):return []
    plist = []
    try:
        for y in eval(x):
            plist.extend(df[df[1] == int(y)][0].values)
    except:
        print(y)
    return plist
def mergePathway(fpid_gid,finfo,foutInfo):
    df_pid_gid = pd.read_table(fpid_gid, header=None)[[0,1]]
    df = pd.read_table(finfo,header=None)
    df[26] = df[24].apply(lambda x:funcMapPid_Gid(x,df_pid_gid))
    df[27] = df[25].apply(lambda x:funcMapPid_Gid(x,df_pid_gid))
    df.to_csv(foutInfo,header=None,sep='\t',index=None)
def pathwayOverlap(finfo,foutOverlap):
    df = pd.read_table(finfo,header=None)
    df[28] = df[[26,27]].apply(lambda x:list((set(eval(x[26])) & set(eval(x[27])))),axis=1)
    df.to_csv(foutOverlap,header=None,index=None,sep='\t')
def diseasePathway(dirin,fout):
    path_diseaList = []
    for eachfile in os.listdir(dirin):
        with open(os.path.join(dirin,eachfile)) as fo:
            pathway_file = fo.read()
            for id,name in extractGeneFromLocalPathway(pathway_file,item='DISEASE'):
                path_diseaList.append([eachfile.split('.')[0],id,name])
    df = pd.DataFrame(path_diseaList)
    df.to_csv(fout,header=None,index=None,sep='\t')


######################################## end of  handle data from cytoscape hubbaTable ##########################


def funCheckDisease(x,diseaseList):
    diseasePath = []
    for x in eval(x):
        if x in diseaseList:
            diseasePath.append(x)
    return None if diseasePath==[] else diseasePath

def _1main():
    dirin = 'file/5statistic/positive'
    f_posiGeneIDinfo = os.path.join(dirin, '14posiGeneIDinfo.tsv')  #
    f_posiPDB_PDB = os.path.join(dirin, '15Tmp_nonTMP_hasPDB.tsv')  #

    dirKeegOut = 'file/6bioAnalysis/keggDB'
    f_keegdb_1pathway_human = os.path.join(dirKeegOut, '1pathway_human.tsv')
    f_keegdb_1pathway_human_info = os.path.join(dirKeegOut, '1pathway_human_info.tsv')
    f_keegdb_1pathway_gene_info_human = os.path.join(dirKeegOut, '1pathway_gene_info_human.tsv')
    f_keegdb_2pid_gid = os.path.join(dirKeegOut, '2pathwayid_geneid.tsv')
    f_keegdb_3pathway_disease = os.path.join(dirKeegOut, '3pathway_disease.tsv')
    f_keegdb_4diseasePathList = os.path.join(dirKeegOut, '4diseasePath.list')

    dirKeegPathway = os.path.join(dirKeegOut, 'pathwayInfo')

    dirout = 'file/6bioAnalysis/positive'
    f1pahwayInfo = os.path.join(dirout, '1pahwayInfo.tsv')

    f2pahwayOverlap = os.path.join(dirout, '2pahwayOverlap.tsv')
    f2pahwayHasOverlap = os.path.join(dirout, '2pahwayHasOverlap.tsv')
    f2pahwayHasOverlapCount = os.path.join(dirout, '2pahwayHasOverlapCount.tsv')
    f2pahwayHasOverlapRatio = os.path.join(dirout, '2pahwayHasOverlapRatio.tsv')

    f3pairInDiseasePath = os.path.join(dirout, '3pairInDiseasePath.tsv')
    f3pairInDiseasePath_count = os.path.join(dirout, '3pairInDiseasePath_count.tsv')  # (201,)
    f3pairInDiseasePath_ratio = os.path.join(dirout, '3pairInDiseasePath_ratio.tsv')
    f3pairInDiseasePath_ratio_info = os.path.join(dirout, '3pairInDiseasePath_ratio_info.tsv')

    f4caseStudyPair = os.path.join(dirout, '4caseStudyPair.tsv')
    f4caseStudyPair_onlyOnePDB = os.path.join(dirout, '4caseStudyPair_onlyOnePDB.tsv')
    f4caseStudyPair_onlyOnePDB_norepeat = os.path.join(dirout, '4caseStudyPair_onlyOnePDB_norepeat.tsv')

    ################################### kegg db ##############################
    # queryPathway_Gene(f_keegdb_1pathway_gene_info_human,
    #                   fpathway=f_keegdb_1pathway_human,
    #                   fpathwayInfo=f_keegdb_1pathway_human_info,
    #                   hsa='hsa') # 640.85s

    # df_pid_gid = pd.read_table(f_keegdb_1pathway_gene_info_human,header=None)\
    #     .apply(lambda x:pd.Series(eval(x[0])),axis=1)
    # df_pid_gid.to_csv(f_keegdb_2pid_gid,header=None,sep='\t',index=None)

    # diseasePathway(dirKeegPathway, f_keegdb_3pathway_disease)

    ################################## statistic #############################
    # mergePathway(f_keegdb_2pid_gid, f_posiGeneIDinfo,f1pahwayInfo) # 87.1s
    # pathwayOverlap(f1pahwayInfo, f2pahwayOverlap)

    # subcelluCount(f2pahwayHasOverlap, f2pahwayHasOverlapCount, 4) # (231,)
    # calculateRatio(f2pahwayHasOverlapCount,f2pahwayHasOverlapRatio)
    ##################### pair of pathway in disease #########################
    '''
    get disease related pathway list
    '''
    # df = pd.read_table(f_keegdb_3pathway_disease,header=None)
    # diseaseList = df[0].drop_duplicates()
    # diseaseList.to_csv(f_keegdb_4diseasePathList,header=None,index=None,sep='\t')

    '''
    get protein pair in disease related pahway list
    '''
    # diseaseList = readIDlist(f_keegdb_4diseasePathList)
    # df = pd.read_table(f2pahwayHasOverlap,header=None)
    # df[5] = df.apply(lambda x:funCheckDisease(x[4],diseaseList),axis=1)
    # df.dropna()[[0,1,5]].to_csv(f3pairInDiseasePath,header=None,index=None,sep='\t')
    '''
    count and ratio of disease pathway
    '''
    # subcelluCount(f3pairInDiseasePath,f3pairInDiseasePath_count,2)
    # calculateRatio(f3pairInDiseasePath_count,f3pairInDiseasePath_ratio)
    '''
    merge disease name
    '''
    # df1 = pd.read_table(f3pairInDiseasePath_ratio,header=None)
    # df2 = pd.read_table(f_keegdb_3pathway_disease,header=None)
    # df2.columns = [0,3,4]
    # df = df1.merge(df2)
    # df.to_csv(f3pairInDiseasePath_ratio_info,header=None,index=None,sep='\t')
    '''
    case study 
    P00533	P68431	['hsa05131']
    P00533	P27986	['hsa05218', 'hsa05131', 'hsa05212', 'hsa04926', 'hsa05210', 'hsa04014', 'hsa04630', 'hsa05163', 'hsa05214', 'hsa04066', 'hsa05206', 'hsa05224', 'hsa05215', 'hsa05223', 'hsa05165', 'hsa04068', 'hsa05226', 'hsa05225', 'hsa05160', 'hsa04915', 'hsa04510', 'hsa05200', 'hsa04151', 'hsa04015', 'hsa04012', 'hsa05213', 'hsa04810', 'hsa05171']
    O43561	P27986	['hsa04650', 'hsa04660', 'hsa04015', 'hsa04664', 'hsa05135', 'hsa04014', 'hsa04666']
    P49810	P27986	['hsa04722', 'hsa05010']
    P03372	P27986	['hsa04919', 'hsa04915', 'hsa05224', 'hsa04917', 'hsa05200']
    P05556	P27986	['hsa05131', 'hsa04151', 'hsa05135', 'hsa04015', 'hsa04670', 'hsa04510', 'hsa04611', 'hsa04360', 'hsa05165', 'hsa04810', 'hsa05200', 'hsa05222']
    '''

    '''
    screen pair with KEGG pathway in disease and PDB-PDB
    '''

    # df1 = pd.read_table(f3pairInDiseasePath,header=None)
    # df2 = pd.read_table(f_posiPDB_PDB,header=None)
    # df2.columns = [0,1,3,4]
    # df = df1.merge(df2,on=[0,1])
    # df[5] = df[3].apply(lambda x:len(eval(x)))
    # df[6] = df[4].apply(lambda x:len(eval(x)))
    # df.to_csv(f4caseStudyPair,header=None,index=None,sep='\t')
    #
    # df[(df[5]==1)&(df[6]==1)].to_csv(f4caseStudyPair_onlyOnePDB,header=None,index=None,sep='\t')
    # df[(df[5]==1)&(df[6]==1)].drop_duplicates(subset=[3,4]).to_csv(f4caseStudyPair_onlyOnePDB_norepeat,header=None,index=None,sep='\t')
    ######################################## handle data from cytoscape hubbaTable ##########################
    '''
    cytoscape hubbaTable
    find hub node
    '''
    # dir_cytoscape = 'file/6bioAnalysis/cytoscape/'
    # f5_cyto_hub = os.path.join(dir_cytoscape,'HubbaTable.csv')
    # f5_cyto_hub_max100 = os.path.join(dir_cytoscape,'hub_max10000')
    # check_path(f5_cyto_hub_max100)
    #
    # columns = ['name', 'Betweenness', 'BottleNeck', 'Closeness',
    #    'ClusteringCoefficient', 'Degree', 'DMNC', 'EcCentricity', 'EPC', 'MCC',
    #    'MNC', 'Radiality', 'Stress']

    # df = pd.read_table(f5_cyto_hub,sep=',')
    # for idx in df.columns:
    #     if idx == 'name':continue
    #     fout = os.path.join(f5_cyto_hub_max100,'%s.tsv'%idx)
    #     df1 = df[['name',idx]].sort_values(by=[idx],ascending=False)[:10000]
    #     if df1[idx].max()==df1[idx].min():continue
    #     df1.to_csv(fout, index=None, sep='\t')
    #
    # df0 = pd.read_table(os.path.join(f5_cyto_hub_max100,'Betweenness.tsv'))
    # for eachfile in os.listdir(f5_cyto_hub_max100):
    #     if eachfile =='Betweenness.tsv':continue
    #     df = pd.read_table(os.path.join(f5_cyto_hub_max100,eachfile))
    #     df0 = df0.merge(df)
    #     print('%s\t%s\t%s'%(eachfile,df.shape,df0.shape))
    # fout = os.path.join(dir_cytoscape, 'hub_max10000.tsv')
    # df0.to_csv(fout,index=None,sep='\t')



if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    from scipy import stats

    # 需要注意的是16是由17-1得到的
    # set	in set	background	in background
    # a = stats.hypergeom.sf(16, 3586, 141, 59)
    a = stats.hypergeom.sf(16, 3586, 141, 59)
    # 119	148	1735	8041 1.55E-53
    print(a)

    # kegg enrichment

    # df = pd.read_table(f4caseStudyPair_onlyOnePDB,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)
