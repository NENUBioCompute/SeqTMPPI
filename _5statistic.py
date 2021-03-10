import os
import time

from DatabaseOperation2 import DataOperation
from common import readIDlist, saveList, concatFile
from dao import queryProtein, save, getPfam, queryPfam, findKeyProtein, proteinPfam, findGProtein, getALlGprotein
import pandas as pd

from dataset import handleRow


def findpartner(fpair,fin,fout,col=0):
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
    df_pair = pd.read_csv(fpair,sep='\t',header=None)
    df1 = df_pair.merge(df)
    print(df1.shape)
    df1.to_csv(fout,header=None,index=None,sep='\t')
def proteinCount(fpair,f1=None,f2=None):
    df = pd.read_csv(fpair, sep='\t', header=None)
    if f1:df[0].value_counts().to_csv(f1,header=None,sep='\t')
    if f2: df[1].value_counts().to_csv(f2,header=None,sep='\t')
def matchInfo(finPair,finPairInfo,foutPairInfo):
    df_posi = pd.read_table(finPair,header=None)[[0,1]]
    df_all_info = pd.read_table(finPairInfo,header=None)
    df_posi.columns = [0,7]
    df_posiInfo = pd.merge(df_posi,df_all_info,on=[0,7])
    df_posiInfo.to_csv(foutPairInfo,sep='\t',header=None,index=None)
def findSpecies(f1posiInfo,f8species,f8tmp_species,f8nontmp_species,f8sameSpecies,f8posiSpecies):
    df_info = pd.read_table(f1posiInfo,header=None)
    df_info[19] = df_info[2].apply(lambda x:x.split('_')[1])
    df_info[20] = df_info[8].apply(lambda x:x.split('_')[1])
    df_tmp_speceis = df_info[[0,19]].drop_duplicates()
    df_nontmp_speceis = df_info[[1,20]].drop_duplicates()
    df_same = df_info[df_info[19]==df_info[20]]
    df_tmp_speceis.columns = [0,1]
    df_nontmp_speceis.columns = [0,1]
    df_species = pd.concat([df_tmp_speceis,df_nontmp_speceis]).drop_duplicates()

    df_tmp_speceis.to_csv(f8tmp_species,header=None,index=None,sep='\t')
    df_nontmp_speceis.to_csv(f8nontmp_species,header=None,index=None,sep='\t')
    df_same.to_csv(f8sameSpecies,header=None,index=None,sep='\t')
    df_info.to_csv(f8posiSpecies,header=None,index=None,sep='\t')
    df_species.to_csv(f8species,header=None,index=None,sep='\t')
def relatedSpecies(f8posiSpecies,species,f9human_related,f9human_human):
    # species = 'HUMAN'
    df_info = pd.read_table(f8posiSpecies,header=None)[[0,1,19,20]]
    df_human_related = df_info[(df_info[19]==species) | (df_info[20]==species)]
    df_human = df_info[(df_info[19]==species) & (df_info[20]==species)]
    df_human_related.to_csv(f9human_related,header=None,index=None,sep='\t')
    df_human.to_csv(f9human_human,header=None,index=None,sep='\t')
def species_pair_count(f8posiSpecies,f8species_pair_count):
    pd.read_table(f8posiSpecies,header=None)[[19,20]]\
        .groupby(by=[19,20]) \
        .size().reset_index().sort_values(by=0, ascending=False)\
        .to_csv(f8species_pair_count,header=None,index=None,sep='\t')
def mergetPfam(fleft,fright,fout):
    # df1 = pd.read_table(fleft,header=None)
    # df2 = pd.read_table(fright,header=None)
    # df2 = df2[[1,2]]
    # df2 = df2.drop_duplicates()
    # df1.columns = [1,3]
    # df1.merge(df2).to_csv(fout,sep='\t',header=None,index=None)
    mergeTwo(fleft, fright, fout, left=[1,3], right=[1,2])

def mergeTwo(fleft,fright,fout,left=None,right=None):
    # left = None
    # right = [0, 21]
    df1 = pd.read_table(fleft,header=None)
    if left:df1 = df1[left]
    df2 = pd.read_table(fright,header=None).drop_duplicates()
    if right:df2.columns = right
    df3 = df1.merge(df2)
    df3.to_csv(fout,sep='\t',header=None,index=None)
    print('left,right,merge: ',df1.shape,df2.shape,df3.shape)
def findNotIn(f1tmp, f2gpcr, f3notGpcr):
    df1 = pd.read_csv(f1tmp,header=None)
    df2 = pd.read_csv(f2gpcr,header=None)
    df2 = df1.append(df2)
    df3 = df2.drop_duplicates(keep=False)
    df3.to_csv(f3notGpcr,sep='\t',header=None,index=None)
    print(df3.shape)

def subcelluCount(f8posiSpecies,f11TmpSubcellularCount,col):
    subcellular = []
    for item in pd.read_table(f8posiSpecies,header=None)[col].iteritems():
        subcellular.extend([] if pd.isna(item[1]) else eval(item[1]))
    df2 = pd.Series(subcellular).value_counts()
    print(df2.shape)
    df2.to_csv(f11TmpSubcellularCount,header=None,sep='\t')
def calcuKEGG(x):
    tmp = set() if pd.isna(x[21]) else set(eval(x[21]))
    nontmp = set() if pd.isna(x[22]) else set(eval(x[22]))
    print(tmp & nontmp)
    x[23] = list(tmp & nontmp)
    return x
def dropNul(fin,fout):
    df = pd.read_table(fin,header=None)
    df1 = df.dropna()
    df1.to_csv(fout,header=None,index=None,sep='\t')
    print(df1.shape)
def mergeThree(fpair,ftmpinfo,fnontmpinfo,fout):
    df = pd.read_table(fpair, header=None)[[0,1]]
    df_tmp = pd.read_table(ftmpinfo, header=None)
    df_tmp.columns=[0,2]
    df_nontmp = pd.read_table(fnontmpinfo, header=None)
    df_nontmp.columns=[1,3]
    df_pair = df.merge(df_tmp).merge(df_nontmp).dropna()
    print(df_pair.shape)
    df_pair.to_csv(fout,header=None,index=None,sep='\t')

def getPDBPair(fin,fout):
    df = pd.read_table(fin, header=None)
    rowlist = []
    for row in df.iterrows():
        tmplist = row[1][2]
        nontmplist = row[1][3]
        for tmp in eval(tmplist):
            for nontmp in eval(nontmplist):
                rowlist.append([row[1][0],row[1][1],tmp,nontmp])
    df_new = pd.DataFrame(rowlist)
    df_new.to_csv(fout,header=None,index=None,sep='\t')


def calculateRatio(f11TmpSubcellularCount,f11TmpSubcellularRatio):
    '''

    :param f11TmpSubcellularCount: fin
    :param f11TmpSubcellularRatio: fout
    :return:
    '''
    df = pd.read_table(f11TmpSubcellularCount, header=None)
    total = df[1].sum()
    df[1].astype(float)
    df[2] = df[1].apply(lambda x:x/total)
    df.to_csv(f11TmpSubcellularRatio,header=None,index=None,sep='\t')

    pass


if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    fposi = 'file/3cluster/4posi.tsv'
    fallInfo = 'file/1positive/3subcellular.tsv'
    # P34709	P34708	1
    # P34709	P34691	1
    # Q61824	Q60631	1

    dirout = 'file/5statistic/positive'
    f1tmp = os.path.join(dirout,'1tmp.list')
    f1nontmp = os.path.join(dirout,'1nontmp.list')
    f1all = os.path.join(dirout,'1all.list')
    f1posiInfo = os.path.join(dirout,'1posiInfo.tsv')

    f2gpcr = os.path.join(dirout,'2gpcr.list')
    f3gpcr_con = os.path.join(dirout,'3gpcr_con.tsv')

    f2notGpcr = os.path.join(dirout,'2notGpcr.list')
    f3notGpcr_con = os.path.join(dirout,'3notGpcr_con.list') # 62062

    f3tmpcount = os.path.join(dirout,'3tmpcount.tsv')
    f3nontmpcount = os.path.join(dirout,'3nontmpcount.tsv')
    f4tmpcount = os.path.join(dirout,'4tmpcount.tsv')
    f4nontmpcount = os.path.join(dirout,'4nontmpcount.tsv')

    f5tmpPfam = os.path.join(dirout,'5tmpPfam.tsv')
    f5nontmpPfam = os.path.join(dirout,'5nontmpPfam.tsv')
    f5tmpPfamCount = os.path.join(dirout,'5tmpPfamCount.tsv')
    f5nontmpPfamCount = os.path.join(dirout,'5nontmpPfamCount.tsv')
    f5tmpPfamCount_info = os.path.join(dirout,'5tmpPfamCount_info.tsv')
    f5nontmpPfamCount_info = os.path.join(dirout,'5nontmpPfamCount_info.tsv')

    f5tmpPfams = os.path.join(dirout, '5tmpPfams.tsv')
    f5nontmpPfams = os.path.join(dirout, '5nontmpPfams.tsv')
    f5tmpPfamsCount = os.path.join(dirout, '5tmpPfamsCount.tsv') # 2081
    f5nontmpPfamsCount = os.path.join(dirout, '5nontmpPfamsCount.tsv') # 5643

    f5tmpPfamsRatio = os.path.join(dirout, '5tmpPfamsRatio.tsv') # 2081
    f5nontmpPfamsRatio = os.path.join(dirout, '5nontmpPfamsRatio.tsv') # 5643

    f6signal = os.path.join(dirout, '6signal.list')
    f6signal_contactor = os.path.join(dirout, '6signal_contactor.tsv')

    f7signal = os.path.join(dirout, '7nontmpSignal.list')
    f7signal_contactor = os.path.join(dirout, '7tmp_nontmpSignal.tsv')

    f8tmp_species = os.path.join(dirout, '8tmp_species.tsv')
    f8nontmp_species = os.path.join(dirout, '8nontmp_species.tsv')
    f8species = os.path.join(dirout, '8species.tsv')
    f8sameSpecies = os.path.join(dirout, '8sameSpecies.tsv')
    f8posiSpecies = os.path.join(dirout, '8posiSpecies.tsv')

    f8tmp_species_count = os.path.join(dirout, '8tmp_species_count.tsv')
    f8nontmp_species_count = os.path.join(dirout, '8nontmp_species_count.tsv')
    f8species_count = os.path.join(dirout, '8species_count.tsv')
    f8species_Ratio = os.path.join(dirout, '8species_Ratio.tsv')

    f8species_pair_count = os.path.join(dirout, '8species_pair_count.tsv')

    f9human_related = os.path.join(dirout, '9human_related.tsv')
    f9human_human = os.path.join(dirout, '9human_human.tsv')

    f10Gprotein = os.path.join(dirout, '10Gprotein.tsv')
    f10nontmpGprotein = os.path.join(dirout, '10nontmpGprotein.tsv')
    f10AllGprotein = os.path.join(dirout, '10AllGprotein.tsv')

    f10Gprotein_con = os.path.join(dirout, '10Gprotein_con.tsv') # (16, 3)
    f10tmp_Gprotein = os.path.join(dirout, '10tmp_Gprotein.tsv') # (302, 3)
    f10gpcr_g = os.path.join(dirout, '10gpcr_g.tsv') # (60, 3)

    f10notGprotein = os.path.join(dirout, '10notGprotein.tsv') # (8250, 1)
    f10notnontmpGprotein = os.path.join(dirout, '10notnontmpGprotein.tsv') # (16627, 1)

    f10notGprotein_nontmp = os.path.join(dirout, '10notGprotein_nontmp.tsv') # 64702
    f10tmp_notGprotein = os.path.join(dirout, '10tmp_notGprotein.tsv') # 64702
    f10notGpcr_g = os.path.join(dirout, '10notGpcr_g.tsv') # 182
    f10notGpcr_notG = os.path.join(dirout, '10notGpcr_notG.tsv') # 61880
    f10Gpcr_notG = os.path.join(dirout, '10Gpcr_notG.tsv') # 2822

    f11TmpSubcellularCount = os.path.join(dirout, '11TmpSubcellularCount.tsv') # 260
    f11TmpSubcellularRatio = os.path.join(dirout, '11TmpSubcellularRatio.tsv') # 260

    f12TmpGO = os.path.join(dirout, '12TmpGO.tsv') # 260
    f12nonTmpGO = os.path.join(dirout, '12nonTmpGO.tsv') # 260
    f12TmpGOCount = os.path.join(dirout, '12TmpGOCount.tsv') # 11380
    f12nonTmpGOCount = os.path.join(dirout, '12nonTmpGOCount.tsv') # 17586
    f12TmpGORatio = os.path.join(dirout, '12TmpGORatio.tsv') # 11380
    f12nonTmpGORatio = os.path.join(dirout, '12nonTmpGORatio.tsv') # 17586

    f13TmpKEGG = os.path.join(dirout, '13TmpKEGG.tsv') # 260
    f13nonTmpKEGG = os.path.join(dirout, '13nonTmpKEGG.tsv') # 260
    f13TmpKEGGCount = os.path.join(dirout, '13TmpKEGGCount.tsv') # 8532
    f13nonTmpKEGGCount = os.path.join(dirout, '13nonTmpKEGGCount.tsv') # 17678
    f13posiKEGGinfo = os.path.join(dirout, '13posiKEGGinfo.tsv') # (64939, 24)

    f14TmpGeneID = os.path.join(dirout, '14TmpGeneID.tsv')  #
    f14nonTmpGeneID = os.path.join(dirout, '14nonTmpGeneID.tsv')  #
    f14TmpGeneIDCount = os.path.join(dirout, '14TmpGeneIDCount.tsv')  # 8055
    f14nonTmpGeneIDCount = os.path.join(dirout, '14nonTmpGeneIDCount.tsv')  # 16845
    f14posiGeneIDinfo = os.path.join(dirout, '14posiGeneIDinfo.tsv') # (64939, 24)

    f15TmpPDB = os.path.join(dirout, '15TmpPDB.tsv') #
    f15nonTmpPDB = os.path.join(dirout, '15nonTmpPDB.tsv') #
    f15TmpPDBCount = os.path.join(dirout, '15TmpPDBCount.tsv') # 11473
    f15nonTmpPDBCount = os.path.join(dirout, '15nonTmpPDBCount.tsv') # 47124
    f15TmpPDBRatio = os.path.join(dirout, '15TmpPDBRatio.tsv') # 11473
    f15nonTmpPDBRatio = os.path.join(dirout, '15nonTmpPDBRatio.tsv') # 47124

    f15TmpPDBnotNul = os.path.join(dirout, '15TmpPDBnotNul.tsv')  # (1873, 2)
    f15nonTmpPDBnotNul = os.path.join(dirout, '15nonTmpPDBnotNul.tsv')  # (6781, 2)

    f15Tmp_nonTMP_hasPDB = os.path.join(dirout, '15Tmp_nonTMP_hasPDB.tsv')  # (14544, 4)
    f15TmpPDB_nontmpPDB = os.path.join(dirout, '15TmpPDB_nontmpPDB.tsv')  # (6781, 2)
    f15TmpPDB_nontmpPDB_count = os.path.join(dirout, '15TmpPDB_nontmpPDB_count.tsv')  # (6781, 2)
    f15TmpPDB_nontmpPDB_ratio = os.path.join(dirout, '15TmpPDB_nontmpPDB_ratio.tsv')  # (6781, 2)



########################## temp ###################################################
    dirout_temp = 'file/5statistic/temp'
    f_temp_1_Gprotein_name = os.path.join(dirout_temp, '1_Gprotein_name.tsv') # 417
    f_temp_2swissprot_g = os.path.join(dirout_temp, '2uniprot-guanine+nucleotide-binding-filtered-reviewed_yes.list') # 417
    f_temp_2swissprot_g_name = os.path.join(dirout_temp, '2swissprot_g_name.tsv') # 417


    '''
    get tmp and nontmp from pair
    '''
    # df = pd.read_table(fposi,header=None)
    # tmp = df[0].drop_duplicates()
    # nontmp = df[1].drop_duplicates()
    # tmp.to_csv(f1tmp,header=None,index=None)
    # nontmp.to_csv(f1nontmp,header=None,index=None)

    # concatFile([f1tmp,f1nontmp],f1all)
    # notsvaelist = save(readIDlist(f1all), 'seqtmppi_positive')

    # matchInfo(fposi, fallInfo, f1psoiInfo)

######################################### GPCR #########################################
    # '''
    # get gpcr list
    # '''
    # findKeyProtein(f1tmp, f2gpcr, 'KW-0297')
    # '''
    # get gpcr-contactor
    # '''
    # findpartner(fposi, f2gpcr, f3gpcr_con)

    # findNotIn(f1tmp,f2gpcr,f2notGpcr)
    # findpartner(fposi, f2notGpcr, f3notGpcr_con)


####################################### count #######################################

    # proteinCount(f3gpcr_con, f1=f3tmpcount, f2=f3nontmpcount)
    # proteinCount(fposi, f1=f4tmpcount, f2=f4nontmpcount)

####################################### pfam ########################################

    # proteinPfam(f1tmp, f5tmpPfam)
    # proteinPfam(f1nontmp, f5nontmpPfam)
    # proteinCount(f5tmpPfam, f2=f5tmpPfamCount)
    # proteinCount(f5nontmpPfam,f2=f5nontmpPfamCount)

    # mergetPfam(f5tmpPfamCount, f5tmpPfam, f5tmpPfamCount_info)
    # mergetPfam(f5nontmpPfamCount, f5nontmpPfam, f5nontmpPfamCount_info)

    # proteinPfam(f1tmp, f5tmpPfams,tophit=False) # 83.83 s
    # proteinPfam(f1nontmp, f5nontmpPfams,tophit=False) # 166.54 s

    # subcelluCount(f5tmpPfams, f5tmpPfamsCount,1)
    # subcelluCount(f5nontmpPfams, f5nontmpPfamsCount,1)

    # calculateRatio(f5tmpPfamsCount,f5tmpPfamsRatio)
    # calculateRatio(f5nontmpPfamsCount,f5nontmpPfamsRatio)

####################################### signal ################################
    # '''
    # get gpcr list
    # '''
    # findKeyProtein(f1tmp, f6signal, 'KW-0732')
    # '''
    # get gpcr-contactor
    # '''
    # findpartner(fposi, f6signal, f6signal_contactor)

###################################### tmp- singal ################################
    '''
    get gpcr list 
    '''

    #
    # (f1nontmp, f7signal, 'KW-0732')
    '''
    get gpcr-contactor
    '''
    # findpartner(fposi, f7signal, f7signal_contactor,col=1)

################################## species ############################
    # findSpecies(f1posiInfo,f8species, f8tmp_species, f8nontmp_species, f8sameSpecies,f8posiSpecies)
    # proteinCount(f8tmp_species,f2=f8tmp_species_count)
    # proteinCount(f8nontmp_species,f2=f8nontmp_species_count)
    # proteinCount(f8species,f2=f8species_count)
    # calculateRatio(f8species_count,f8species_Ratio)

    # species_pair_count(f8posiSpecies, f8species_pair_count)


################################# species human #############################
    # species = 'HUMAN'
    # relatedSpecies(f8posiSpecies, species, f9human_related, f9human_human)

################################# G-protein ########################
    # findGProtein(f1tmp,f10Gprotein) # 2
    # findGProtein(f1nontmp,f10nontmpGprotein) # 79

    # findpartner(fposi,f10Gprotein,f10Gprotein_con,col=0) # (16, 3)
    # findpartner(fposi,f10nontmpGprotein,f10tmp_Gprotein,col=1) # (302, 3)
    # findpartner(f3gpcr_con,f10nontmpGprotein,f10gpcr_g,col=1) # (60, 3)

    # findNotIn(f1tmp,f10Gprotein,f10notGprotein) # (8250, 1)
    # findNotIn(f1nontmp,f10nontmpGprotein,f10notnontmpGprotein) # (16627, 1)
    # findpartner(fposi,f10notGprotein,f10notGprotein_nontmp,col=0) # (64923, 3)
    # findpartner(fposi,f10notnontmpGprotein,f10tmp_notGprotein,col=1) # (64637, 3)

    # findpartner(f3notGpcr_con,f10nontmpGprotein,f10notGpcr_g,col=1) # (242, 3)
    # findpartner(f3notGpcr_con,f10notnontmpGprotein,f10notGpcr_notG,col=1) # (61820, 3)
    # findpartner(f3gpcr_con,f10notnontmpGprotein,f10Gpcr_notG,col=1) # (2817, 3)
############################## subcellularlocation ##################
    # subcelluCount(f8posiSpecies, f11TmpSubcellularCount,6)
    # calculateRatio(f11TmpSubcellularCount,f11TmpSubcellularRatio)

    # f11TmpSubcellularRatio
############################## go ###################################

    # proteinPfam(f1tmp, f12TmpGO, tophit=False, item='GO')
    # proteinPfam(f1nontmp, f12nonTmpGO, tophit=False, item='GO') # 190.83

    # subcelluCount(f12TmpGO, f12TmpGOCount,1)
    # subcelluCount(f12nonTmpGO, f12nonTmpGOCount,1)

    # calculateRatio(f12TmpGOCount,f12TmpGORatio)
    # calculateRatio(f12nonTmpGOCount,f12nonTmpGORatio)

    ############################## kegg #########################

    # proteinPfam(f1tmp, f13TmpKEGG, tophit=False, item='KEGG')
    # proteinPfam(f1nontmp, f13nonTmpKEGG, tophit=False, item='KEGG') # 190.83

    # subcelluCount(f13TmpKEGG, f13TmpKEGGCount,1)
    # subcelluCount(f13nonTmpKEGG, f13nonTmpKEGGCount,1)

    # left, right, merge: (64939, 21)(8252, 2)(64939, 22)
    # left, right, merge: (64939, 22)(16706, 2)(64939, 23)

    # mergeTwo(f8posiSpecies, f13TmpKEGG, f13posiKEGGinfo, left=None, right=[0,21])
    # mergeTwo(f13posiKEGGinfo, f13nonTmpKEGG, f13posiKEGGinfo, left=None, right=[1,22])
    # handleRow(f13posiKEGGinfo, f13posiKEGGinfo, calcuKEGG) # 122.0

##################################### GeneID #######################################
    # proteinPfam(f1tmp, f14TmpGeneID, tophit=False, item='GeneID') # 63.2
    # proteinPfam(f1nontmp, f14nonTmpGeneID, tophit=False, item='GeneID') # 190.84

    # (8055,)
    # (16845,)

    # subcelluCount(f14TmpGeneID, f14TmpGeneIDCount,1)
    # subcelluCount(f14nonTmpGeneID, f14nonTmpGeneIDCount,1)

    # left,right,merge:  (64939, 24) (8252, 2) (64939, 25)
    # left,right,merge:  (64939, 25) (16706, 2) (64939, 26)
    # time 152.2784025669098

    # mergeTwo(f13posiKEGGinfo, f14TmpGeneID, f14posiGeneIDinfo, left=None, right=[0,24])
    # mergeTwo(f14posiGeneIDinfo, f14nonTmpGeneID, f14posiGeneIDinfo, left=None, right=[1,25])


    ##################################### all G protein ########################
    # getALlGprotein(f_temp_1_Gprotein_name) # 417
    # findGProtein(f_temp_2swissprot_g, f_temp_2swissprot_g_name)

    ##################################### PDB ##################################
    # (11473,)
    # (47124,)
    # stop 2021-02-06 10:11:30
    # time 207.28172993659973

    # proteinPfam(f1tmp, f15TmpPDB, tophit=False, item='PDB')
    # proteinPfam(f1nontmp, f15nonTmpPDB, tophit=False, item='PDB')
    #
    # subcelluCount(f15TmpPDB, f15TmpPDBCount,1) # 11473
    # subcelluCount(f15nonTmpPDB, f15nonTmpPDBCount,1) # 47124

    # calculateRatio(f15TmpPDBCount,f15TmpPDBRatio)
    # calculateRatio(f15nonTmpPDBCount,f15nonTmpPDBRatio)

    # dropNul(f15TmpPDB, f15TmpPDBnotNul)     # (1873, 2)
    # dropNul(f15nonTmpPDB, f15nonTmpPDBnotNul) # (6781, 2)

    # mergeThree(fposi, f15TmpPDBnotNul, f15nonTmpPDBnotNul, f15Tmp_nonTMP_hasPDB) # (14544, 4)

    # getPDBPair(f15Tmp_nonTMP_hasPDB, f15TmpPDB_nontmpPDB)

    '''
    time 249.97137308120728
    '''
    # df = pd.read_table(f15TmpPDB_nontmpPDB, header=None)
    # df1 = df.apply(lambda x:(x[2],x[3]),axis=1)
    # df1.value_counts().to_csv(f15TmpPDB_nontmpPDB_count,header=None,sep='\t')
    #
    # calculateRatio(f15TmpPDB_nontmpPDB_count,f15TmpPDB_nontmpPDB_ratio)


    # df = pd.read_table(f15Tmp_nonTMP_hasPDB, header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)



