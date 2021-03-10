import os

import pandas as pd

def findpartner(fpair,fin,fout):
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
    df_pair = pd.read_csv(fpair,sep='\t',header=None)
    df1 = pd.DataFrame()
    for idx,row in df_pair.iterrows():
        print(idx)
        for i,r in df.itertuples():
            if row[7]==r:
                print(row[0],row[7])
                df1 = df1.append(row)
    df1.to_csv(fout,header=None,index=None,sep='\t')

def simplifyPairInfo(fin,fout):
    df = pd.read_csv(fin,sep='\t',header=None,index=None)
    df_sim = df[[0,7]]
if __name__ == '__main__':
    pass
    # fpair = 'file/2intAct_pair_norepeat_info.txt'
    # fin = 'file_kegg/all_protein.csv'
    # fout = 'file_kegg/protein_partner.csv'
    # findpartner(fpair, fin, fout)

    # fpair = 'file_kegg/protein_partner.csv'
    # dirin = 'file_kegg/separate'
    # dirout = 'file_kegg/separate_partner'
    # for eachfile in os.listdir(dirin):
    #     fin = os.path.join(dirin,eachfile)
    #     fout = os.path.join(dirout,eachfile)
    #     findpartner(fpair, fin, fout)


    # dirin = 'file_kegg/separate_partner'
    # dirout = 'file_kegg/separate_partner_1'
    # for eachfile in os.listdir(dirin):
    #     fin = os.path.join(dirin,eachfile)
    #     fout = os.path.join(dirout,eachfile)
    #     df = pd.read_csv(fin, sep='\t', header=None)
    #     df_sim = df.loc[:,[0,7]]
    #     df_sim.to_csv(fout, header=None, index=None)
    #
    # dirin = 'file_kegg/separate_partner'
    # dirout = 'file_kegg/separate_partner_2'
    # for eachfile in os.listdir(dirin):
    #     fin = os.path.join(dirin,eachfile)
    #     fout = os.path.join(dirout,eachfile)
    #     df = pd.read_csv(fin, sep='\t', header=None)
    #     df_sim = df.loc[:,[0,1,5,6,7,8,12,13]]
    #     df_sim.to_csv(fout, header=None, index=None, sep='\t')


    # fin = 'file_kegg/protein_partner.csv'
    # fout = 'file_kegg/TMP_contact.csv'
    # df = pd.read_csv(fin, sep='\t', header=None)
    # df_sim = df.loc[:,[0,7]]
    # df_sim.to_csv(fout, header=None, index=None)





