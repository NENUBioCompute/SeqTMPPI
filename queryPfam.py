# Title     : queryPfam.py
# Created by: julse@qq.com
# Created on: 2021/3/26 10:09
# des : query pfam from mongodb
import pandas as pd

from DatabaseOperation2 import DataOperation


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


import time

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    fin = '' # protein list file
    fout = '' # protein pfam list file
    proteinPfam(fin, fout, tophit=True, item='Pfam')

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

