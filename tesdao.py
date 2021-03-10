from DatabaseOperation2 import DataOperation
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
def queryPfam(ac,do,tophit=False):
    dic = {'accession':ac,'dbReference.@type':'Pfam'}
    qa = do.QueryObj(dic)
    for q in qa:
        dbReference = q['dbReference']
        # {'@type': 'Pfam', '@id': 'PF18424', 'property': [{'@type': 'entry name', '@value': 'a_DG1_N2'}
        #  {'@type': 'Pfam', '@id': 'PF12743', 'property': [{'@type': 'entry name', '@value': 'ESR1_C'}, {'@type': 'match status', '@value': '1'}]}
        result = getPfam(dbReference)
        return result if not tophit else result[0]
def addProteinPfam():
    pass
def funcPfam(x):
    return x
if __name__ == '__main__':
    ac = 'P03372'
    do = DataOperation('uniprot', 'uniprot_sprot')
    result = queryPfam(ac, do, tophit=True)

    # pa = 'P03372'
    # # projcetion = {'dbReference':True,'_id':False}
    # dic = {'accession':pa,'dbReference.@type':'Pfam'}
    # do = DataOperation('uniprot', 'uniprot_sprot')
    # qa = do.QueryObj(dic)
    # for q in qa:
    #     dbReference = q['dbReference']
    #     # {'@type': 'Pfam', '@id': 'PF18424', 'property': [{'@type': 'entry name', '@value': 'a_DG1_N2'}
    #     #  {'@type': 'Pfam', '@id': 'PF12743', 'property': [{'@type': 'entry name', '@value': 'ESR1_C'}, {'@type': 'match status', '@value': '1'}]}
    #     result = getPfam(dbReference)
    #     print(result)
