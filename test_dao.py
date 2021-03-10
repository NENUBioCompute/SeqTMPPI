from DatabaseOperation2 import DataOperation
from dao import queryProtein


def test_get_pfam():
    pa = 'Q14118'
    projcetion = {'dbReference.text':True,'_id':False}
    dic = {'accession':pa,'dbReference.@type':'pfam'}
    do = DataOperation('uniprot', 'uniprot_sprot')
    qa = do.QueryObj(dic,projcetion= projcetion)
    dbReference = qa['dbReference']
    subcellularLocations = []
