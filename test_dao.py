from DatabaseOperation2 import DataOperation
from ProteinDealer import Protein
from dao import queryProtein, ensomblePortein


import time

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    pa = 'Q14118'
    projcetion = {'dbReference.text':True,'_id':False}
    dic = {'accession':pa,'dbReference.@type':'pfam'}
    do = DataOperation('uniprot', 'uniprot_sprot')
    qa = do.QueryObj(dic,projcetion= projcetion)
    dbReference = qa['dbReference']
    subcellularLocations = []

    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id': True, 'sequence.@length': True, 'sequence.#text': True, 'keyword.@id': True,
                  'comment.subcellularLocation.location': True}
    AC = 'P82432'
    pro = queryProtein(AC, do, projection=projection)
    pro['accession'] = AC
    proinfo = ensomblePortein(pro)

    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)


