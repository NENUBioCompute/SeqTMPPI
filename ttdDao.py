# Title     : targetDao.py
# Created by: julse@qq.com
# Created on: 2021/4/5 10:22
# des : TODO
from DatabaseOperation2 import DataOperation
from common import saveList, check_path
import time


def multiSplit(protein,sep = ['/','-',';']):
    for s in sep[:-1]:
        protein = protein.replace(s,sep[-1])
    return protein.split(sep[-1])

def getallProtein():
    '''
    dao
    :param AC:
    :param do:
    :return:
    '''
    do = DataOperation(db_name, table_target)
    projection = {'_id': True, 'UNIPROID':True}
    docs = do.GetALL(projection=projection,limit=0)
    for protein in docs:
        if protein['UNIPROID'] == '':continue
        for pro in multiSplit(protein['UNIPROID']):
            yield pro.strip()
def writeProtins(fout):
    proteins = []
    for protein in getallProtein():
        proteins.extend(protein)
    saveList(getallProtein(),fout)


if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    db_name = 'ttd'
    table_target = 'target'

    fout = 'file/otherfile/2ttd_target_protein.list'
    check_path('file/otherfile/')
    writeProtins(fout)
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)




