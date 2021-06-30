# Title     : targetDao.py
# Created by: julse@qq.com
# Created on: 2021/4/5 10:22
# des : TODO
from DatabaseOperation2 import DataOperation
from common import saveList, check_path


def getallProtein(fout):
    '''
    dao
    :param AC:
    :param do:
    :return:
    '''
    do = DataOperation('DrugKB', 'protein')
    projection = {'_id': True, 'accession':True}
    # one accession mapping several protein sequence
    docs = do.GetALL(projection=projection,limit=0)
    proteinlist = [x['accession'][0] if isinstance(x['accession'], list) else x['accession'] for x in docs]
    saveList(proteinlist,fout)


import time

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    fout = 'file/otherfile/1drugtarget.list'
    check_path('file/otherfile/')
    getallProtein(fout)
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)




