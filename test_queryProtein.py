# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/26 18:24
@desc:
"""
from unittest import TestCase

from DatabaseOperation2 import DataOperation
from dao import ensomblePortein


class TestQueryProtein(TestCase):
    def test_queryProtein(self):
        do = DataOperation('uniprot', 'uniprot_sprot')
        AC = 'P34397'
        protein = None
        # one accession mapping several protein sequence
        projection = {'_id': True, 'sequence.@length': True, 'sequence.#text': True, 'keyword.@id': True,
                      'comment.subcellularLocation.location': True}
        docs = do.Query('accession', AC, projection=projection)
        count = 0
        for doc in docs:
            count = count + 1
            if count > 1:
                #    一个accession 查询到多个蛋白质
                # 保存这个列表
                print('%s is more than one entry' % AC)
            protein = doc
        protein['accession'] = AC
        ensomblePortein(protein)
