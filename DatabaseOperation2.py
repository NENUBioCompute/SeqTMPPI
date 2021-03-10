# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/11/25 14:58
@desc:
"""
"""
Author:Xian Tan
data:19/12/7
"""

import pymongo
from pymongo import MongoClient

class DataOperation:
    def __init__(self, firstname,secondname, address="39.97.240.2", port=32705):
        self.firstname = firstname
        self.secondname = secondname
        self.address = address
        self.port = port
        self.collection = self.Connected()
    def Connected(self):
        client = pymongo.MongoClient(self.address, self.port)
        db = client['admin']
        db.authenticate("root", "@biodata_psydrugkb_nenu_icb_2019_2022#")
        return client[self.firstname][self.secondname]
    def UpdateOne(self,oldvalue, newvalue):
        newvalue = {'$set':newvalue}
        return self.collection.update_one(oldvalue, newvalue)
    def UpSertOne(self,oldvalue, newvalue):
        newvalue = {'$set':newvalue}
        return self.collection.update_one(oldvalue, newvalue,upsert=True)
    def QueryObj(self,dic,projection=None):
        return self.collection.find(dic,projection=projection)
    def QueryOne(self, key, value,projection = None):
        dic = {}
        dic[key] = value
        # print(str(dic))
        return self.collection.find_one(dic,projection=projection)
    def Query(self, key, value,projection = None):
        dic = {}
        dic[key] = value
        # print(str(dic))
        return self.collection.find(dic,projection=projection)
    def GetALL(self,projection = None,limit=3):
        if limit ==0:
            return self.collection.find(projection=projection)
        else:
            return self.collection.find(projection = projection,limit=limit)
    def StorageOne(self,dic):
        return self.collection.insert(dic)
    def StorageIter(self, iter):
        for i in iter:
            self.collection.insert(i)
    def JoinTable(self, piop):
        pass







class DataQuery:

    @staticmethod
    def QueryOne(name, key, value, address="39.97.240.2", port=32705):
        client = pymongo.MongoClient(address,port)
        db=client['admin']
        db.authenticate("root","@biodata_psydrugkb_nenu_icb_2019_2022#")
        collection = client['DrugKB'][name]
        dic={}
        dic[key] = value
        return collection.find_one(dic)