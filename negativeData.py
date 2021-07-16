import os

from common import getCol, readIDlist
import pandas as pd

# def dropPositive(fin:str, fpositive:str, fout,saveNum=0):
#     '''
#     淘汰在正样本中出现过的对
#     :param fin:
#     :param fpositive:
#     :param fout:
#     :param saveNum: final saved pair num
#     :return:
#     start 2021-01-20 16:42:18
#     get 100000 pair
#     stop 2021-01-20 16:58:31
#     time 972.4898459911346
#     time 0.2701360771391127
#     '''
#     filename = list('ABCD')
#     getCol(fin,'A',col=0,repeat=True)
#     getCol(fin,'B',col=1,repeat=True)
#     getCol(fpositive,'C',col=0,repeat=True)
#     getCol(fpositive,'D',col=1,repeat=True)
#     AList = readIDlist('A')
#     BList = readIDlist('B')
#     CList = readIDlist('C')
#     DList = readIDlist('D')
#     pairNum = 0
#     with open(fout,'w') as fo:
#         for i in range(len(AList)):
#             positiveFlag = False
#             for j in range(len(CList)):
#                 if AList[i] == CList[j] and BList[i] == DList[j]:
#                     print(AList[i], BList[i], '是正样本')
#                     positiveFlag = True
#                     break
#                 if BList[i] == CList[j] and AList[i] == DList[j]:
#                     positiveFlag = True
#                     break
#             if not positiveFlag:
#                 pairNum = pairNum + 1
#                 fo.write(AList[i]+'\t'+BList[i]+'\n')
#                 fo.flush()
#                 if saveNum==pairNum:break
#     for eachFile in filename:
#         os.remove(eachFile)
#     print('get %d pair '%pairNum)
# def dropSameSubcell(fin, fSubcellDict, fout,sep='\t',title=True):
#     mydict= getSubcelluDict(fSubcellDict)
#     keys = mydict.keys()
#     func = processPair
#     genre = processTXTbyLine(fin, func,sep,title=title)
#     with open(fout,'w')as fo:
#         for g in genre:
#             if g[0] not in keys or g[1] not in keys:continue
#             if ',' not in mydict[g[0]] or ',' not in mydict[g[1]]:continue
#             subA = mydict[g[0]][:-1].split(',')
#             subB = mydict[g[1]][:-1].split(',')
#             if len(set(subA))+len(set(subB)) == len(set(subA+subB)):
#                 fo.write('%s%s%s\n'%(g[0],sep,g[1]))
#                 fo.flush()
def dropPositiveAndRepeate(fin,fbase,fout):
    df = pd.read_csv(fin, sep='\t', header=None)[[0,1]]
    df_base = pd.read_csv(fbase, sep='\t', header=None)[[0,1]].drop_duplicates()
    df_all = pd.concat([df_base,df]).drop_duplicates()
    df_save = df_all.iloc[df_base.shape[0]:,:]
    df_save.to_csv(fout,header=None,index=None,sep='\t')
    print('origin %d,%s'%(df.shape[0],fin))
    print('delete reperate %d,%s'%(df.shape[0]-df_save.shape[0],fbase))
    print('save %d,%s'%(df_save.shape[0],fout))
    print()
    return df_all
def myfunc(x):

    return x
def dropRepeate(fin,fbase,fout,saveNum=0):
    df = pd.read_csv(fin,sep='\t',header=None,index=None)
    df_base = pd.read_csv(fbase,sep='\t',header=None,index=None)
    # df.apply(lambda x:myfunc(x),axis=1)
    s1 = pd.merge(df, df_base, how='inner', on=[0,1])

if __name__ == '__main__':
    pass