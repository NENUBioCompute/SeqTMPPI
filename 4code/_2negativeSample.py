import os

from _1positiveSample import handlePair
from dao import composeTMP_nonTMP, getPairInfo
from dataset import handleRow, calcuSubcell, saveDifferSubcell, simplifyTable
from negativeData import dropPositiveAndRepeate


def main():
    '''
    config path
    '''
    # fpositive = 'file/_1pair.txt'
    fpositive = 'file/1positive/2pair.tsv'
    foutdir = 'file/2negative'
    # os.path.join(foutdir,
    '''
    1. compose pair
    '''
    f1tmp = os.path.join(foutdir,'1tmp.list')
    f1nontmp = os.path.join(foutdir,'1nontmp.list')
    f1pair = os.path.join(foutdir,'1pair.tsv')
    # composeTMP_nonTMP(f1tmp,f1nontmp,f1pair,10)
    composeTMP_nonTMP(f1tmp,f1nontmp,f1pair,100000)

    '''
    2. drop repeate and positive
    '''
    f2pair = os.path.join(foutdir,'2pair_drop_positiveAndrepeate.tsv')
    dropPositiveAndRepeate(f1pair, fpositive, f2pair)

    '''
    3. qualified
    process like positive
    '''
    f3outdir = os.path.join(foutdir,'3qualified')
    handlePair(f2pair, f3outdir,sep='\t',dbname='seqtmppi_negative')

    '''
    4.drop same subcellular
    '''
    f4outdir = os.path.join(foutdir, '4subcellular')
    f32qualified_info = os.path.join(f3outdir,'2tmp_nontmp_info_qualified.tsv')
    f41subcelluar = os.path.join(f4outdir,'1subcelluar.tsv')
    handleRow(f32qualified_info, f41subcelluar,calcuSubcell)
    f42subcelluar_differ = os.path.join(f4outdir,'2subcellular_differ.tsv')
    saveDifferSubcell(f41subcelluar,f42subcelluar_differ)


def handleNegative(foutdir,fpositive=None,num=10,dbname=None,jumpStep=None):
    '''

    :param fpositive:
    :param foutdir:
    :param dbname:
    :param jumpStep:
    :return:

    fpositive = 'file/1positive/2pair.tsv'
    foutdir = 'file/2negative'
    dbname = 'seqtmppi_negative
    jumpStep[1,2]
    '''
    '''
    config path
    '''

    f1tmp = os.path.join(foutdir, '1tmp.list')
    f1nontmp = os.path.join(foutdir, '1nontmp.list')
    f1pair = os.path.join(foutdir, '1pair.tsv')
    f2pair = os.path.join(foutdir, '2pair_drop_positiveAndrepeate.tsv')
    # f3qualified_info = os.path.join(foutdir, '3tmp_nontmp_info_qualified.tsv')
    f3outdir = os.path.join(foutdir, '3qualified')
    f4outdir = os.path.join(foutdir, '4subcellular')
    f32qualified_info = os.path.join(f3outdir, '2tmp_nontmp_info_qualified.tsv')
    f41subcelluar = os.path.join(f4outdir, '1subcelluar.tsv')
    f42subcelluar_differ = os.path.join(f4outdir, '2subcellular_differ.tsv')

    '''
    1. compose pair
    '''
    if jumpStep==None or 1 not in jumpStep:
        # composeTMP_nonTMP(f1tmp,f1nontmp,f1pair,10)
        composeTMP_nonTMP(f1tmp, f1nontmp, f1pair, int(num))

    '''
    2. drop repeate and positive
    '''
    if jumpStep==None or 2 not in jumpStep:
        dropPositiveAndRepeate(f1pair, fpositive, f2pair)

    '''
    3. qualified
    process like positive
    '''
    if jumpStep==None or 3 not in jumpStep:
        handlePair(f3outdir, sep='\t', dbname=dbname,checkTMP=False,
                   jumpStep=[3,4,5], fin=f2pair, f2tmp_nonTtmp_info_qualified=f32qualified_info)


    '''
    4.drop same subcellular
    '''
    if jumpStep==None or 4 not in jumpStep:
        handleRow(f32qualified_info, f41subcelluar, calcuSubcell)
        saveDifferSubcell(f41subcelluar, f42subcelluar_differ)

    '''
    5. save as related form
    '''
    if jumpStep == None or 5 not in jumpStep:
        handlePair(f4outdir, sep='\t', dbname=dbname,jumpStep=[1,2,5],f2tmp_nonTtmp_info_qualified=f42subcelluar_differ)


if __name__ == '__main__':
    # fpositive = 'file/1positive/2pair.tsv'
    # foutdir = 'file/2negative'
    # dbname = 'seqtmppi_negative'
    # num = 10000
    # handleNegative(foutdir, fpositive=fpositive, num=num, dbname=dbname, jumpStep=[1,2,3,4])

    # f4outdir = 'file/2negative/4subcellular'
    # f42subcelluar_differ = os.path.join(f4outdir,'2subcellular_differ.tsv')
    # f2positive = os.path.join(f4outdir,'2pair.tsv')
    # simplifyTable(f42subcelluar_differ, f2positive)

    pass
