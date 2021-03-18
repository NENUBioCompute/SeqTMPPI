# Title     : _9imgPlot.py
# Created by: julse@qq.com
# Created on: 2021/3/12 15:46
# des : TODO
import matplotlib
import squarify

from tool.treeplot import treemapPlot

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import venn

import pandas as pd
import time
import os

from common import check_path, saveList
import matplotlib
def topNContain(f1posiInfo,fin_tmp_subcell,dirout,num,col,labelCol=0):
    df1 = pd.read_table(f1posiInfo,header=None)
    df2 = pd.read_table(fin_tmp_subcell,header=None)
    for subcellu_name in df2[labelCol][:num]:
        if '(' in subcellu_name:subcellu_name = eval(subcellu_name) #  "('PF00001', '7tm_1')"
        proteinlist = []
        for item in range(df1.shape[0]):
            if subcellu_name in ([] if pd.isna(df1.iloc[item,col]) else eval(df1.iloc[item,col])):
                proteinlist.append(df1.iloc[item,0])
        if isinstance(subcellu_name,tuple):subcellu_name = subcellu_name[1]
        print(subcellu_name,len(proteinlist))
        saveList(proteinlist,os.path.join(dirout, '%s.list'%subcellu_name.replace(' ', '_')))
def plotVenn(fins,names,fout):
    dfs = []
    for x in fins:
        dfs.append(list(pd.read_table(x,header=None)[0].values))
    labels = venn.get_labels(dfs, fill=['number'])
    num = len(fins)
    if num == 2: fig, ax = venn.venn2(labels, names=names)
    elif num == 3: fig, ax = venn.venn3(labels, names=names)
    elif num == 4: fig, ax = venn.venn4(labels, names=names)
    elif num == 5: fig, ax = venn.venn5(labels, names=names)
    elif num == 6: fig, ax = venn.venn6(labels, names=names)
    else:print('only support 2~6 set')
    fig.savefig(fout, bbox_inches='tight')
    plt.close()
def doPlotVenn(f1posiInfo,fin_nontmp_subcell,f5subcellu,fout,num,col,labelCol=0):
    '''

    :param f1posiInfo: fin
    :param fin_nontmp_subcell: fin count
    :param f5subcellu: fout
    :param num: top 5
    :param col: col to be calculate in f1posiInfo
    :param labelCol: label to show in fin_nontmp_subcell
    :param fout: png dir
    :return:
    '''
    topNContain(f1posiInfo, fin_nontmp_subcell, f5subcellu, num, col,labelCol=labelCol)
    fins = [os.path.join(f5subcellu,x) for x in os.listdir(f5subcellu)]
    names = [x.replace('_',' ').split('.')[0] for x in os.listdir(f5subcellu)]
    plotVenn(fins, names, fout)
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    findir = 'file/1positive/1tmp_nontmp/statistic'
    f_length = os.path.join(findir, '1length.tsv')
    f1length_img = os.path.join(findir, '1length_img.png')
    f1length_img_pie = os.path.join(findir, '1length_pie.png')
    f1length_img_plot = os.path.join(findir, '1length_img_plot.png')

    foutdir = 'file/9imgPlot'
    check_path(foutdir)

    '''
    Distribution of Proteins bar
    '''
    # df = pd.read_table(f_length, header=None)
    # df.columns = [0,'Distribution of Proteins\' Length' ]
    # df.hist(bins=[x * 200 for x in range(20)])
    # plt.savefig(f1length_img)

    '''
    f1length_img_plot 
    '''
    # df = pd.read_table(f_length, header=None)
    # df.columns = [0,'Distribution of Proteins\' Length' ]
    # df.plot()
    # plt.savefig(f1length_img_plot)

    '''
    venn
    '''

    # ftest = os.path.join(foutdir, 'test.png')
    #
    # labels = venn.get_labels([range(10), range(5, 15)], fill=['number', 'logic'])
    # fig, ax = venn.venn2(labels, names=['list 1', 'list 2'])
    # fig.savefig(ftest, bbox_inches='tight')
    # plt.close()

    '''
    tmp nontmp species venn
    '''
    # f1venn_species = os.path.join(foutdir, '1venn_species.png')
    f1venn_species_1 = os.path.join(foutdir, '1venn_species_1.png')
    #
    fin_tmp_speceis = 'file/5statistic/positive/8tmp_species_count.tsv'
    fin_nontmp_speceis = 'file/5statistic/positive/8nontmp_species_count.tsv'

    # df1 = pd.read_table(fin_tmp_speceis,header=None)[0]
    # df2 = pd.read_table(fin_nontmp_speceis,header=None)[0]
    #
    # labels = venn.get_labels([df1.values, df2.values], fill=['number'])
    # fig, ax = venn.venn2(labels, names=['Species of TMP', 'Species of nonTMP'])
    # fig.savefig(f1venn_species, bbox_inches='tight')
    # plt.close()

    '''
    tmp
    subcellular 
    ['Cell membrane',
     'Endoplasmic reticulum membrane',
     'Membrane',
     'Cytoplasm',
     'Nucleus']
    '''
    f2subcellu = os.path.join(foutdir, '2subcell')
    check_path(f2subcellu)
    f2subcellu_venn = os.path.join(foutdir, '2tmp_subcell_top5.png')
    fin_tmp_subcell = 'file/5statistic/positive/11TmpSubcellularCount.tsv'

    dirout = 'file/5statistic/positive'
    f1posiInfo = os.path.join(dirout, '1posiInfo.tsv')
    # doPlotVenn(f1posiInfo, fin_tmp_subcell, f2subcellu,f2subcellu_venn, 5, 6, labelCol=0)

    '''
    pfam tmp
    '''
    # f3tmpPfams = 'file/5statistic/positive/5tmpPfams.tsv'
    # f3tmpPfams_rank = 'file/5statistic/positive/5tmpPfamsCount.tsv'
    #
    # f3pfams = os.path.join(foutdir, '3tmp_pfams')
    # check_path(f3pfams)
    # f3pfams_png = os.path.join(foutdir, '3tmp_pfams_top5.png')
    #
    # # topNContain(f3tmpPfams, f3tmpPfams_rank, f3pfams, 5, 1,labelCol=0)
    #
    # fins = [os.path.join(f3pfams,x) for x in os.listdir(f3pfams)]
    # names = [x.replace('_',' ').split('.')[0] for x in os.listdir(f3pfams)]
    # plotVenn(fins, names, f3pfams_png)

    '''
    pfam nontmp
    '''
    # f4nontmpPfams = 'file/5statistic/positive/5nontmpPfams.tsv'
    # f4nontmpPfams_rank = 'file/5statistic/positive/5nontmpPfamsCount.tsv'
    #
    # f4pfams = os.path.join(foutdir, '4nontmp_pfams')
    # check_path(f4pfams)
    # f4pfams_png = os.path.join(foutdir, '4nontmp_pfams_top5.png')
    #
    # topNContain(f4nontmpPfams, f4nontmpPfams_rank, f4pfams, 5, 1,labelCol=0)
    #
    # fins = [os.path.join(f4pfams, x) for x in os.listdir(f4pfams)]
    # names = [x.replace('_', ' ').split('.')[0] for x in os.listdir(f4pfams)]
    # plotVenn(fins, names, f4pfams_png)

    '''
    nontmp
    subcellular 
    '''
    # f5subcellu = os.path.join(foutdir, '5nontmp_subcell')
    # check_path(f5subcellu)
    # f5subcellu_venn = os.path.join(foutdir, '5nontmp_subcell_top5.png')
    # fin_nontmp_subcell = 'file/5statistic/positive/11nonTmpSubcellularCount.tsv'
    # doPlotVenn(f1posiInfo, fin_nontmp_subcell, f5subcellu,f5subcellu_venn, 5, 12, labelCol=0)
    '''
    gpcr g tmp nontmp allprotein
    '''

    # fins = ['file/5statistic/positive/1tmp.list',
    # 'file/5statistic/positive/2gpcr.list',
    # 'file/5statistic/positive/10allGprotein.tsv',
    # 'file/5statistic/positive/1nontmp.list']
    # names = ['Transmembrane Protein','GPCR Protein','G protein','nonTransmembrane Protein']
    #
    # f6proteinType = os.path.join(foutdir, '6proteinType_venn.png')
    # plotVenn(fins, names, f6proteinType)
    '''
    tmp subcell tree plot
    '''
    f2subcellu_venn = os.path.join(foutdir, '7subcell_tree.png')
    fin_tmp_subcell = 'file/5statistic/positive/11TmpSubcellularCount.tsv'
    treemapPlot(fin_tmp_subcell,10, title='')


    pass
    # df = pd.read_table(f3tmpPfams,header=None)
    # df1.to_csv(fout,header=None,index=None,sep='\t')

    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

