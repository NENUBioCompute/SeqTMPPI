# Title     : _9imgPlot1.py
# Created by: julse@qq.com
# Created on: 2021/4/6 10:05
# des : bar plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time

import seaborn as sns

from common import check_path
import os

def drowBarplot(fin,fout_img,columns,top=10,ratio= True,title = '',figsize=(28,5)):
    '''

    :param fin:
    :param fout_img:
    :param columns:
    :param top:
    :return:

    dirout = 'file/9imgPlot/8barplot'
    check_path(dirout)
    fout_img = os.path.join(dirout,'1nontmpPfamCount.png')
    fin = 'file/5statistic/positive/5nontmpPfamsRatio.tsv'
    columns = ['Protein Family','Count','Ratio']

    drowBarplot(fin, fout_img, columns, top=10)
    '''
    df = pd.read_table(fin,header=None)[:top]
    df.columns = columns
    x = df[columns[1]]
    sns.set(font_scale=1.6,style='white',context='paper')
    plt.figure(figsize=figsize)
    # plt.figure(dpi = 300, figsize=figsize)
    plt.autoscale()
    plt.xlim(left=min(x)*0.8,right=max(x))
    # ax = sns.barplot(x=df[columns[2]],y=df[columns[0]],palette=sns.hls_palette(8,l=.7,s=.9),dodge=False)
    ax = sns.barplot(x=x,y=df[columns[0]],palette=sns.color_palette('Blues_r'),dodge=True)
    ax.set(title=title)  # title barplot# label each bar in barplot
    # ax = addDatalabel(df,ax,ratio)
    for idx, p in enumerate(ax.patches):
        height = p.get_height()  # height of each horizontal bar is the same
        width = p.get_width()  # width (average number of passengers)
        # adding text to each bar
        # if ratio:
        #     s = '%.3f ( %.0f%% )' % (round(df.iloc[idx, 2], 3) * 100,width)
        # else:
        #     s = '%.0f' % (width)
        ax.text(x=width + 3,  # x-coordinate position of data label, padded 3 to right of bar
                y=p.get_y() + (height / 2),
                # # y-coordinate position of data label, padded to be in the middle of the bar
                s='%.0f ( %.3f%% )'%(width,round(df.iloc[idx,2],3)*100),
                va='center')  # sets vertical alignment (va) to center

    # plt.show()
    plt.savefig(fout_img,bbox_inches='tight')
def addDatalabel(df,ax,ratio):
    for idx, p in enumerate(ax.patches):
        height = p.get_height()  # height of each horizontal bar is the same
        width = p.get_width()  # width (average number of passengers)
        # adding text to each bar
        if ratio:
            s = '%.3f ( %.0f%% )' % (round(df.iloc[idx, 2], 3) * 100,width)
        else:
            s = '%.0f' % (width)
        ax.text(x=width + 3,  # x-coordinate position of data label, padded 3 to right of bar
                y=p.get_y() + (height / 2),
                # # y-coordinate position of data label, padded to be in the middle of the bar
                s=s,
                va='center')  # sets vertical alignment (va) to center
    return ax

def drowSubBarplot(fin,columns,myax,top=10,title = '',colors = 'Blues_r'):
    '''

    :param fin:
    :param fout_img:
    :param columns:
    :param top:
    :return:

    dirout = 'file/9imgPlot/8barplot'
    check_path(dirout)
    fout_img = os.path.join(dirout,'1nontmpPfamCount.png')
    fin = 'file/5statistic/positive/5nontmpPfamsRatio.tsv'
    columns = ['Protein Family','Count','Ratio']

    drowBarplot(fin, fout_img, columns, top=10)
    '''
    df = pd.read_table(fin,header=None)[:top]
    df.columns = columns
    x = df[columns[1]]
    # plt.figure(figsize=figsize)
    # plt.figure(dpi = 300, figsize=figsize)
    plt.autoscale()
    # plt.xlim(left=min(x)*0.8,right=max(x))
    # ax = sns.barplot(x=df[columns[2]],y=df[columns[0]],palette=sns.hls_palette(8,l=.7,s=.9),dodge=False)
    ax = sns.barplot(x=x,y=df[columns[0]],palette=sns.color_palette(colors,n_colors=top),dodge=True,ax=myax)
    ax.set(title=title)  # title barplot# label each bar in barplot
    # ax = addDatalabel(df,ax,ratio)
    for idx, p in enumerate(ax.patches):
        height = p.get_height()  # height of each horizontal bar is the same
        width = p.get_width()  # width (average number of passengers)
        # adding text to each bar
        # if ratio:
        #     s = '%.3f ( %.0f%% )' % (round(df.iloc[idx, 2], 3) * 100,width)
        # else:
        #     s = '%.0f' % (width)
        ax.text(x=width + 3,  # x-coordinate position of data label, padded 3 to right of bar
                y=p.get_y() + (height / 2),
                # # y-coordinate position of data label, padded to be in the middle of the bar
                s='%.0f ( %.3f%% )'%(width,round(df.iloc[idx,2],3)*100),
                va='center')  # sets vertical alignment (va) to center

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # plt.show()
    # plt.savefig(fout_img,bbox_inches='tight')
    return fig,plt
if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()


    # fin = 'file/5statistic/positive/5nontmpPfamsRatio.tsv'
    # df = pd.read_table(fin,header=None)[:10]
    # plt.barh(df[0], df[1], label='hello')
    # plt.legend()
    # plt.show()
    #
    '''
    start
    '''
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # fout_img = os.path.join(dirout,'5nontmpPfamsRatio.png')
    # fin = 'file/5statistic/positive/5nontmpPfamsRatio.tsv'
    # columns = ['Family','Count','Ratio']
    # title = 'Top 10 Families of nonTransmembrane Protein'
    # drowBarplot(fin, fout_img, columns, top=10,title = title)
    #
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # fout_img = os.path.join(dirout,'5tmpPfamsRatio.png')
    # fin = 'file/5statistic/positive/5tmpPfamsRatio.tsv'
    # columns = ['Families','Count','Ratio']
    # title = 'Top 10 Families of Transmembrane Protein'
    # drowBarplot(fin, fout_img, columns, top=10,title=title)
    #
    #
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # fout_img = os.path.join(dirout,'8tmp_species_Ratio.png')
    # fin = 'file/5statistic/positive/8tmp_species_Ratio.tsv'
    # columns = ['Species','Count','Ratio']
    # title = 'Top 10 Species of Transmembrane Protein'
    # drowBarplot(fin, fout_img, columns, top=10,title=title)
    #
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # fout_img = os.path.join(dirout,'8nontmp_species_Ratio.png')
    # fin = 'file/5statistic/positive/8nontmp_species_Ratio.tsv'
    # columns = ['Species','Count','Ratio']
    # title = 'Top 10 Species of nonTransmembrane Protein'
    # drowBarplot(fin, fout_img, columns, top=10,title=title)
    #
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # fout_img = os.path.join(dirout,'12TmpGORatio.png')
    # fin = 'file/5statistic/positive/12TmpGORatio.tsv'
    # columns = ['GO items','Count','Ratio']
    # title = 'Top 10 GO items of Transmembrane Protein'
    # drowBarplot(fin, fout_img, columns, top=10,title=title)
    #
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # fout_img = os.path.join(dirout,'12nonTmpGORatio.png')
    # fin = 'file/5statistic/positive/12nonTmpGORatio.tsv'
    # columns = ['GO items', 'Count', 'Ratio']
    # title = 'Top 10 GO items of nonTransmembrane Protein'
    # drowBarplot(fin, fout_img, columns, top=10,title=title)
####################################################################################################
    # dirout = 'file/9imgPlot/8barplot'
    # check_path(dirout)
    # files = ['8tmp_species_Ratio','8nontmp_species_Ratio','5tmpPfamsRatio','5nontmpPfamsRatio','12TmpGORatio','12nonTmpGORatio','11TmpSubcellularRatio','11nonTmpSubcellularRatio']
    # # fout_img = [os.path.join(dirout, '%s.png'%x) for x in files]
    # fout_img = os.path.join(dirout, '6in1.png')
    # fin = ['file/5statistic/positive/%s.tsv' %x for x in files]
    # ylabels = ['Species','Species','Families','Families','GO items','GO items','Subcellularlocations','Subcellularlocations']
    # # colors = ['Blues_r','Blues_r','GnBu_r','GnBu_r','Greens_r','coolwarm']
    # colors = 'Blues_r'
    # columns = [['%s'%x,'Count','Ratio'] for x in ylabels]
    # title = ['Top 10 Species of Transmembrane Protein',
    #          'Top 10 Species of nonTransmembrane Protein',
    #          'Top 10 Families of Transmembrane Protein',
    #          'Top 10 Families of nonTransmembrane Protein',
    #          'Top 10 GO items of Transmembrane Protein',
    #          'Top 10 GO items of nonTransmembrane Protein',
    #          'Top 10 Subcellularlocations of the TMP-nonTMP Interactions',
    #          'Top 10 Subcellularlocations of nonTransmembrane Protein',
    #          ]
    # top = 10
    # figsize = (30,10)

    # sns.set(font_scale=1.6,style='white',context='paper')
    # fig, axes = plt.subplots(6, 1,dpi = 300, figsize=(21,28))  # fig是整个画布，axes是子图,3，2表示3行两列

    # for idx in range(6):
    #     # fout_img = os.path.join(dirout, '%s.png' % files[idx])
    #     # fin = 'file/5statistic/positive/%s.tsv' % files[idx]
    #     if idx%2 ==0: fig, axes = plt.subplots(2, 1,dpi = 300, figsize=(10,10))  # fig是整个画布，axes是子图,3，2表示3行两列
    #
    #     fig,plt = drowSubBarplot(fin[idx], columns[idx], axes[idx%2], top=10, title=title[idx],colors=colors)
    #     plt.subplots_adjust(hspace=0.4)
    #     if idx%2 ==1: plt.savefig(os.path.join(dirout, '%s.png'%ylabels[idx]),bbox_inches='tight')

    # idx = 6
    # fig, axes = plt.subplots(1, 1, dpi=300, figsize=(10, 5))  # fig是整个画布，axes是子图,3，2表示3行两列
    # fig, plt = drowSubBarplot(fin[idx], columns[idx], axes, top=10, title=title[idx], colors=colors)
    # plt.subplots_adjust(hspace=0.4)
    # plt.savefig(os.path.join(dirout, '%s.png' % ylabels[idx]), bbox_inches='tight')

####################################################################################################
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)
