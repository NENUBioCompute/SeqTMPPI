# Title     : treeplot.py
# Created by: julse@qq.com
# Created on: 2021/3/13 21:02
# des : TODO

# pip install squarify
import squarify
import matplotlib.pyplot as plt
import pandas as pd

def treemapPlot(fin,num,title=''):
    # title = 'Treemap of Vechile Class'
    # Import Data
    # fin = "https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv"
    df = pd.read_table(fin,header=None)[:num]

    # Prepare Data
    labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
    sizes = df[1].values.tolist()
    colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

    # Draw Plot
    plt.figure(figsize=(12,8), dpi= 80)
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

    # Decorate
    plt.title(title)
    plt.axis('off')
    plt.show()

import time

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    # fin = '../file/9imgPlot/7treeplot/treeplotcase.csv'
    # df_raw = pd.read_csv(fin)
    # df = df_raw.groupby('class').size().reset_index(name='counts')
    # labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
    # sizes = df['counts'].values.tolist()
    # colors = [plt.cm.Spectral(i / float(len(labels))) for i in range(len(labels))]
    #
    # # Draw Plot
    # plt.figure(figsize=(12, 8), dpi=80)
    # squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
    #
    # # Decorate
    # plt.title('ddddddddd')
    # plt.axis('off')
    # plt.show()

    fin_tmp_subcell = '../file/5statistic/positive/11TmpSubcellularCount.tsv'
    treemapPlot(fin_tmp_subcell, 10, title='Treemap of Transmembrane Proteins\' Subcellular Location')
    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

