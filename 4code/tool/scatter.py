# Title     : scatter.py
# Created by: julse@qq.com
# Created on: 2021/3/13 19:56
# des : test for plot
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



#导入数据
midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

#准备标签列表和颜色列表

#不重复地提取出category列中的种类
categories = np.unique(midwest['category'])
#使用列表推导式来生成颜色标签,更多光谱点击 (https://matplotlib.org/tutorials/colors/colormaps.html)
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

#建立画布
plt.figure(figsize=(16, 10)   #绘图尺寸
           , dpi=100          #图像分辨率
           , facecolor='w'    #图像的背景颜色，设置为白色，默认也是白色
           , edgecolor='k'    #图像的边框颜色，设置为黑色，默认也是黑色
          )

#循环绘图
for i, category in enumerate(categories):
    plt.scatter('area'    #横轴
    , 'poptotal'   #纵轴
    , data=midwest.loc[midwest.category==category, :]
    , s=20
    , c=np.array(colors[i]).reshape(1,-1)
    , label=str(category))


#对图像进行装饰
#plt.gca() 获取当前的子图，如果当前没有任何子图的话，就创建一个新的子图
plt.gca().set(xlim=(0, 0.12), ylim=(0, 80000))  #控制横纵坐标的范围
plt.xticks(fontsize=12)  #坐标轴上的标尺的字的大小
plt.yticks(fontsize=12)
plt.ylabel('Population',fontsize=22)  #坐标轴上的标题和字体大小
plt.xlabel('Area',fontsize=22)
plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)  #整个图像的标题和字体的大小
plt.legend(fontsize=12)  #图例的字体大小
plt.show()
