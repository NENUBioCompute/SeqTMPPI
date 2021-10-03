构建数据集

有相同亚细胞定位的蛋白交互对作为正样本
没有的作为负样本

基础是
posi: file/3cluster/4posi.tsv 删掉亚细胞定位没有overlap的蛋白对
nega: file/3cluster/4nega.tsv  已经全部是不同亚细胞定位的蛋白对了

亚细胞定位数据来自
file/1positive/3subcellular/1subcellular.tsv  (73784, 19)
file/2negative/4subcellular/2subcellular_differ.tsv

overlap 为空的时候，18 是True
14                                                   []
18                                                 True

需要删除为空的数据，剩余 (13049, 19) 条

去冗余（只保留出现在cluaster中的蛋白对）
    fout_nega = 'file/21samesubcellular/2nega.tsv'
    fout_posi = 'file/21samesubcellular/2posi.tsv'


正负样本组合之后
file/21samesubcellular/0/all.txt 26098 条

划分训练集，测试集
    save 19503 pair to file/21samesubcellular/0/vali_train.txt
    save 1950 pair to file/21samesubcellular/0/test.txt

用五折交叉验证进行实验