all.txt concat 2pair DIP关于上面几个文件夹物种中的TMP-nonTMP 不包括HP中的蛋白对
    '''
    concat DIP posi exclude HP _8DIPPredict_1.py
    '''
    # dirin = 'file/8DIPPredict/data'
    # fileList = [os.path.join(dirin,eachfile,'2pair.tsv') for eachfile in ['Ecoli', 'Mus', 'Human', 'SC']]
    # fout = os.path.join(dirin,'all.txt')
    # concatFile(fileList, fout)

去冗余做了吗？做了，2pair 与正样本不重复,
3pair 与十折交叉验证的训练集，验证集不重复发

all.txt concat f3pair

start 2021-07-16 10:50:48
origin 1000,file/8DIPPredict/data/Ecoli/2pair.tsv
delete reperate 0,file/10humanTrain/4train/10cross_1/train_validate.txt
save 1000,file/8DIPPredict/data/Ecoli/3pair.tsv

origin 458,file/8DIPPredict/data/Mus/2pair.tsv
delete reperate 0,file/10humanTrain/4train/10cross_1/train_validate.txt
save 458,file/8DIPPredict/data/Mus/3pair.tsv

origin 1002,file/8DIPPredict/data/Human/2pair.tsv
delete reperate 196,file/10humanTrain/4train/10cross_1/train_validate.txt
save 806,file/8DIPPredict/data/Human/3pair.tsv

origin 2363,file/8DIPPredict/data/SC/2pair.tsv
delete reperate 0,file/10humanTrain/4train/10cross_1/train_validate.txt
save 2363,file/8DIPPredict/data/SC/3pair.tsv

origin 26,file/8DIPPredict/data/HP/2pair.tsv
delete reperate 0,file/10humanTrain/4train/10cross_1/train_validate.txt
save 26,file/8DIPPredict/data/HP/3pair.tsv

stop 2021-07-16 10:50:48
time 0.2036724090576172