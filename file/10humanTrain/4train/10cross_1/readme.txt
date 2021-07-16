验证正负样本是1:1
生成这批数据的代码
来自_10human_crosstrain.py
    # flist = ['file/10humanTrain/3cluster/4posi.tsv','file/10humanTrain/3cluster/4nega.tsv']
    # dirout = 'file/10humanTrain/4train/10cross/'
    # ratios = [1,1]
    # limit = int(40215/11)
    # labels = [1, 0]
    # ComposeData().save(dirout,flist,ratios,limit,labels=None,groupcount=11,repeate=False)

    # check_path('file/10humanTrain/4train/10cross_1/')
    # dirin = 'file/10humanTrain/4train/10cross/'
    # flist = [os.path.join(dirin,'%s/all.txt'%x) for x in range(10)]
    # ftrain = 'file/10humanTrain/4train/10cross_1/train_validate.txt'
    # concatFile(flist,ftrain)
    # ftest = 'file/10humanTrain/4train/10cross_1/test.txt'
    # copyfile(os.path.join(dirin,'%s/all.txt'%10),ftest)
    # fall = 'file/10humanTrain/4train/10cross_1/all.txt'
    # concatFile([ftrain,ftest], fall)