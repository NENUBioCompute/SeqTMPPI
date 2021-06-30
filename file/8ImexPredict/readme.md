related _8ImexPredict.py  _15crossvalidate.py
2pair.tsv 2060 qualified tmp-nontmp from /home/jjhnenu/data/PPI/stage2/uniprotPairFromImex.txt
3pair.tsv 2060 drop repeate from 'file/4train/0/all.txt'
4pair.tsv 941 drop repeate from 'file/4train/5CV/data/train_vali.txt'

load 941 [0:941] data from file/8ImexPredict/data/4pair.tsv
load 941 [0:941] data from file/8ImexPredict/data_nega/4pairInfo_subcell_differ_related/2pair.tsv