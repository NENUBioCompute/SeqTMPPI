tips:
- files from _1* to _5* records part of our methods steps
- tool/_8DIPPredict_support.py provides a guid to construct a balanced dataset of TMP-nonTMP for evaluating the model with whole matrix
- _5DIPPredict.py provides a guid to predicting with our well trained model

quickly start with this:
`python _6predict.py`

the dataset was processed on cpu
the model was trained on gpu

requirement:
pip install -r requirements_cpu.txt
pip install -r requirements_gpu.txt


content tree

E:.
│  readme.txt
│
├─1images
│      10Top 10 important proteins by cytohubba..png
│      11Top 7 most significant protein modules.png
│      12Structure of the architecture.png
│      1Distribution of the top 10 species of the proteins.png
│      2Distribution of the top 10 protein families..png
│      3Overlaps in Different Protein Types.png
│      4Distribution of the top 10 subcellular locations of the interactions.png
│      5GO annotation of Transmembrane proteins.png
│      6GO annotation of non-Transmembrane proteins..png
│      7KEGG pathway enrichment of TMPs.png
│      8KEGG pathway enrichment of nonTMPs..png
│      9Network Visualized on TMP-nonTMP interaction pairs.png
│
├─2model
│  │  result.csv
│  │
│  ├─0
│  │  │  acc.png
│  │  │  loss.png
│  │  │  matthews_correlation.png
│  │  │  metric_F1score.png
│  │  │  metric_precision.png
│  │  │  metric_recall.png
│  │  │  _evaluate.txt
│  │  │  _history_dict.txt
│  │  │  _my_model.h5
│  │  │  _my_model.json
│  │  │
│  │  ├─test
│  │  │      log.txt
│  │  │      result.csv
│  │  │
│  │  ├─testDIP
│  │  │  ├─Ecoli
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  ├─HP
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  ├─Human
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  ├─Mus
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  └─SC
│  │  │          log.txt
│  │  │          result.csv
│  │  │
│  │  ├─testDIP1
│  │  │  │  result.csv
│  │  │  │
│  │  │  ├─Ecoli
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  ├─HP
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  ├─Human
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  ├─Mus
│  │  │  │      log.txt
│  │  │  │      result.csv
│  │  │  │
│  │  │  └─SC
│  │  │          log.txt
│  │  │          result.csv
│  │  │
│  │  ├─testHuman
│  │  │      log.txt
│  │  │      result.csv
│  │  │
│  │  └─testHuman01
│  │          log.txt
│  │          result.csv
│  │
│  ├─1
│  │  │  acc.png
│  │  │  loss.png
│  │  │  matthews_correlation.png
│  │  │  metric_F1score.png
│  │  │  metric_precision.png
│  │  │  metric_recall.png
│  │  │  _evaluate.txt
│  │  │  _history_dict.txt
│  │  │  _my_model.h5
│  │  │  _my_model.json
│  │  │
│  │  └─test
│  │          log.txt
│  │          result.csv
│  │
│  ├─2
│  │  │  acc.png
│  │  │  loss.png
│  │  │  matthews_correlation.png
│  │  │  metric_F1score.png
│  │  │  metric_precision.png
│  │  │  metric_recall.png
│  │  │  _evaluate.txt
│  │  │  _history_dict.txt
│  │  │  _my_model.h5
│  │  │  _my_model.json
│  │  │
│  │  └─test
│  │          log.txt
│  │          result.csv
│  │
│  ├─3
│  │  │  acc.png
│  │  │  loss.png
│  │  │  matthews_correlation.png
│  │  │  metric_F1score.png
│  │  │  metric_precision.png
│  │  │  metric_recall.png
│  │  │  _evaluate.txt
│  │  │  _history_dict.txt
│  │  │  _my_model.h5
│  │  │  _my_model.json
│  │  │
│  │  └─test
│  │          log.txt
│  │          result.csv
│  │
│  └─4
│      │  acc.png
│      │  loss.png
│      │  matthews_correlation.png
│      │  metric_F1score.png
│      │  metric_precision.png
│      │  metric_recall.png
│      │  _evaluate.txt
│      │  _history_dict.txt
│      │  _my_model.h5
│      │  _my_model.json
│      │
│      └─test
│              log.txt
│              result.csv
│
├─3dataset
│  ├─DIP_homo
│  │  │  2pair.fasta
│  │  │
│  │  └─0
│  │          all.txt
│  │
│  ├─DIP_mus
│  │  │  2pair.fasta
│  │  │
│  │  └─0
│  │          all.txt
│  │
│  └─TMPPI_bench
│      │  1all.fasta
│      │
│      ├─0
│      │      all.txt
│      │      test.txt
│      │      train.txt
│      │      validate.txt
│      │
│      ├─1
│      │      all.txt
│      │      test.txt
│      │      train.txt
│      │      validate.txt
│      │
│      ├─2
│      │      all.txt
│      │      test.txt
│      │      train.txt
│      │      validate.txt
│      │
│      ├─3
│      │      all.txt
│      │      test.txt
│      │      train.txt
│      │      validate.txt
│      │
│      └─4
│              all.txt
│              test.txt
│              train.txt
│              validate.txt
│
└─4code
    │  .gitignore
    │  readme.txt
    │  requirements_cpu.txt
    │  requirements_gpu.txt
    │  _1positiveSample.py
    │  _2negativeSample.py
    │  _3handleCluster.py
    │  _4train.py
    │  _5DIPPredict.py
    │  _6predict.py
    │
    ├─Rscript
    │      gene_enrichment.R
    │
    ├─tool
    │  │  calculate_performance.py
    │  │  common.py
    │  │  dao.py
    │  │  DatabaseOperation2.py
    │  │  dataset.py
    │  │  entry.py
    │  │  FastaDealear.py
    │  │  FeatureDealer.py
    │  │  handleCluster.py
    │  │  myData.py
    │  │  myEvaluate.py
    │  │  myModel.py
    │  │  mySupport.py
    │  │  negativeData.py
    │  │  PairDealer.py
    │  │  ProteinDealer.py
    │  │  queryPfam.py
    │  │  README.md
    │  │  README.txt
    │  │  scatter.py
    │  │  stepdeal20201226.py
    │  │  targetDao.py
    │  │  venn.py
    │  │  VennPlot.py
    │  │  _10humanTrain_support.py
    │  │  _5statistic.py
    │  │  _7imgPlotBar.py
    │  │  _7imgPlotVenn.py
    │  │  _8DIPPredict_support.py
    │  └─
    └─



