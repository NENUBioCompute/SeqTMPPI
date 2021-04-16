# A predictor for Transmembrane-nonTransmembrane protein interaction

[click here to use this tool](https://github.com/NENUBioCompute/SeqTMPPI/tree/main/4code)

Sequence-Based Protein Interaction Recognition Between Transmembrane and non-Transmembrane Protein

<p align="center"><img width="100%" src="https://github.com/NENUBioCompute/SeqTMPPI/blob/main/1images/Structure_of_the_architecture.png" /></p>

## Download data

We provide the test dataset used in this study,  you can download test.tsv, test.fasta to evaluate our method.

## Quick Start

### Requirements
- Python ≥ 3.6
- Tensorflow and Keras

### Testing & Evaluation in Command Line
We provide run.py that is able to run pre-trained models. Run it with:

~~~
python run.py --model ../2model/0/_my_model.h5 --fasta sample/pair.fasta --pair sample/pair.tsv --output_path result/
~~~


* To set the path of model file, use `--model` or `-m`.
* To set the path of fasta file, use `--fasta` or `-f`.
* To set the path of pair file, use `--pair` or `-p`.
* To save outputs to a directory, use `--output_path` or `-o`.

# Performance of the 5 models
|Subset|Acc|Precision|Recall|F1score|MCC|
|:----|:----|:----|:----|:----|:----|
|0|0.790|0.770|0.813|0.789|0.581|
|1|0.788|0.828|0.721|0.769|0.578|
|2|0.758|0.705|0.894|0.786|0.537|
|3|0.752|0.733|0.781|0.755|0.503|
|4|0.750|0.729|0.811|0.766|0.503|
|mean|0.768|0.753|0.804|0.773|0.541|


# content tree

~~~
root
│  readme.md
│  tree.txt
│  
├─.idea
│  │  deployment.xml
│  │  misc.xml
│  │  modules.xml
│  │  remote-mappings.xml
│  │  SeqTMPPI20201226.iml
│  │  vcs.xml
│  │  workspace.xml
│  │  
│  └─inspectionProfiles
│          profiles_settings.xml
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
│  │  └─test
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
    │  readme.md
    │  requirements_cpu.txt
    │  requirements_gpu.txt
    │  run.py
    │  support.py
    │  
    ├─sample
    │      pair.fasta
    │      pair.tsv
    │      
    └─
       
            


~~~

