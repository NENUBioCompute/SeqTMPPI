# SeqTMPPI 实验流程
## 1. 输入IntAct中提取的蛋白交互对
1intAct_pair_norepeat.txt

## 2. 从mongodb中查询蛋白，判断蛋白对是否合格,得到TMP-nonTMP 基本信息
TMP + nonTMP ['accession', 'name', 'length', 'noX', 'inlenRange', 'subcellularLocations', 'seq']
由于多了一个\t 导致最后面有空列
2intAct_pair_norepeat_info.txt

## 3. 去重
取所有合格的蛋白对，根据序列去重
3intAct_pair_norepeat_info_qualified.txt

save 73784 pair
save 26440 protein fasta

save 8744 tmp
save 17696 nontmp



info table 1

info table 2

0                                                Q9CQC7 tmp-ac
1                                                P62259 nontmp-ac
2                                           NDUB4_MOUSE tmp-name
3                                                   129 tmp-length
4                                                  True tmp-noX
5                                                  True tmp-inlenRange
6                      ['Mitochondrion inner membrane'] tmp-subcellularLocations
7     MSGSKYKPAPLATLPSTLDPAEYDVSPETRRAQVERLSIRARLKRE... tmp-seq
8                                           1433E_MOUSE nontmp-name
9                                                   255 nontmp-length
10                                                 True nontmp-noX
11                                                 True nontmp-inlenRange
12               ['Cytoplasm', 'Melanosome', 'Nucleus'] nontmp-subcellularLocations
13    MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLS... nontmp-seq
14                                                   [] subcellularLocations of tmp & nontmp
15    ['Mitochondrion inner membrane', 'Melanosome',... subcellularLocations of tmp | nontmp
16                     ['Mitochondrion inner membrane'] subcellularLocations of tmp - nontmp
17               ['Melanosome', 'Nucleus', 'Cytoplasm'] subcellularLocations of nontmp - tmp
18                                                 True subcellularLocations of tmp & nontmp == []
19                                                MOUSE tmp-species
20                                                MOUSE nontmp-species
21                       ['mmu:100042503', 'mmu:68194'] tmp-keggid
22                                        ['mmu:22627'] nontmp-keggid
23                                                   [] overlap of keggid
24                                           ['174137'] tmp-geneID
25                                                  NaN nontmp-geneID
26    ['hsa00040', 'hsa00053', 'hsa00140', 'hsa00830... tmp_pathway
27                 ['hsa00040', 'hsa00053', 'hsa00520'] nontmp_pathway
28                             ['hsa00040', 'hsa00053'] overlap of pathway


