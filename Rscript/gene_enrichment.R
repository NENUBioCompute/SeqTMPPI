# Title     : gene_enrichment.R
# Created by: julse@qq.com
# Created on:  2021/3/22 19:48
# des : https://blog.csdn.net/weixin_43569478/article/details/83744384
# 图中灰色的点代表基因，黄色的点代表富集到的pathways, 默认画top5富集到的pathwayss, pathways节点的大小对应富集到的基因个数。
# rm(list=ls())
# library(DOSE)

## "file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich.png"
## 'file/6bioAnalysis/KEGGenrich/14TmpGeneIDCount_enrich.tsv'

# fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
# fout = 'file/6bioAnalysis/KEGGenrich/14TmpGeneIDCount_enrich.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
# head(summary(kk))
# write.csv(kk,file=fout)
#
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich.png'
#
# png(file = dirout, bg = "transparent")
# barplot(kk, showCategory = 10)
# dev.off()

## "file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich.png"
## 'file/6bioAnalysis/KEGGenrich/14nonTmpGeneIDCount_enrich.tsv'

# fin = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# fout = 'file/6bioAnalysis/KEGGenrich/14nonTmpGeneIDCount_enrich.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
# head(summary(kk))
# write.csv(kk,file=fout)
#
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich.png'
#
# png(file = dirout, bg = "transparent")
# barplot(kk, showCategory = 10)
# dev.off()

## "file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_dot.png"

# fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
#
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich_dot.png'
#
# png(file = dirout, bg = "transparent")
# dotplot(kk, showCategory = 10,orderBy = "x")
# dev.off()


## "file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_dot.png"

# fin = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_dot.png'
#
# png(file = dirout, bg = "transparent")
# dotplot(kk, showCategory = 10,orderBy = "x")
# dev.off()


## "file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich_cne.png"


# fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
#
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich_cne.png'
#
# png(file = dirout, bg = "white")
# cnetplot(kk, showCategory = 5)
# dev.off()



## "file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_cne.png"

# fin = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# fout = 'file/6bioAnalysis/KEGGenrich/14nonTmpGeneIDCount_enrich.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
#
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_cne.png'
#
# png(file = dirout, bg = "white")
# cnetplot(kk, showCategory = 5)
# dev.off()


## "file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich_ema.png"


# fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# library(enrichplot)
# library(GOSemSim)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
# ekk = pairwise_termsim(kk)
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich_ema.png'
#
# png(file = dirout, bg = "white")
# emapplot_cluster(ekk, showCategory = 30)
# dev.off()

## "file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_ema.png"

# fin = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# de = read.csv(fin,sep='\t',header = F)[,1]
#
# library(clusterProfiler)
# kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
#                  qvalueCutoff=0.1)
#
# dirout = 'file/6bioAnalysis/KEGGenrich/img/14nonTmpGeneIDCount_enrich_ema.png'
# ekk = pairwise_termsim(kk)
# png(file = dirout, bg = "white")
# emapplot_cluster(ekk, showCategory = 30)
# dev.off()



## "file/6bioAnalysis/KEGGenrich/img/14TmpGeneIDCount_enrich.png"
## 'file/6bioAnalysis/KEGGenrich/14TmpGeneIDCount_enrich.tsv'

rm(list=ls())
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(GOSemSim)
library(DOSE)
myfunc.draw = function(fout_img,kk,ty){
  png(file = fout_img, bg = "white",width = 680, height = 480)
  if(ty=='ema'){
    # ekk = pairwise_termsim(kk)
    print(emapplot_cluster(kk))
  }
  else if (ty=='cne'){print(cnetplot(kk, showCategory = 10))}
  else if (ty=='dot'){print(dotplot(kk, showCategory = 10,orderBy = "pvalue"))}
  else if (ty=='dot2'){print(dotplot(kk, showCategory = 10))}
  else if (ty=='bar'){print(barplot(kk, showCategory = 10))}
  else if (ty == 'dot3'){
    print(dotplot(kk, split="ONTOLOGY"))+ facet_grid(ONTOLOGY~., scale="free")
  }


  dev.off()
}
myfunc.draw_batch= function(outdir,kk,type){
  for(ty in type){
    fn = paste0(ty,'.png')
    fout_img = file.path(outdir,fn)
    myfunc.draw(fout_img,kk,ty)
  }
}
myfunc.enrichKEGG = function (de,fout_data){
  kk <- enrichKEGG(de, organism="hsa", pvalueCutoff=0.05, pAdjustMethod="BH",
                   qvalueCutoff=0.1)
  kk = pairwise_termsim(kk)
  write.csv(kk,file=fout_data,row.names = F)
  return (kk)
}
myfunc.enrichGO = function (de,fout_data,ont){
  ego <- enrichGO(gene  = de,
    # universe      = names(geneList),
    OrgDb         = org.Hs.eg.db,
    ont           = ont,
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.01,
    qvalueCutoff  = 0.05,
    readable      = TRUE)
  d <- godata('org.Hs.eg.db', ont=ont)
  kk <- pairwise_termsim(ego, semData = d)
  write.csv(kk,file=fout_data)
  return (kk)
}

myfunc.enrich = function (fin,foutdir,col,content = list('KEGG','GO','GO_all'),type=list('ema','cne','dot','bar','dot2','dot3')){

  # fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
  de = read.csv(fin,sep='\t',header = F)[,col]
  for(con in content){
    f1outdir = file.path(foutdir,con,'data')
    f2outdir = file.path(foutdir,con,'img')
    dir.create(f1outdir, showWarnings = FALSE, recursive = TRUE)
    dir.create(f2outdir, showWarnings = FALSE, recursive = TRUE)

    if(con =='KEGG'){
      fout_data = file.path(f1outdir,'enrich_data.csv')
      kk = myfunc.enrichKEGG(de,fout_data)
      myfunc.draw_batch(f2outdir,kk,type)
    }
    else if (con =='GO'){
      for(ont in list('BP','CC','MF')){
        fn = paste0('enrich_data_',ont,'.csv')
        fout_data = file.path(f1outdir,fn)
        kk = myfunc.enrichGO(de,fout_data,ont)
        for(ty in type){
          outdir = file.path(f2outdir,ont)
          dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
          myfunc.draw_batch(outdir,kk,ty)
        }

      }
    }
    else if (con == 'GO_all'){
      fn = paste0('enrich_data_all','.csv')
      fout_data = file.path(f1outdir,fn)
      kk <- enrichGO(de, OrgDb = "org.Hs.eg.db", ont="all")
      # dotplot(kk, split="ONTOLOGY") + facet_grid(ONTOLOGY~., scale="free")
      # dotplot(kk, split="ONTOLOGY",orderBy = "pvalue") + facet_grid(ONTOLOGY~., scale="free")
      myfunc.draw_batch(foutdir,kk,type)
      write.csv(kk,file=fout_data,row.names = F)
    }
  }
}


fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
foutdir = 'file/6bioAnalysis/1enrich/goall/TMP'
# fin = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# foutdir = 'file/6bioAnalysis/1enrich/nonTMP'
col = 1
# myfunc.enrich(fin,foutdir,col,content = list('GO_all'),type=list('dot3'))
# myfunc.enrich(fin,foutdir,col,content = list('KEGG','GO'),type=list('ema','cne','dot','bar'))

myfunc.enrich_compare = function (fin1,fin2,foutdir,col,content = list('KEGG','GO','Pathway','GO_all'),type=list('dot2')){

  # fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
  de1 = read.csv(fin1,sep='\t',header = F)[,col]
  de2 = read.csv(fin2,sep='\t',header = F)[,col]
  de = list(de1,de2)
  names(de)[1] = 'tmp'
  names(de)[2] = 'nontmp'
  for(con in content){
    f1outdir = file.path(foutdir,con,'data')
    f2outdir = file.path(foutdir,con,'img')
    dir.create(f1outdir, showWarnings = FALSE, recursive = TRUE)
    dir.create(f2outdir, showWarnings = FALSE, recursive = TRUE)

    if(con =='KEGG'){
      fout_data = file.path(f1outdir,'enrich_data.csv')
      kk=compareCluster(de, fun='enrichKEGG')
      write.csv(kk,file=fout_data,row.names = F)
      myfunc.draw_batch(f2outdir,kk,type)
    }
    else if (con == 'Pathway'){

    }
    else if (con =='GO'){
      for(ont in list('BP','CC','MF')){
        fn = paste0('enrich_data_',ont,'.csv')
        fout_data = file.path(f1outdir,fn)
        kk=compareCluster(de, fun='enrichGO',ont = ont,OrgDb = 'org.Hs.eg.db')
        write.csv(kk,file=fout_data,row.names = F)
        for(ty in type){
          outdir = file.path(f2outdir,ont)
          dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
          myfunc.draw_batch(outdir,kk,type)
        }

      }
    }

  }
}



# fin_tmp = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
# fin_nontmp = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# dirout = 'file/6bioAnalysis/1enrich/tmp_nontmp_concat'
# col = 1
# myfunc.enrich_compare(fin_tmp,fin_nontmp,dirout,col,content = list('GO_all'))
#



fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
fout_img = 'file/6bioAnalysis/1enrich/goall/TMP/GO_all/img/dot.png'
de = read.csv(fin,sep='\t',header = F)[,1]
go <- enrichGO(de, OrgDb = "org.Hs.eg.db", ont="all")
png(file = fout_img, bg = "white",width = 800, height = 648)
dotplot(go, split="ONTOLOGY",orderBy='GeneRatio') + facet_grid(ONTOLOGY~., scale="free")
dev.off()

# fin = 'file/5statistic/positive/14nonTmpGeneIDCount.tsv'
# fout_img = 'file/6bioAnalysis/1enrich/goall/nonTMP/GO_all/img/dot.png'
# dir.create('file/6bioAnalysis/1enrich/goall/nonTMP/GO_all/img', showWarnings = FALSE, recursive = TRUE)
#
# de = read.csv(fin,sep='\t',header = F)[,1]
# go <- enrichGO(de, OrgDb = "org.Hs.eg.db", ont="all")
# png(file = fout_img, bg = "white",width = 1317, height = 648)
# dotplot(go, split="ONTOLOGY",orderBy='GeneRatio') + facet_grid(ONTOLOGY~., scale="free")
# dev.off()