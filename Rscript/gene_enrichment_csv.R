# Title     : gene_enrichment_csv.R
# Created by: julse@qq.com
# Created on:  2021/3/23 15:34
# des : plot img from local file
#  not complete! data read from file may different from data calculated
rm(list=ls())
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(GOSemSim)
myfunc.draw = function(fout_img,kk,ty){
  png(file = fout_img, bg = "white")
  if(ty=='ema'){
    # ekk = pairwise_termsim(kk)
    emapplot_cluster(kk)
  }
  else if (ty=='cne'){cnetplot(kk, showCategory = 10)}
  else if (ty=='dot'){dotplot(kk, showCategory = 10,orderBy = "pvalue")}
  else if (ty=='bar'){barplot(kk, showCategory = 10)}
  dev.off()
}
myfunc.enrich = function (fin,foutdir,col,content = list('KEGG','GO'),type=list('ema','cne','dot','bar')){

  # fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
  for(con in content){
    f1outdir = file.path(foutdir,con,'data')
    f2outdir = file.path(foutdir,con,'img')

    if(con =='KEGG'){
      fout_data = file.path(f1outdir,'enrich_data.csv')
      kk = read.csv(fout_data)
    }
    else if (con =='GO'){
      for(ont in list('BP','CC','MF')){
        fn = paste('enrich_data_',ont,'.csv')
        fout_data = file.path(f1outdir,fn)
        kk = read.csv(fout_data)
      }
    }

    for(ty in type){
      fn = paste0(ty,'.png')
      fout_img = file.path(f2outdir,fn)
      myfunc.draw(fout_img,kk,ty)
    }
  }
}


fin = 'file/5statistic/positive/14TmpGeneIDCount.tsv'
foutdir = 'file/6bioAnalysis/1enrich'
col = 1
myfunc.enrich(fin,foutdir,col,content = list('KEGG','GO'),type=list('ema','cne','dot','bar'))