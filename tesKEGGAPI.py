# Title     : tesKEGGAPI.py
# Created by: julse@qq.com
# Created on: 2021/2/4 15:23
# des : TODO
import os

from common import readIDlist

if __name__ == '__main__':
    from Bio.KEGG import REST
    dirout = 'file/6bioAnalysis/keggDB/pathwayInfo'
    human_pathways = REST.kegg_list("pathway", "hsa").read()
    repair_pathways = readIDlist('file/6bioAnalysis/keggDB/1pathway_human.tsv')

    # Get the genes for pathways and add them to a list
    repair_genes = []
    for idx,pathway in enumerate(repair_pathways):
        print(idx,pathway)
        pathway_file = REST.kegg_get(pathway).read()  # query and read each pathway
        with open(os.path.join(dirout,'%s.txt'%pathway),'w') as fo:
            fo.write(pathway_file)
            fo.flush()