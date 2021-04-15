from DatabaseOperation2 import DataOperation
from dao import queryProtein

def handleNode(node):
    if isinstance(node,list):
        for n in node:
            yield from handleNode(n)
    elif isinstance(node,dict):
        for key in node.keys():
            if key in ['subcellularLocation','location','#text']:
                yield from handleNode(node[key])
    elif isinstance(node,str):
        yield node
    else:pass

if __name__ == '__main__':
    pass
    pa = 'Q8NBJ4'
    do = DataOperation('uniprot', 'uniprot_sprot')
    qa = queryProtein(pa, do)
    comment = qa['comment']
    subcellularLocations = []
    # 'comment' list,null
    # 'subcellularLocation' list,dict
    # 'location' list,dict
    # '#text'
    for comm in comment:
        if comm == {}:continue
        result = handleNode(comm)
        if result !=None:
            for r in result:
                print(r)