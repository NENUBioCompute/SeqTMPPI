# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/26 15:05
@desc:
和数据库具体字段高度相关
"""
import os
import pandas as pd
from DatabaseOperation2 import DataOperation
from ProteinDealer import Protein
from common import getPairs, saveList, readIDlist
import random



def checkTaN(keywordDic):
    '''

    :param keywordDic:
    :return:
        True: TMP
        False: nonTMP
    '''
    # check T and N
    if isinstance(keywordDic,list):
        for keyword in keywordDic:
            if keyword['@id'] == 'KW-0812':
                return True
    else:
        if keywordDic['@id'] == 'KW-0812':
            return True
    return False
def getTaN(pa,pb,do,checkTMP=True,keepOne=False):
    '''

    :param pa:
    :param pb:
    :param do:
    :return: TMP
    '''
    projection = {'_id':True,'sequence.@length':True,'sequence.#text':True,'keyword.@id':True,'comment.subcellularLocation.location':True}
    qa = queryProtein(pa, do,projection=projection,keepOne=keepOne)
    qb = queryProtein(pb, do,projection=projection,keepOne=keepOne)
    if qa==None or qb==None:return None
    # projection = {'_id': True, 'sequence.@length': True, 'sequence.#text': True, 'keyword.@id': True,
    #               'comment.subcellularLocation.location.#text': True}
    # {'accession': 'P34709'}
    # {'_id': 'TRA2_CAEEL', 'comment': [{}, {}, {}, {}, {}, {'subcellularLocation': {'location': {'#text': 'Membrane'}}}, {}, {}, {}, {}, {}, {}, {}], 'keyword': [{'@id': 'KW-0024'}, {'@id': 'KW-0025'}, {'@id': 'KW-0217'}, {'@id': 'KW-0221'}, {'@id': 'KW-0472'}, {'@id': 'KW-0675'}, {'@id': 'KW-1185'}, {'@id': 'KW-0726'}, {'@id': 'KW-0732'}, {'@id': 'KW-0744'}, {'@id': 'KW-0812'}, {'@id': 'KW-1133'}], 'sequence': {'@length': '1475', '#text': 'MKLKYNKLLVSVVIVTFVTFGLLLAECFGKSIDYQEKSIFPSFVSQGFFETRTNNEEYIIEKIAQTQENGVDMRSTLHFTQHGYLLNNISNLKIKFRQKTYTLNDVCFKPHITIFQQSSSSDQNEYPHYIQRLLLEMQRLSPCLIVTPLNCFYDIYRIHGEISNWNKNTDFLNRRLRNSYIEAIGENDERPYVKSNYGPSLIKSWADHMFDLPSKSFTNSTKDALFQKIKLWLLSIEPRQKTCAASIHSCDTPLDSEHYFNICTDMQSVDNFAEKKTKFKLEDVDEEFAMNLDCVDDQEQFIEWMQELEIRKMYSHVTEKPDYPNVVNQTCDKIFHDLNSTGIEFFDGSRSFSSTKSQFDTMQTEIVLLTPEMLLSAMQHSDFVNGFESIWTIEKAEELIHEFRLALKEETEKFKENRMSKMIRVTSRVLDNTVTTKLQSFSEKQTIHFVVNVHSLIVILFTIFVWSGAPLRSAFMFFVRDALTCLLFCFVCSTDGVIVLDTELIKYIIVLTLANLYFTTRSSFCTERLSRCIQREKRFPINSNFASLITVDTMTDSRQIQYFLSTVTKYQAAQDSYSNELFERFPKNWGCTSILIFPIVFVYWYFIDSNFDKICVSVLPSFCLAAGEELFAKNMFWKEREAMQAKQRLENEEQAESITGSSLEKLFAGNKPVSNTDKANIVKKSSIIRNQKPCLQDLSPGTYDVSNFMKYPHQASRIFREKIIGLYLRILKLRTLGVILCIPAILLIVISIGLLFIPVKRETLHTDSKQDDIFIEFEIFNFSTNWKIVNQNLKQFSEDIESIGTLYTISNWQKSFERFEQETNKNASAEWNILFKWINDEPINSAVTLFSEKSSGNQTIANPFKFRLRYGFDAKNETTVIEIVQKIDELLSKCSKNLSPKAVGVLYEHYHRIAVVWNLFAFNQLTTAGIFIILLSIITFIFAITPTIKATFLFSLLVVGTQIEVAALVHLFSLDHHQIYTNLALFAGFLAAWDPFCALLRYRRRILYKSETRRTPELASKRRVLLPIVATADIAQFFVLLITAFSILAIICSIVPELNIFFVPTVILIVIQIVAVFNSIIVSIATKQMFESEVRHYLHRDLRGSTTAVRVYNLVQKQRLASSLDEPQVELDEFSIKRSSPPCRYYAPPPKYSCKKRSRSSDEDEDSDPNQPGPSNRRSPKTGNKRVRGNGDNTELYIPNRYELIVSGKSVGGNTSAAWNGPGSTLEQNMNALEECFELGVDEYDFDEHDGDEGCELVQDMLDRERNLMNKRSTAQRRESRNIEKMKKSQENLDKEKSEEKISESKKNQDDSIESPNLPGTPANLPVDEPLPPVGRLYIVEHVLPEEYRRDPLTEPPSMEDCIRAHSDPNLPPHPRADQYPASFTRPMVEYCEDIYWTHRTGQLPPGLQVPRRPYDYYHITERTPPPEDLNWVPPAESPPIPIPQQAFDLLEERRRNHREQQDEAREGDLSDPEV'}}
    qa['accession'] = pa
    qb['accession'] = pb
    if checkTMP:
        pa_TMP_flag = False if 'keyword' not in qa.keys() else checkTaN(qa['keyword'])
        pb_TMP_flag = False if 'keyword' not in qb.keys() else checkTaN(qb['keyword'])
        if pa_TMP_flag and not pb_TMP_flag: return qa, qb
        if not pa_TMP_flag and pb_TMP_flag: return qb, qa
    else:return qa,qb
    return None

def queryProtein(AC,do,projection=None,keepOne=False):
    '''
    dao
    :param AC:
    :param do:
    :return:
    '''
    # one accession mapping several protein sequence
    protein = None
    docs = do.Query('accession',AC,projection=projection)
    count = 0
    for doc in docs:
        count = count +1
        if count >1:
        #    一个accession 查询到多个蛋白质
        # 保存这个列表
            print('%s is more than one entry'%AC)
            return None
        protein = doc
        if keepOne:return protein
    return protein

def ensomblePortein(pro):
    ens_pro = {}
    prd = Protein()
    ens_pro['accession'] = pro['accession']
    ens_pro['name'] = pro['_id']
    ens_pro['length'] = pro['sequence']['@length']
    ens_pro['noX'] = prd.checkUncomm(pro['sequence']['#text'])
    ens_pro['inlenRange'] = prd.checkLengthRange(int(pro['sequence']['@length']), min=50, max=2000)
    ens_pro['subcellularLocations'] = [] if 'comment' not in pro.keys() else getSubcelllist(pro['comment'])
    ens_pro['seq'] = pro['sequence']['#text']
    return ens_pro

############################# start  subcellularLocation ###########################

def handleSubcelluarLeaf(node,keys=['subcellularLocation','location','#text']):
    if isinstance(node,list):
        for n in node:
            yield from handleSubcelluarLeaf(n,keys=keys)
    elif isinstance(node,dict):
        for key in node.keys():
            if key in keys:
                yield from handleSubcelluarLeaf(node[key],keys=keys)
    elif isinstance(node,str):
        yield node
    else:pass

def getSubcelllist(comment,keys=['subcellularLocation','location','#text']):
    '''
    :param comment: pro['comment']
        [{}, {'subcellularLocation': {'location': {'@evidence': '2', '#text': 'Cell inner membrane'}}}, {}, {}, {}, {}, {}]
        [{}, {}, {'subcellularLocation': {'location': 'Periplasm'}}, {'subcellularLocation': [{'location': 'Secreted'}, {'location': 'Cell surface'}]}, {'subcellularLocation': {'location': {'@evidence': '14', '#text': 'Cell outer membrane'}}}, {}, {}, {}, {}, {}, {}]
        projection 'comment.subcellularLocation.location':True

        {'subcellularLocation': {'location': [{'@evidence': '1', '#text': 'Secreted'},
   {'@evidence': '1', '#text': 'Cell wall'}]}}

    :return:


    '''
    # subcellularLocations = []
    # for comm in comment:
    #     if comm == {}:continue
    #     result = handleSubcelluarLeaf(comm,keys=keys)
    #     if result !=None:
    #         for r in result:
    #             subcellularLocations.append(r)
    #     return list(set(subcellularLocations))


    subcellularLocations = []
    result = handleSubcelluarLeaf(comment,keys=keys)
    if result !=None:
        for r in result:
            subcellularLocations.append(r)
    return list(set(subcellularLocations))

############################# end  subcellularLocation ###########################
def proteinPfam(fin,fout,tophit=True,item='Pfam'):
    do = DataOperation('uniprot', 'uniprot_sprot')
    df = pd.read_csv(fin, header=None)
    df = pd.DataFrame(df)
    df2 = df.apply(lambda x: funcPfam(x,do,tophit=tophit,item=item), axis=1)
    df2.to_csv(fout, sep='\t',header=None, index=None)
def funcPfam(x,do,tophit=False,item='Pfam'):
    # ac = 'P03372'
    result = queryPfam(x[0], do, tophit=tophit,item=item)
    if result:
        print(result)
        if tophit:
            x[1] = result[0]
            x[2] = result[1]
        else:
            x[1] = result
    return x
def getKEGG(dbref,item):
    '''

    :param dbref: <dbReference type="PDB" id="2ITO">
    :param item: PDB
    :return:
    '''
    pfamInfo = []
    for dbr in dbref:
        if dbr['@type'] == item:
            pfamInfo.append((dbr['@id']))
    return pfamInfo
def getGO(dbref):
    pfamInfo = []
    for dbr in dbref:
        if dbr['@type'] == 'GO':
            for pr in dbr['property']:
                if pr['@type'] == 'term':
                    pfamInfo.append((dbr['@id'],pr['@value']))
    return pfamInfo
def getPfam(dbref):
    '''
    :param dbref:
    :return: list of (pfamId,pfamName)
    [('PF12743', 'ESR1_C'), ('PF00104', 'Hormone_recep'), ('PF02159', 'Oest_recep'), ('PF00105', 'zf-C4')]
    '''
    pfamInfo = []
    for dbr in dbref:
        if dbr['@type'] == 'Pfam':
            for pr in dbr['property']:
                if pr['@type'] == 'entry name':
                    pfamInfo.append((dbr['@id'],pr['@value']))
    return pfamInfo
def queryPfam(ac,do,tophit=False,item='Pfam'):
    dic = {'accession':ac}
    projection = {'dbReference':True,'_id':False}
    qa = do.QueryObj(dic,projection=projection)
    result = None
    for q in qa:
        dbReference = q['dbReference']
        # {'@type': 'Pfam', '@id': 'PF18424', 'property': [{'@type': 'entry name', '@value': 'a_DG1_N2'}
        #  {'@type': 'Pfam', '@id': 'PF12743', 'property': [{'@type': 'entry name', '@value': 'ESR1_C'}, {'@type': 'match status', '@value': '1'}]}
        if item == 'Pfam':result = getPfam(dbReference)
        if item == 'GO':result = getGO(dbReference)
        if item == 'KEGG':result = getKEGG(dbReference,'KEGG')
        if item == 'GeneID':result = getKEGG(dbReference,'GeneID')
        if item == 'PDB':result = getKEGG(dbReference,'PDB')
        return result if not tophit else result[0] if result else None
    return None
def getPairInfo(fin,fout,sep='\t'):
    getPairInfo_TMP_nonTMP(fin, fout, sep=sep, checkTMP=False)


def getSingleInfo(fin, fout,fin_type='single',col=[0,1]):
    if fin_type == 'pair':
        df = pd.read_table(fin, header=None)[col]
        dat = df.to_numpy().reshape(1,-1)
        proteins = set(dat[0])
    elif fin_type == 'single':
        proteins = readIDlist(fin)
    else:pass
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':True,'sequence.@length':True,'sequence.#text':True,'keyword.@id':True,'comment.subcellularLocation.location':True}
    prod = Protein()
    with open(fout, 'w') as fo:
        for AC in proteins:
            pro = queryProtein(AC,do,projection=projection)
            pro['accession'] = AC
            if not prod.checkProtein(pro['sequence']['#text'],50,2000,uncomm=True):continue
            proinfo = ensomblePortein(pro)
            for v in proinfo.values():
                fo.write(str(v))
                fo.write('\t')
            fo.write('\n')
            fo.flush()
def getFasta(fin_all_protein,fout):
    proteins = readIDlist(fin_all_protein)
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'sequence.#text':True}
    with open(fout, 'w') as fo:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        records = []


        for AC in proteins:
            pro = queryProtein(AC,do,projection=projection)
            if not pro:continue
            record = SeqRecord(Seq(pro['sequence']['#text']), id=AC, description='')
            records.append(record)

            #fo.write('>%s\n%s\n'%(AC,pro['sequence']['#text']))
            #fo.flush()

        SeqIO.write(records, fout, 'fasta')

def getPairInfo_TMP_nonTMP(fin,fout,sep='\t',checkTMP=True,keepOne=False):
    '''

    :param fin:
        Q7BCK4	B6JN06
        E7QG89	B2FN41
    :param fout:
        TMP + nonTMP ['accession', 'name', 'length', 'noX', 'inlenRange', 'subcellularLocations', 'seq']
        Q7BCK4	ICSA_SHIFL	1102	True	True	['Periplasm', 'Secreted', 'Cell surface', 'Cell outer membrane']	MNQIHKFFCNMTQCSQGGAGELPTVKEKTCKLSFSPFVVGASLLLGGPIAFATPLSGTQELHFSEDNYEKLLTPVDGLSPLGAGEDGMDAWYITSSNPSHASRTKLRINSDIMISAGHGGAGDNNDGNSCGGNGGDSITGSDLSIINQGMILGGSGGSGADHNGDGGEAVTGDNLFIINGEIISGGHGGDSYSDSDGGNGGDAVTGVNLPIINKGTISGGNGGNNYGEGDGGNGGDAITGSSLSVINKGTFAGGNGGAAYGYGYDGYGGNAITGDNLSVINNGAILGGNGGHWGDAINGSNMTIANSGYIISGKEDDGTQNVAGNAIHITGGNNSLILHEGSVITGDVQVNNSSILKIINNDYTGTTPTIEGDLCAGDCTTVSLSGNKFTVSGDVSFGENSSLNLAGISSLEASGNMSFGNNVKVEAIINNWAQKDYKLLSADKGITGFSVSNISIINPLLTTGAIDYTKSYISDQNKLIYGLSWNDTDGDSHGEFNLKENAELTVSTILADNLSHHNINSWDGKSLTKSGEGTLILAEKNTYSGFTNINAGILKMGTVEAMTRTAGVIVNKGATLNFSGMNQTVNTLLNSGTVLINNINAPFLPDPVIVTGNMTLEKNGHVILNNSSSNVGQTYVQKGNWHGKGGILSLGAVLGNDNSKTDRLEIAGHASGITYVAVTNEGGSGDKTLEGVQIISTDSSDKNAFIQKGRIVAGSYDYRLKQGTVSGLNTNKWYLTSQMDNQESKQMSNQESTQMSSRRASSQLVSSLNLGEGSIHTWRPEAGSYIANLIAMNTMFSPSLYDRHGSTIVDPTTGQLSETTMWIRTVGGHNEHNLADRQLKTTANRMVYQIGGDILKTNFTDHDGLHVGIMGAYGYQDSKTHNKYTSYSSRGTVSGYTAGLYSSWFQDEKERTGLYMDAWLQYSWFNNTVKGDGLTGEKYSSKGITGALEAGYIYPTIRWTAHNNIDNALYLNPQVQITRHGVKANDYIEHNGTMVTSSGGNNIQAKLGLRTSLISQSCIDKETLRKFEPFLEVNWKWSSKQYGVIMNGMSNHQIGNRNVIELKTGVGGRLADNLSIWGNVSQQLGNNSYRDTQGILGVKYTF	B6JN06	G6PI_HELP2	545	True	True	['Cytoplasm']	MLTQLKTYPKLLKHYEEIKEAHMRDWFSKDKERASRYFVQLESLSLDYSKNRLNDTTLKLLFELANDCSLKEKIEAMFKGEKINTTEKRAVLHTALRSLNDTEILLDNMEVLKSVRSVLKRMRAFSDSVRSGKRLGYTNQVITDIVNIGIGGSDLGALMVCTALKRYGHPRLKMHFVSNVDGTQILDVLEKINPASTLFIVASKTFSTQETLTNALTARKWFVERSGDEKHIAKHFVAVSTNKEAVQQFGIDEHNMFEFWDFVGGRYSLWSAIGLSIMIYLGKKNFNALLKGAYLMDEHFRNAPFESNLPVLMGLIGVWYINFFQSKSHLIAPYDQYLRHFPKFIQQLDMESNGKRISKKGETIPYDTCPVVWGDMGINAQHAFFQLLHQGTHLIPIDFIASLDKKPNAKGHHEILFSNVLAQAQAFMKGKSYEEALGELLFKGLDKDEAKDLAHHRVFFGNRPSNILLLEKISPSNIGALVALYEHKVFVQGVIWDINSFDQWGVELGKELAVPILQELEGHKSNAYFDSSTKHLIELYKNYNQ
        E7QG89	SEC11_YEASZ	167	True	True	['Endoplasmic reticulum membrane']	MNLRFELQKLLNVCFLFASAYMFWQGLAIATNSASPIVVVLSGSMEPAFQRGDILFLWNRNTFNQVGDVVVYEVEGKQIPIVHRVLRQHNNHADKQFLLTKGDNNAGNDISLYANKKIYLNKSKEIVGTVKGYFPQLGYITIWISENKYAKFALLGMLGLSALLGGE	B2FN41	EX7L_STRMK	443	True	True	['Cytoplasm']	MQPRNNDILTPSQLNTLARDLLEGSFPAIWVEAELGSVARPASGHLYFTLKDARAQLRAAMFRMKAQYLKFVPREGMRVLVRGKVTLYDARGEYQMVLDHMEEAGEGALRRAFEELKARLEAEGLFDPARKRPLPTHVQRLAVITSPTGAAVRDVLSVLGRRFPLLEVDLLPTLVQGSSAAAQITRLLQAADASGRYDVILLTRGGGSLEDLWAFNDEALARAIAASRTPVVSAVGHETDFSLSDFAADLRAPTPSVAAELLVPDQRELALRLRRTAARMVQLQRHAMQQAMQRADRALLRLNAQSPQARLDLLRRRQLDLGRRLHAVFNQQQERRAARLRHAAAVLRGHHPQRQLDAMQRRLAALRGRPQAAMQRLLERDALRLRGLARSLEAVSPLATVARGYSILTRTDDGTLVRKVNQVQPGDALQARVGDGVIDVQVK

    :return:
    fin = 'file/_1pair.txt'
    fout = 'file/_2pair_info.txt'
    getPairInfo_TMP_nonTMP(fin,fout)
    '''
    do = DataOperation('uniprot', 'uniprot_sprot')
    with open(fout,'w') as fo:
        for pa,pb in getPairs(fin, sep=sep, title=False):
            print('%s\t%s'%(pa,pb))
            result = getTaN(pa,pb,do,checkTMP=checkTMP,keepOne=keepOne)
            if result==None:continue
            tmp = ensomblePortein(result[0])
            nontmp = ensomblePortein(result[1])
            for v in tmp.values():
                fo.write(str(v))
                fo.write('\t')
            for v in nontmp.values():
                fo.write(str(v))
                fo.write('\t')
            fo.write('\n')
            fo.flush()

def tagPair(pa,pb,do):
    '''

    :param pa:
    :param pb:
    :param do:
    :return: TMP
    '''
    projection = {'_id':True,'sequence.@length':True,'sequence.#text':True,'keyword.@id':True,'comment.subcellularLocation.location':True}
    qa = queryProtein(pa, do,projection=projection)
    qb = queryProtein(pb, do,projection=projection)
    if qa==None or qb==None:return None
    # projection = {'_id': True, 'sequence.@length': True, 'sequence.#text': True, 'keyword.@id': True,
    #               'comment.subcellularLocation.location.#text': True}
    # {'accession': 'P34709'}
    # {'_id': 'TRA2_CAEEL', 'comment': [{}, {}, {}, {}, {}, {'subcellularLocation': {'location': {'#text': 'Membrane'}}}, {}, {}, {}, {}, {}, {}, {}], 'keyword': [{'@id': 'KW-0024'}, {'@id': 'KW-0025'}, {'@id': 'KW-0217'}, {'@id': 'KW-0221'}, {'@id': 'KW-0472'}, {'@id': 'KW-0675'}, {'@id': 'KW-1185'}, {'@id': 'KW-0726'}, {'@id': 'KW-0732'}, {'@id': 'KW-0744'}, {'@id': 'KW-0812'}, {'@id': 'KW-1133'}], 'sequence': {'@length': '1475', '#text': 'MKLKYNKLLVSVVIVTFVTFGLLLAECFGKSIDYQEKSIFPSFVSQGFFETRTNNEEYIIEKIAQTQENGVDMRSTLHFTQHGYLLNNISNLKIKFRQKTYTLNDVCFKPHITIFQQSSSSDQNEYPHYIQRLLLEMQRLSPCLIVTPLNCFYDIYRIHGEISNWNKNTDFLNRRLRNSYIEAIGENDERPYVKSNYGPSLIKSWADHMFDLPSKSFTNSTKDALFQKIKLWLLSIEPRQKTCAASIHSCDTPLDSEHYFNICTDMQSVDNFAEKKTKFKLEDVDEEFAMNLDCVDDQEQFIEWMQELEIRKMYSHVTEKPDYPNVVNQTCDKIFHDLNSTGIEFFDGSRSFSSTKSQFDTMQTEIVLLTPEMLLSAMQHSDFVNGFESIWTIEKAEELIHEFRLALKEETEKFKENRMSKMIRVTSRVLDNTVTTKLQSFSEKQTIHFVVNVHSLIVILFTIFVWSGAPLRSAFMFFVRDALTCLLFCFVCSTDGVIVLDTELIKYIIVLTLANLYFTTRSSFCTERLSRCIQREKRFPINSNFASLITVDTMTDSRQIQYFLSTVTKYQAAQDSYSNELFERFPKNWGCTSILIFPIVFVYWYFIDSNFDKICVSVLPSFCLAAGEELFAKNMFWKEREAMQAKQRLENEEQAESITGSSLEKLFAGNKPVSNTDKANIVKKSSIIRNQKPCLQDLSPGTYDVSNFMKYPHQASRIFREKIIGLYLRILKLRTLGVILCIPAILLIVISIGLLFIPVKRETLHTDSKQDDIFIEFEIFNFSTNWKIVNQNLKQFSEDIESIGTLYTISNWQKSFERFEQETNKNASAEWNILFKWINDEPINSAVTLFSEKSSGNQTIANPFKFRLRYGFDAKNETTVIEIVQKIDELLSKCSKNLSPKAVGVLYEHYHRIAVVWNLFAFNQLTTAGIFIILLSIITFIFAITPTIKATFLFSLLVVGTQIEVAALVHLFSLDHHQIYTNLALFAGFLAAWDPFCALLRYRRRILYKSETRRTPELASKRRVLLPIVATADIAQFFVLLITAFSILAIICSIVPELNIFFVPTVILIVIQIVAVFNSIIVSIATKQMFESEVRHYLHRDLRGSTTAVRVYNLVQKQRLASSLDEPQVELDEFSIKRSSPPCRYYAPPPKYSCKKRSRSSDEDEDSDPNQPGPSNRRSPKTGNKRVRGNGDNTELYIPNRYELIVSGKSVGGNTSAAWNGPGSTLEQNMNALEECFELGVDEYDFDEHDGDEGCELVQDMLDRERNLMNKRSTAQRRESRNIEKMKKSQENLDKEKSEEKISESKKNQDDSIESPNLPGTPANLPVDEPLPPVGRLYIVEHVLPEEYRRDPLTEPPSMEDCIRAHSDPNLPPHPRADQYPASFTRPMVEYCEDIYWTHRTGQLPPGLQVPRRPYDYYHITERTPPPEDLNWVPPAESPPIPIPQQAFDLLEERRRNHREQQDEAREGDLSDPEV'}}
    qa['accession'] = pa
    qb['accession'] = pb

    pa_TMP_flag = False if 'keyword' not in qa.keys() else checkTaN(qa['keyword'])
    pb_TMP_flag = False if 'keyword' not in qb.keys() else checkTaN(qb['keyword'])
    # TMP_SP
    if pa_TMP_flag and not pb_TMP_flag: return qa, qb, 0
    if not pa_TMP_flag and pb_TMP_flag: return qb, qa, 0
    # TMP_TMP
    if pa_TMP_flag and pb_TMP_flag: return qa, qb, 1
    # SP_SP
    if not pa_TMP_flag and not pb_TMP_flag: return qa, qb, 2
    return None
def getPairTag(fin,f1pairWithTag_info,sep='\t'):
    '''

    :param fin: fin pair
    :param f1pairWithTag_info: fout
    :return:
    tag
    0, TMP_SP
    1, TMP_TMP
    2, SP_SP
    '''
    do = DataOperation('uniprot', 'uniprot_sprot')
    with open(f1pairWithTag_info, 'w') as fo:
        for pa, pb in getPairs(fin, sep=sep, title=False):
            print('%s\t%s' % (pa, pb))
            result = tagPair(pa, pb, do)
            if result==None:continue
            proA = ensomblePortein(result[0])
            proB = ensomblePortein(result[1])
            for v in proA.values():
                fo.write(str(v))
                fo.write('\t')
            for v in proB.values():
                fo.write(str(v))
                fo.write('\t')
            fo.write(str(result[2]))
            fo.write('\n')
            fo.flush()
def save(aclist,dbnam,fnotSave=None):
    # aclist = ['P06685', 'P06686']
    # dbnam = 'seqtmppi_test1'
    do_all = DataOperation('uniprot', 'seqtmppi_positive')
    do_new = DataOperation('uniprot', dbnam)
    notsaveCount = 0
    notsaveList = []
    for ac in aclist:
        protein = queryProtein(ac,do_all)
        if protein == None:
            print('not save protein %s'%ac)
            continue
        # projection = {'_id': True, 'sequence.@length': True, 'sequence.#text': True, 'keyword.@id': True,
        #               'comment.subcellularLocation.location': True}
        # protein_ens = ensomblePortein(queryProtein(ac, do_all,projection=projection))
        # protein_ens['accession'] = ac
        # protein['ensomble'] = protein_ens
        result = do_new.UpSertOne({'accession': ac}, protein)
        if result.matched_count!=0:
            print('not save %s'%ac)
            notsaveCount = notsaveCount + 1
            notsaveList.append(ac)
    print('not save %d'%(notsaveCount))
    if fnotSave:
        saveList(notsaveList,fnotSave)
    return notsaveList

def generateCriterLists(ftmp,fnontmp):
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':False,'accession':True}
    docs = do.QueryObj({"keyword.@id":'KW-0812','comment.subcellularLocation.location.#text': {'$exists':True}},projection=projection)
    tmplist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    docs = do.QueryObj({"keyword.@id":{'$ne':'KW-0812'},'keyword':{'$exists':True},'comment.subcellularLocation.location.#text': {'$exists':True}},projection=projection)
    nontmplist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    print('query %d tmp and %d nontmp'%(len(tmplist),len(nontmplist)))
    saveList(tmplist,ftmp)
    saveList(nontmplist,fnontmp)
    return tmplist,nontmplist

def generateHumanLists(ftmp,fnontmp):
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':False,'accession':True}
    docs = do.QueryObj({"keyword.@id":'KW-0812','_id':{'$regex':'HUMAN'}},projection=projection)
    tmplist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    docs = do.QueryObj({"keyword.@id":{'$ne':'KW-0812'},'keyword':{'$exists':True},'_id':{'$regex':'HUMAN'}},projection=projection)
    nontmplist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    print('query %d tmp and %d nontmp'%(len(tmplist),len(nontmplist)))
    saveList(tmplist,ftmp)
    saveList(nontmplist,fnontmp)
    return tmplist,nontmplist
def generateSomeSpeciesLists(ftmp,fnontmp,species='MYCPN'):
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id':False,'accession':True}
    docs = do.QueryObj({"keyword.@id":'KW-0812','_id':{'$regex':species}},projection=projection)
    tmplist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    docs = do.QueryObj({"keyword.@id":{'$ne':'KW-0812'},'keyword':{'$exists':True},'_id':{'$regex':'HUMAN'}},projection=projection)
    nontmplist = [x['accession'][0] if isinstance(x['accession'],list) else x['accession'] for x in docs]
    print('query %d tmp and %d nontmp'%(len(tmplist),len(nontmplist)))
    saveList(tmplist,ftmp)
    saveList(nontmplist,fnontmp)
    return tmplist,nontmplist

def composeTMP_nonTMP(ftmp,fnontmp,fpair,num):
    '''
    如果给了ftmp,fnontmp，就参考所给的列表组合
    如果这个文件不存在，则从数据库中查询这两个列表
    :param ftmp:
    :param fnontmp:
    :param fpair: 组合结果
    :param num: 组合的总数
    :return:
    '''
    if not (os.access(ftmp,os.F_OK) and os.access(fnontmp,os.F_OK)):
        tmplist,nontmplist = generateCriterLists(ftmp,fnontmp)
    else:
        tmplist, nontmplist = readIDlist(ftmp),readIDlist(fnontmp)
    sampL1 = min(len(tmplist),num)
    sampL2 = min(len(nontmplist),num)
    L1 = random.sample(range(0, len(tmplist)), sampL1)
    L2 = random.sample(range(0, len(nontmplist)), sampL2)
    with open(fpair,'w') as fo:
        for idx in range(num):
            fo.write('%s\t%s\n'%(tmplist[L1[(idx+random.randint(2,9))%sampL1]],nontmplist[L2[(idx+random.randint(1,5))%sampL2]]))
            fo.flush()
            print(idx)

def fullyComposeTMP_nonTMP(ftmp,fnontmp,fpair):
    '''
    如果给了ftmp,fnontmp，就参考所给的列表组合
    如果这个文件不存在，则从数据库中查询这两个列表
    :param ftmp:
    :param fnontmp:
    :param fpair: 组合结果
    :param num: 组合的总数
    :return:
    '''
    if not (os.access(ftmp,os.F_OK) and os.access(fnontmp,os.F_OK)):
        print('Cretira not found, qury from Mongodb...')
        tmplist,nontmplist = generateCriterLists(ftmp,fnontmp)
    else:
        tmplist, nontmplist = readIDlist(ftmp),readIDlist(fnontmp)
    with open(fpair, 'w') as fo:
        for elem1 in tmplist:
            for elem2 in nontmplist:
                print(elem1,elem2)
                fo.write('%s\t%s\n'%(elem1,elem2))
                fo.flush()
############################## partner ##################################
def findKeyProtein(fin,fout,keyword):
    # keyword = 'KW-0297'
    tmplist = readIDlist(fin)
    GPCRlist = []
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id': True}
    for ac in tmplist:
        dic = {'accession':ac,"keyword.@id":keyword}
        result = do.QueryObj(dic,projection=projection)
        for r in result:
            GPCRlist.append(ac)
            print(r)
    saveList(GPCRlist,fout)

def findGProtein(fin,fout):
    # {"protein.recommendedName.fullName":{$regex:/Guanine nucleotide-binding protein*/}}
    # keyword = 'KW-0297'
    tmplist = readIDlist(fin)
    GPCRlist = []
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id': True}
    count = 0
    for ac in tmplist:
        dic = {
            'accession': ac,
            '$or': [
                {"protein.recommendedName.fullName": {'$regex': 'G protein '}},
                {"protein.recommendedName.fullName": {'$regex': 'Guanine nucleotide-binding protein'}},
                {"protein.alternativeName.fullName": {'$regex': 'G protein '}},
                {"protein.alternativeName.fullName": {'$regex': 'Guanine nucleotide-binding protein'}}
            ]
        }
        result = do.QueryObj(dic,projection=projection)
        for r in result:
            count = count +1
            GPCRlist.append(ac)
            print(count,r)
    saveList(GPCRlist,fout)
def getALlGprotein(fout):
    # {$or:[
    # 	{"protein.recommendedName.fullName":{$regex:/G protein +/}},
    # 	{"protein.recommendedName.fullName":{$regex:/G protein-coupled receptor +/}},
    # 	{"protein.alternativeName.fullName":{$regex:/G protein +/}},
    # 	{"protein.alternativeName.fullName":{$regex:/G protein-coupled receptor +/}}
    # ]}

    GPCRlist = []
    do = DataOperation('uniprot', 'uniprot_sprot')
    projection = {'_id': True,'protein.recommendedName.fullName':True,'protein.alternativeName.fullName':True}
    count = 0
    dic = {
        # 'accession': ac,
               '$or': [
                   {"protein.recommendedName.fullName": {'$regex': 'G protein '}},
                   {"protein.recommendedName.fullName": {'$regex': 'Guanine nucleotide-binding protein'}},
                   {"protein.alternativeName.fullName": {'$regex': 'G protein '}},
                   {"protein.alternativeName.fullName": {'$regex': 'Guanine nucleotide-binding protein'}}
               ]
           }
    result = do.QueryObj(dic,projection=projection)
    for r in result:
        count = count +1
        GPCRlist.append(((r['_id'],[x for x in handleSubcelluarLeaf(r['protein'],keys=['alternativeName','recommendedName','fullName'])])))
        print(((r['_id'],[x for x in handleSubcelluarLeaf(r['protein'],keys=['alternativeName','recommendedName','fullName'])])))
    saveList(GPCRlist,fout)
if __name__ == '__main__':
    pass
    # aclist = ['P06685', 'P06686']
    # save(aclist)