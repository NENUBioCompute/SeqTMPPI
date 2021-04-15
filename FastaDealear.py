# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/12/26 21:07
@desc:
"""
import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ProteinDealer import Protein
from common import readIDlist, check_path, getPairs
import numpy as np

class FastaDealer:
    def extractFasta(self, fin_fasta, fin_idlist,fout_fasta, in_multi=True,out_multi=True):
        oridict = self.getDict(fin_fasta, multi=in_multi)
        desdict = {}
        idlist = readIDlist(fin_idlist)
        for id in idlist:
            desdict[id] = oridict[id]
        self.dict2fasta(desdict,fout_fasta,multi=out_multi)

    def getDict(self,fin_fasta, fout='', multi=True):
        """

        :param fin:
        :param fout:if != '',save dict in the file
        :param multi:
            False:sequence occupied one line
            True:... multiple lines
        :return:{ID:seq}
        """
        mydict = {}
        for ID,seq in self.getYield(fin_fasta,multi=multi):
            mydict[ID] = seq
        if fout:
            with open(fout, 'w')as fo:
                fo.write(mydict)
                fo.flush()
        return mydict

    def getNpy(self,fin_fasta,out_dir,multi=True,checkprotein = True):
        check_path(out_dir)
        p = Protein()
        for ID, seq in self.getYield(fin_fasta,multi=multi):
            if not checkprotein or p.checkProtein(seq, 50, 2000, uncomm=True):
                filename = os.path.join(out_dir,"%s.npy"%ID)
                result =self.seq2num(seq)
                if len(result)!=0:
                    np.save(filename,result)

    def dict2fasta(self,mydict,fout_fasta,multi=True):
        if multi:
            records = []
            for ID in mydict.keys():
                record = SeqRecord(Seq(mydict[ID]), id=ID, description='')
                records.append(record)
            SeqIO.write(records, fout_fasta, 'fasta')
        else:
            with open(fout_fasta,'w') as fo:
                for ID in mydict.keys():
                    fo.write('>%s\n%s\n'%(ID,mydict[ID]))

    def getPairseq(self,fin_fasta,fin_ID_pair,fout_seq_pair,saveID=False,num=0,multi=True):
        # fd = FastaDealer()
        # fin_fasta = '/home/jjhnenu/data/PPI/release/featuredb/positiveV1.fasta'
        # fin_ID_pair = '/home/jjhnenu/data/PPI/release/pairdata/positive_2049.txt'
        # fout_seq_pair = '/home/jjhnenu/data/PPI/release/pairdata/positive_2049_seq.txt'
        # mydict = fd.getDict(fin_fasta, multi=True)
        mydict= self.getDict(fin_fasta, multi=multi)
        file_num = 0
        count = 0
        fout_seq_pair = fout_seq_pair
        myfout_seq_pair = fout_seq_pair.split('.')[0] + '_%d.txt' % file_num
        myfout_seq_ID_pair = fout_seq_pair.split('.')[0] + '_ID_%d.txt' % file_num
        fo = open(myfout_seq_pair, 'w')
        fo_ID = open(myfout_seq_ID_pair,'w')
        for record in getPairs(fin_ID_pair, sep='\t', title=False):
            a = record[0]
            b = record[1]
            c = ''
            if len(record) == 3:
                c = record[2]
            if saveID:
                # fo.write('>%s %s\n' % (a, c))
                fo.write('>%s\n' % a)
                fo.flush()
            fo.write(mydict[a] + '\n')
            fo.flush()
            if saveID:
                # fo.write('>%s %s\n' % (b, c))
                fo.write('>%s\n' % b)
            fo_ID.write('%s\t%s\t%s\n'%(a,b,c))
            fo.write(mydict[b] + '\n')
            fo.flush()
            count = count + 1
            if num != 0 and count == num:
                file_num = file_num + 1
                count = 0
                fo.close()
                fo_ID.close()
                myfout_seq_pair = fout_seq_pair.split('.')[0] + '_%d.txt' % file_num
                myfout_seq_ID_pair = fout_seq_pair.split('.')[0] + '_ID_%d.txt' % file_num
                fo = open(myfout_seq_pair, 'w')
                fo_ID = open(myfout_seq_ID_pair, 'w')
        fo.close()
        fo_ID.close()
        print()
    # support
    def getYield(self,fin_fasta,multi=True):
        """
        :param fin_fasta:
        :param multi:
            True
                >db|ID
                seqpart
                seqpart

                >db|ID xxx
                seqpart
                seqpart

                >ID
                seqpart
                seqpart


            False
                >db|ID
                seq

                >db|ID xxx
                seq

                >ID
                seq

        :return:
        yied like
        A6NIM6 MSVTGFTITDEKVHLYHSIEKEKTVRHIGDLCSSHSVKKIQVGICLLLVELCERFTFFEVVCNMIPFCTIKLGYHNCQAAILNLCFIGTSILTPVFVRWLTDVYLGRNKLVYICLFLHFLGTALLSVVAFPLEDFYLGTYHAVNNIPKTEQHRLFYVALLTICLGIGGVRAIVCPLGAFGLQEYGSQKTMSFFNWFYWLMNLNATIVFLGISYIQHSQAWALVLLIPFMSMLMAVITLHMIYYNLIYQSEKRCSLLTGVGVLVSALKTCHPQYCHLGRDVTSQLDHAKEKNGGCYSELHVEDTTFFLTLLPLFIFQLLYRMCIMQIPSGYYLQTMNSNLNLDGFLLPIAVMNAISSLPLLILAPFLEYFSTCLFPSKRVGSFLSTCIIAGNLFAALSVMIAGFFEIHRKHFPAVEQPLSGKVLTVSSMPCFYLILQYVLLGVAETLVNPALSVISYRFVPSNVRGTSMNFLTLFNGFGCFTGALLVKLVYLISDGNWFPNTLNKGNLESFFFFLASLTLLNVLGFCSVSQRYCNLNHFNAQNIRGSNLEETLLLHEKSLKFYGSIQEFSSSIDLWETAL
        Q14160 MLKCIPLWRCNRHVESVDKRHCSLQAVPEEIYRYSRSLEELLLDANQLRELPKPFFRLLNLRKLGLSDNEIQRLPPEVANFMQLVELDVSRNDIPEIPESIKFCKALEIADFSGNPLSRLPDGFTQLRSLAHLALNDVSLQALPGDVGNLANLVTLELRENLLKSLPASLSFLVKLEQLDLGGNDLEVLPDTLGALPNLRELWLDRNQLSALPPELGNLRRLVCLDVSENRLEELPAELGGLVLLTDLLLSQNLLRRLPDGIGQLKQLSILKVDQNRLCEVTEAIGDCENLSELILTENLLMALPRSLGKLTKLTNLNVDRNHLEALPPEIGGCVALSVLSLRDNRLAVLPPELAHTTELHVLDVAGNRLQSLPFALTHLNLKALWLAENQAQPMLRFQTEDDARTGEKVLTCYLLPQQPPPSLEDAGQQGSLSETWSDAPPSRVSVIQFLEAPIGDEDAEEAAAEKRGLQRRATPHPSELKVMKRSIEGRRSEACPCQPDSGSPLPAEEEKRLSAESGLSEDSRPSASTVSEAEPEGPSAEAQGGSQQEATTAGGEEDAEEDYQEPTVHFAEDALLPGDDREIEEGQPEAPWTLPGGRQRLIRKDTPHYKKHFKISKLPQPEAVVALLQGMQPDGEGPVAPGGWHNGPHAPWAPRAQKEEEEEEEGSPQEEEVEEEEENRAEEEEASTEEEDKEGAVVSAPSVKGVSFDQANNLLIEPARIEEEELTLTILRQTGGLGISIAGGKGSTPYKGDDEGIFISRVSEEGPAARAGVRVGDKLLEVNGVALQGAEHHEAVEALRGAGTAVQMRVWRERMVEPENAVTITPLRPEDDYSPRERRGGGLRLPLLPPESPGPLRQRHVACLARSERGLGFSIAGGKGSTPYRAGDAGIFVSRIAEGGAAHRAGTLQVGDRVLSINGVDVTEARHDHAVSLLTAASPTIALLLEREAGGPLPPSPLPHSSPPTAAVATTSITTATPGVPGLPSLAPSLLAAALEGPYPVEEIRLPRAGGPLGLSIVGGSDHSSHPFGVQEPGVFISKVLPRGLAARSGLRVGDRILAVNGQDVRDATHQEAVSALLRPCLELSLLVRRDPAPPGLRELCIQKAPGERLGISIRGGARGHAGNPRDPTDEGIFISKVSPTGAAGRDGRLRVGLRLLEVNQQSLLGLTHGEAVQLLRSVGDTLTVLVCDGFEASTDAALEVSPGVIANPFAAGIGHRNSLESISSIDRELSPEGPGKEKELPGQTLHWGPEATEAAGRGLQPLKLDYRALAAVPSAGSVQRVPSGAAGGKMAESPCSPSGQQPPSPPSPDELPANVKQAYRAFAAVPTSHPPEDAPAQPPTPGPAASPEQLSFRERQKYFELEVRVPQAEGPPKRVSLVGADDLRKMQEEEARKLQQKRAQMLREAAEAGAEARLALDGETLGEEEQEDEQPPWASPSPTSRQSPASPPPLGGGAPVRTAKAERRHQERLRVQSPEPPAPERALSPAELRALEAEKRALWRAARMKSLEQDALRAQMVLSRSQEGRGTRGPLERLAEAPSPAPTPSPTPVEDLGPQTSTSPGRLSPDFAEELRSLEPSPSPGPQEEDGEVALVLLGRPSPGAVGPEDVALCSSRRPVRPGRRGLGPVPS
        """
        if multi:
            for record in SeqIO.parse(fin_fasta, 'fasta'):
                ID = record.id
                ID = ID.split(' ')[0]
                if '|' in ID:ID = ID.split('|')[1]
                seq = str(record.seq)
                yield ID, seq
        else:
            with open(fin_fasta, 'r')as fo:
                line = fo.readline()
                while (line):
                    ID = line[1:-1]
                    ID = ID.split(' ')[0]
                    if '|' in ID: ID = ID.split('|')[1]
                    line = fo.readline()
                    seq = line[:-1]
                    line = fo.readline()
                    yield ID,seq
    def seq2num(self,seq):
        """
        FastaDealer().seq2num('GAVLIPFWMYSTCNQDEKRH')
        Out[4]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        :param seq:
        :return:
        """
        # list1 = 'GAVLIPFWMYSTCNQDEKRHXOUBZJ'
        list1 = 'GAVLIPFWMYSTCNQDEKRH'
        numList = []
        for e in seq:
            if e not in list1:return []
            numList.append(list1.index(e)+1)
        return numList


    def splitFasta(self,fin_fasta,dirout_fasta,maxnum):
        count = 0
        numbers = 0
        fout_fasta = os.path.join(dirout_fasta, '%d.txt' % numbers)
        fo = open(fout_fasta, 'w')
        with open(fin_fasta,'r') as fi:
            line = fi.readline()
            while(line!=''):
                if '>' in line:
                    count = count + 1
                    if count > maxnum:
                        count = 0
                        numbers = numbers + 1
                        fo.flush()
                        fo.close()
                        fout_fasta = os.path.join(dirout_fasta,'%d.txt' % numbers)
                        fo = open(fout_fasta, 'w')
                        print(fout_fasta)
                fo.write(line)
                fo.flush()
                line = fi.readline()
        fo.flush()
        fo.close()