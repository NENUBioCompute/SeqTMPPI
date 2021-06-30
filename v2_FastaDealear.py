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

    def getPhsi_Blos(self,fin_fasta,out_dir,multi=True,checkprotein = True):
        check_path(out_dir)
        p = Protein()
        for ID, seq in self.getYield(fin_fasta,multi=multi):
            if not checkprotein or p.checkProtein(seq, 50, 2000, uncomm=True):
                filename = os.path.join(out_dir,"%s.npy"%ID)
                result =self.phsi_blos(seq)
                if len(result)!=0:
                    np.save(filename,result)

    def convertSampleToPhysicsVector_pca(self,seq):
        """
        Convertd the raw data to physico-chemical property
        PARAMETER
        seq: "MLHRPVVKEGEWVQAGDLLSDCASSIGGEFSIGQ" one fasta seq
            X denoted the unknow amino acid.
        probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
        """
        letterDict = {}
        letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181]
        letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364]
        letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055]
        letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156]
        letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374]
        letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112]
        letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303]
        letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106]
        letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021]
        letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107]
        letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206]
        letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027]
        letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077]
        letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208]
        letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384]
        letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196]
        letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274]
        letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297]
        letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091]
        letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299]
        letterDict["X"] = [0, -0.00005, 0.00005, 0.0001, -0.0001]
        letterDict["-"] = [0, 0, 0, 0, 0, 1]
        AACategoryLen = 5  # 6 for '-'
        l = len(seq)
        probMatr = np.zeros((l, AACategoryLen))
        AANo = 0
        for AA in seq:
            if not AA in letterDict:
                probMatr[AANo] = np.full(AACategoryLen, 0)
            else:
                probMatr[AANo] = letterDict[AA]

            AANo += 1
        return probMatr

    def convertSampleToBlosum62(self,seq):
        letterDict = {}
        letterDict["A"] = [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0]
        letterDict["R"] = [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3]
        letterDict["N"] = [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3]
        letterDict["D"] = [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3]
        letterDict["C"] = [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1]
        letterDict["Q"] = [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2]
        letterDict["E"] = [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2]
        letterDict["G"] = [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3]
        letterDict["H"] = [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3]
        letterDict["I"] = [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3]
        letterDict["L"] = [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1]
        letterDict["K"] = [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2]
        letterDict["M"] = [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1]
        letterDict["F"] = [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1]
        letterDict["P"] = [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2]
        letterDict["S"] = [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2]
        letterDict["T"] = [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0]
        letterDict["W"] = [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3]
        letterDict["Y"] = [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1]
        letterDict["V"] = [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
        AACategoryLen = 20  # 6 for '-'
        l = len(seq)
        probMatr = np.zeros((l, AACategoryLen))
        AANo = 0
        for AA in seq:
            if not AA in letterDict:
                probMatr[AANo] = np.full(AACategoryLen, 0)
            else:
                probMatr[AANo] = letterDict[AA]

            AANo += 1
        return probMatr

    def readPSSM(self,pssmfile):
        pssm = []
        with open(pssmfile, 'r') as f:
            count = 0
            for eachline in f:
                count += 1
                if count <= 3:
                    continue
                if not len(eachline.strip()):
                    break
                line = eachline.split()
                pssm.append(line[2: 22])  # 22:42
        return np.array(pssm)
    def phsi_blos(self,seq):
        PhyChem = self.convertSampleToPhysicsVector_pca(seq)
        pssm = self.convertSampleToBlosum62(seq)
        pssm = pssm.astype(float)
        pssm = np.concatenate((PhyChem, pssm), axis=1)
        return pssm

    def phsi_pssm(self,seq,pssmfile):
        PhyChem = self.convertSampleToPhysicsVector_pca(seq)
        pssm = self.readPSSM(pssmfile)
        pssm = pssm.astype(float)
        pssm = np.concatenate((PhyChem, pssm), axis=1)
        return pssm