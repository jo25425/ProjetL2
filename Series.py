# TODO
# + Cleanup
# + dump methode
# + Easy switch faq/pc
# + Traitement du texte
# + Ajout Reversed WrdKey
# + Ajout sélection inverse des paths
# + Refactore AddStrToMat to AddEpiToRow
# + Matrice conversion for stats
# + Gestion des langues
# + Ajout AddSeason
# - Groupement par k-means
# - Traitement de texte + précis

# - Ajout total
# - Ajout temps
# - Ajout Counter de Collections
# - Ajout en groupe

###Idées
###Ajouter des groupes à l'algo k-means jusqu'à ce que tous les groupes soient de taille inférieure à n(Mylil_matrix.subgroups!)
###Représenter les séries dans le plan où la droite des abscisses est le cosinus similarité avec un vecteur et la droite des ordonnées, celui avec un autre vecteur
###Représenter les series dans un espace de dimension n comme celui d'au dessus, on peut alors trier les vecteurs qu'on représente(totalité des séries) par norme pour établir une liste triée qu'une personne pourrait aimer

import numpy
from scipy.sparse import *
from scipy.io import mmwrite, mmread
from math import sqrt
from collections import Counter
import os
import re
import pickle
import nltk
import random
import cProfile

if os.environ['COMPUTERNAME'] == 'TIE':
    pathDumps = 'C:/Coding/Python workshop/Cours/Projet L2'
    pathData = 'E:/Documents/Programmes/addic7ed'
else:
    pathDumps = '/tmp'
    pathData = '/tmp/addic7ed'

class My_lil_matrix():
    def __init__(self,arg):

        """
        LInked List type of sparse matrix made to work with scipy.sparse
        :param tuple or list or scipy.sparse.lil_matrix or scipy.sparse.csr_matrix:
        """
        if type(arg)==tuple or type(arg)==list:
            self.shape=list(arg)
            self.rows=[list() for i in range(self.shape[0])]
            self.data=[list() for i in range(self.shape[0])]

        elif isspmatrix_lil(arg):
            self.shape=list(arg.shape)
            self.rows=[list(i) for i in arg.rows]
            self.data=[list(i) for i in arg.data]
        elif isspmatrix_csr(arg):
            self.shape=list(arg.shape)
            self.rows=[arg.indices[arg.indptr[i]:arg.indptr[i+1]].tolist() for i in range(self.shape[0])]
            self.data=[arg.data[arg.indptr[i]:arg.indptr[i+1]].tolist() for i in range(self.shape[0])]

    def resize(self,shape):
        """
        Resize the matrix to the new shape by suppressing data in suppressed rows or adding zeros.
        :type shape: tuple of integers
        """
        if shape[0]<self.shape[0]:
            self.rows=self.rows[:shape[0]]
            self.data=self.data[:shape[0]]

        else:
            self.rows+=[list() for i in range(shape[0]-self.shape[0])]
            self.data+=[list() for i in range(shape[0]-self.shape[0])]
        self.shape[0]=shape[0]

        if shape[1]<self.shape[1]:
            self.rows=[[i for i in j if i<shape[1]] for j in self.rows]
            self.data=[self.data[i][:len(self.rows[i])] for i in range(self.shape[0])]

        self.shape[1]=shape[1]

    def combine(self,Matrixes,copy=True):

        """
        Return a matrix that's the vertical stacking of self and the matrixes in Matrixes.
        The number of a column of the result is adapted to the new matrixes.
        self isn't modified if copy is True, it is modified then returned otherwise.
        :param Matrixes : list of My_lil_matrix matrixes:
        :type copy boolean:
        :return:
        """
        if copy:
            Mat=self.copy()
        else:
            Mat=self
        for ToAddMat in Matrixes:
            Mat.shape[0]+=ToAddMat.shape[0]
            Mat.shape[1]=max(Mat.shape[1],ToAddMat.shape[1])
            Mat.data+=ToAddMat.data
            Mat.rows+=ToAddMat.rows
        return Mat

    def removerow(self,Rows,copy=False):
        """
        Return a matrix with rows removed by popping them from the data.
        Returned matrix is self is copy is False and a copy of self if copy is True.
        Not advised if the matrix is indexed, see removerowsind.
        :param copy: boolean
        :type Rows: List of integers
        """
        if copy:
            Mat=self.copy()
        else:
            Mat=self
        Rows=list(set(Rows))
        Rows.sort()
        Rows.reverse()
        for i in Rows:
            Mat.rows.pop(i)
            Mat.data.pop(i)
        Mat.shape[0]-=len(Rows)

    def swaplines(self,Gr1,Gr2,copy=False):
        """
        Swap the lines indicated by Gr1 with the lines indicated by Gr2.
        Returns a new matrix if copy is True and returns self if copy is False.
        :type Gr1: List of indices
        :type Gr2: List of indices
        """
        if copy:
            Mat=self.copy()
        else:
            Mat=self
        for i,j in zip(Gr1,Gr2):
            td,tr=Mat.data[i],Mat.rows[i]
            Mat.data[i],Mat.rows[i]=Mat.data[j],Mat.rows[j]
            Mat.data[j],Mat.rows[j]=td,tr

    def removerowsind(self,Rows):
        """
        Remove the rows indicated by Rows, they're first swapped with the last rows of the matrix then deleted,
         the function return the previous indexes of the rows that are now at the deleted rows's place.
        :param Rows: List of indices of rows to be removed
        :return : List of rows now at the place of the removed rows, they're taken from the last rows
        """
        R=list(range(self.shape[0]-len(Rows),self.shape[0]))
        self.swaplines(Rows,R)
        self.resize((self.shape[0]-len(Rows),self.shape[1]))
        return R

    def subgroups(self,SubGroups):
        #each element of SubGroups is a list of indices to be added to the group
        """
        Each element of SubGroups must be a list of indexes, the function will return a list of matrixes with rows corresponding to the SubGroups lists of indexes.
        :param SubGroups: list of list of indexes
        :return:
        """
        NewMatrixes=[My_lil_matrix((len(SubGroups[i]),self.shape[1])) for i in range(len(SubGroups))]
        for i in range(len(SubGroups)):
            NewMatrixes[i].data=[self.data[j] for j in SubGroups[i]]
            NewMatrixes[i].rows=[self.rows[j] for j in SubGroups[i]]
        return NewMatrixes

    def transpose(self):
        NewMat=My_lil_matrix((self.shape[1],self.shape[0]))
        for i in range(self.shape[0]):
            for j in range(len(self.rows[i])):
                NewMat.rows[self.rows[i][j]].append(i)
                NewMat.data[self.rows[i][j]].append(self.data[i][j])
        return NewMat

    def copy(self):
        NewMat=My_lil_matrix(self.shape)
        NewMat.rows=[list(i) for i in self.rows]
        NewMat.data=[list(i) for i in self.data]
        return NewMat

    def apply(self, f, copy=False,axis=0):
        """
        Applies the function f to every non zero cell of the matrix if axis is 0.
        Applies the function f to every row if axis is 1.
        Return and doesn't modify itself if copy is True.
        Return a new matrix with f applied otherwise
        :type axis: int equal to 0 or 1
        :type copy: boolean
        :type f: function that takes 1 argument and returns 1 value
        """
        if copy:
            Mat=self.copy()
        else:
            Mat=self
        if axis==0:
            for i in range(Mat.shape[0]):
                for j in range(len(Mat.rows[i])):
                    Mat.data[i][j]=f(Mat.data[i][j])
        elif axis==1:
            for i in range(Mat.shape[0]):
                Mat.data[i]=[f(Mat.data[i])]
                Mat.rows[i]=[0]
                Mat.shape=[Mat.shape[0],1]
        return Mat

    def sum(self,axis=1,copy=True):

        if copy:
            Mat=self.copy()
        else:
            Mat=self
        for i in range(Mat.shape[0]):
            Mat.rows[i]=[0]
            Mat.data[i]=[sum(Mat.data[i])]
            Mat.shape=[Mat.shape[0],1]
        if axis==1:
            return Mat
        elif axis==0:
            return sum([i[0] for i in Mat.data])

    def cossim(self,Mat2):

        n=min(self.shape[0],Mat2.shape[0])
        Res=[0]*n
        def tmp(list):
            res=0
            for i in list:
                res+=i*i
            return sqrt(res)
        NrmMat1=self.apply(tmp,True,1)
        NrmMat2=Mat2.apply(tmp,True,1)
        for i in range(n):
            Res[i]=sum([a*b for a,b in zip(self.data[i],Mat2.data[i])])/(NrmMat1.data[i][0]*NrmMat2.data[i][0])
        return Res

    def averagerow(self):
        Res=My_lil_matrix([1,self.shape[1]])
        ResDict=Counter()
        for i in range(self.shape[0]):
            for key,value in zip(self.rows[i],self.data[i]):
                ResDict[key]+=value
        V=list(ResDict.items())
        V.sort(key=lambda x:x[0])
        for i,j in V:
            Res.data[0].append(j/self.shape[0])
            Res.rows[0].append(i)
        return Res

    def dot(self,other):
        return self.tocsr().dot(other)

    def tocsr(self):
        indices=[]
        indptr=[0]
        data=[]
        j=0
        for i in range(self.shape[0]):
            data+=self.data[i]
            indices+=self.rows[i]
            j+=len(self.rows[i])
            indptr.append(j)
        return csr_matrix((data,indices,indptr),tuple(self.shape))

    def tolil(self):
        m=lil_matrix(tuple(self.shape))
        m.rows=self.rows
        m.data=self.data
        return m

class Projet():
    def __init__(self, EpiMat=None, nrow=0):

        # Initialising variables
        self.WrdKey = dict()
        self.SsnKey = dict()
        self.RevSsnKey = dict()
        self.RevWrdKey = dict()
        if EpiMat:
            self.EpiMat = dok_matrix(EpiMat)
        else:
            self.EpiMat = dok_matrix((nrow, 0), dtype=int)
        self.NbrWrd = self.EpiMat.shape[1]

        #Initialising Constants

        self.Initialised=0
        self.pathDumps = pathDumps
        self.Languages = ['english']

        # Regular expressions used to treat strings

        self.SsnPat = re.compile(r'(\d+)___(\S+)')
        self.EpiPat = re.compile(r'(\d+)__(\S+).txt')
        self.SubPat = re.compile(
            r'(\d+)\n(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)\n(.*?)(?=(\n\d+\n)|\Z)', re.DOTALL)
        self.TrtPat = re.compile(r'\W+')

        # Logging failures
        self.ReadErr = []
        self.LangErr = []

    def TxtTrt(self, Text):
        '''Prends en argument une chaine de charactères, retourne une liste de mots'''
        LstWrd = self.TrtPat.sub(' ', Text).lower().split(' ')

        return LstWrd

    def InitStats(self):
        if self.Initialised:
            return
        self.EpiMat=My_lil_matrix(self.EpiMat.tolil())
        self.EpiMat.apply(float)
        print('Matrix format changed to Lil, do not add more series')
        self.Initialised=1

    def CleanUpEpiMat(self):
        if self.Initialised==0:
            self.InitStats()
        self.UpdateReversedWrdKey()
        RowToDel = []
        ColToDel = []
        n = self.EpiMat.shape[0]
        m = self.EpiMat.shape[1]
        Mat = self.EpiMat.copy()

        LangMat = None
        for Lang in nltk.corpus.stopwords._fileids:
            stpwrds = nltk.corpus.stopwords.words(Lang)
            line = dok_matrix((1, m), dtype=int)
            for wrd in stpwrds:
                try:
                    line[0, self.WrdKey[wrd]] = 1
                except KeyError:
                    continue
            if LangMat != None:
                LangMat = vstack([LangMat, line],format='csr')
            else:
                LangMat = line

        LangMat = LangMat.tocsc().transpose()
        LangMat = Mat.dot(LangMat).toarray()
        LangMat = LangMat.argmax(1)

        for i in range(n):
            if nltk.corpus.stopwords._fileids[LangMat[i]] not in self.Languages:
                RowToDel.append(i)

        print('Starting to remove ', len(RowToDel), ' series')
        l=Mat.removerowsind(RowToDel)

        ##Adjusting indices
        for i in range(len(RowToDel)):
            t1=self.RevSsnKey[RowToDel[i]]
            t2=self.RevSsnKey[l[i]]
            self.SsnKey[t2]=RowToDel[i]
            self.RevSsnKey[RowToDel[i]]=t2
            self.SsnKey.pop(t1)
            self.RevSsnKey.pop(l[i])


        print(len(RowToDel),' series removed')

        ##Moving onto columns
        Mat=Mat.transpose()
        self.UpdateReversedWrdKey()
        #Filtering columns
        NMat=[i for i in range(Mat.shape[0]) if len(Mat.rows[i])==Mat.shape[1] or len(Mat.rows[i])==0]
        ColToDel+=NMat

        print('Starting to remove ', len(ColToDel),' words')
        l=Mat.removerowsind(ColToDel)
        #return ColToDel, Mat, l

        #Adjusting indices
        for i in range(len(ColToDel)-1,-1,-1):
            t1=self.RevWrdKey[ColToDel[i]]
            t2=self.RevWrdKey[l[i]]
            self.WrdKey[t2]=ColToDel[i]
            self.RevWrdKey[ColToDel[i]]=t2
            self.WrdKey.pop(t1)
            self.RevWrdKey.pop(l[i])
        print(len(ColToDel),' words removed')

        #done
        Mat=Mat.transpose()
        self.EpiMat=Mat

    def GrpByK(self,k,PrtInd=[]):
        print('Starting GrbByK')
        Mat=self.EpiMat
        PrtList=random.sample(range(self.EpiMat.shape[0]),k)
        for i in range(len(PrtInd)):
            if PrtInd[i] not in PrtList:
                PrtList[i]=PrtInd[i]

        print('Selected rows as prototypes : ', PrtList)
        PrtMat=Mat.subgroups([PrtList])[0]
        OldPrt=PrtMat.copy()
        print('Rows put into matrix format')

        while True:
            Grps=Mat.dot(PrtMat.transpose().tocsr())
            print('Normalisation')

            NrmMat=Mat.apply(lambda x:x*x,copy=True).apply(sum,axis=1).apply(sqrt)
            NrmPrt=PrtMat.apply(lambda x:x*x,copy=True).apply(sum,axis=1).apply(sqrt).transpose()

            NrmMat=NrmMat.dot(NrmPrt.tocsr())
            Grps=numpy.divide(Grps,NrmMat)
            Grps=Grps.argmax(1).tolist()

            print('Calcul des nouveaux prototypes')
            OldPrt=PrtMat.copy()
            PrtMat=[list() for i in range(PrtMat.shape[0])]
            for i in range(Mat.shape[0]):
                PrtMat[Grps[i][0]].append(i)
            PrtMat=Mat.subgroups(PrtMat)
            PrtMat=[i.averagerow() for i in PrtMat]
            PrtMat=PrtMat[0].combine(PrtMat[1:])
            print('Nouveaux prototypes calculés, comparaison avec les anciens')
            if min(OldPrt.cossim(PrtMat))>0.99:
                break

        return Grps,OldPrt

    def UpdateReversedWrdKey(self):
        self.RevWrdKey = {key: word for (word, key) in self.WrdKey.items()}
        self.RevSsnKey = {key: word for (word, key) in self.SsnKey.items()}

    def dump(self):
        '''Ecrit self.EpiMat, self.WrdKey, self.SsnKey, self.StatsMat sur le disque sous forme de fichiers .dump.
Le répertoire utilisé est self.pathDumps'''
        file = open(self.pathDumps + '/EpiMat.dump', 'w+b')
        mmwrite(file, self.EpiMat)
        file.close()

        file = open(self.pathDumps + '/WrdKey.dump', 'w+b')
        pickle.dump(self.WrdKey, file)
        file.close()

        file = open(self.pathDumps + '/SsnKey.dump', 'w+b')
        pickle.dump(self.SsnKey, file)
        file.close()

        # file = open(self.pathDumps + '/StatsMat.dump', 'w+b')
        # mmwrite(file, self.StatsMat)
        # file.close()

    def load(self):
        '''Charge self.EpiMat, self.WrdKey, self.SsnKey, self.StatsMat depuis le répertoire spécifié par self.pathf.
Les fichiers EpiMat.dump et StatsMat.dump sont au format renvoyé par scipy.io.mmwrite tandis que self.WrdKey et self.SsnKey sont au format utilisé par le protocole par défaut de pickle'''
        file = open(self.pathDumps + '/EpiMat.dump', 'r+b')
        self.EpiMat = mmread(file).todok()
        file.close()

        file = open(self.pathDumps + '/WrdKey.dump', 'r+b')
        self.WrdKey = pickle.load(file)
        file.close()

        file = open(self.pathDumps + '/SsnKey.dump', 'r+b')
        self.SsnKey = pickle.load(file)
        file.close()

        # file = open(self.pathDumps + '/StatsMat.dump', 'r+b')
        # self.StatsMat = mmread(file).tocsr()
        # file.close()

        self.UpdateReversedWrdKey()

    def AddEpiToRow(self, Text, Row):

        M = self.EpiMat
        Key = self.WrdKey
        Data = self.SubPat.findall(Text)


        EpiWrds = '\n'.join([m[9] for m in Data])

        LstWrd = self.TxtTrt(EpiWrds)

        for i in LstWrd:
            if i == '':
                continue
            try:
                M[Row, Key[i]] += 1

            except KeyError:  # Cas où le mot est rencontré pour la première fois

                Key[i] = self.NbrWrd
                self.NbrWrd += 1
                M.resize((M.shape[0], self.NbrWrd))

                M[Row, self.NbrWrd - 1] = 1

    def AddSeries(self, Path, m=-1, Numbers=[]):
        """

        :type Numbers: IntList
        :type m: Int
        :type Path: String
        """
        LstSri = os.listdir(Path)
        LstSri.remove('grab.txt')
        i = 0
        Numbers = [str(m) for m in Numbers]
        Series = [S for S in LstSri if S.split('__')[0] in Numbers]
        if m > 0 and m < len(Series):
            m = 0
        else:
            m -= len(Series)
        while m != 0:
            Series.append(LstSri[i])
            i += 1
            m -= 1
        nbS=0
        for S in Series:
            nbS+=self.AddSerie(Path + '/' + S)
        print(nbS,' séries ajoutées')

    def AddSerie(self, Path):
        '''Path doit lier à un dossier dont le nom est de la forme spécifiée par SsnPat ('(\d+)___(\S+)'). Le séparateur du path doit être '/' 
        :type Path: String
        '''

        #Seasons = {}
        SriTitle = self.SsnPat.match(Path.split('/')[-1]).group(2)
        print('Processing ', SriTitle, 'at', Path)
        LstSsn = os.listdir(Path + '/')
        LstSsn.sort()

        NbrToAdd = 0
        for Season in LstSsn:
            if '.txt' in Season:
                continue
            LstEpi = os.listdir(Path + '/' + Season)
            NbrToAdd += len(LstEpi)
        if NbrToAdd == 1:
            print('Skipping because of low number of episodes')
            return 0

        NbrSri = self.EpiMat.shape[0]
        self.EpiMat.resize((NbrSri+1,self.EpiMat.shape[1]))

        self.SsnKey[SriTitle]=NbrSri

        for Season in LstSsn:
            if '.txt' in Season:  # Not a season
                continue

            LstEpi = os.listdir(Path + '/' + Season)
            LstEpi.sort()
            if len(LstEpi) == 0:
                continue

            for Epi in LstEpi:

                File = open(Path + '/' + Season + '/' + Epi, 'r', encoding="utf8")
                print(SriTitle, '   ',Season, '  ', Epi)

                try:  # test de l'encoding utf8
                    Contents = File.read()

                except UnicodeDecodeError:
                    File.close()

                    try:  # En cas d'erreur, essai de l'encoding latin-1
                        File = open(Path + '/' + Season + '/' + Epi, 'r', encoding="latin-1")
                        Contents = File.read()

                    except UnicodeDecodeError:  # Si toujours erreur, on passe au suivant en ajoutant cela aux erreurs
                        File.close()
                        self.ReadErr.append([Epi, Season, Path])
                        print('Erreur de décodage : Epi' + Epi + ' Season ' + Season + 'Path' + Path)
                        continue

                File.close()

                self.AddEpiToRow(Contents, NbrSri)
        return 1


def Language(Text):
    L = nltk.word_tokenize(Text.lower())
    words_set = set(L)

    stpwrds = nltk.corpus.stopwords
    Ratios = {}
    for Lang in stpwrds._fileids:
        stopwords = set(stpwrds.words(Lang))
        common_elements = words_set.intersection(stopwords)
        Ratios[Lang] = len(common_elements)
    m = max(Ratios, key=Ratios.get)
    #    print(m,Ratios[m],len(words_set))
    return m


def go(n=10, N=[1000,53,15,1235]):
    Test.AddSeries(pathData, m=n, Numbers=N)
    Test.dump()
def g():
    Test.load()
    Test.CleanUpEpiMat()
    t=Test.GrpByK(10)
    with open(pathDumps+'/Kmeansdata100.txt','a') as F:
        print([t[0].count([i]) for i in range(10)],file=F)
    print('Done')


if __name__ == '__main__':
    Test = Projet()
    l = os.listdir(pathData)


# tmp : t=sum([sum([Test.StatsMat.getrow(i) for i in p.values()]) for p in m])
