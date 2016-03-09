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

import os
import pickle
import random
import re
from math import sqrt

import nltk
import numpy
import scipy.sparse
from scipy.io import mmwrite, mmread

from My_lil_matrix import My_lil_matrix

if os.environ['COMPUTERNAME'] == 'TIE':
    pathDumps = 'C:/Coding/Python workshop/Cours/Projet L2'
    pathData = 'E:/Documents/Programmes/addic7ed'
else:
    pathDumps = '/tmp'
    pathData = '/tmp/addic7ed'


class Projet():
    def __init__(self, EpiMat=None, nrow=0):

        # Initialising variables
        self.WrdKey = dict()
        self.SsnKey = dict()
        self.RevSsnKey = dict()
        self.RevWrdKey = dict()
        if EpiMat:
            self.EpiMat = scipy.sparse.dok_matrix(EpiMat)
        else:
            self.EpiMat = scipy.sparse.dok_matrix((nrow, 0), dtype=int)
        self.NbrWrd = self.EpiMat.shape[1]

        #Initialising Constants

        self.Initialised=0
        self.pathDumps = pathDumps
        self.Languages = ['english']
        self.Stemmer=nltk.stem.SnowballStemmer('english')

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
        LstWrd=[self.Stemmer.stem(i) for i in LstWrd]
        return LstWrd

    def InitStats(self):
        if self.Initialised:
            return
        self.EpiMat= My_lil_matrix(self.EpiMat.tolil())
        self.EpiMat.apply(float)
        print('Matrix format changed to Lil, do not add more series')
        self.Initialised=1

    def CleanUpStatsMat(self, maxDF=100, minDF=5):
        """
Remove lines and rows from self.EpiMat.
Currently removes rows of languages not in self.languages.
Currently removes columns with a Document Frequency DF higher than maxDF% or lower than minDF.(flat amount)
        :param maxDF:
        :param minDF:
        """
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
            line = scipy.sparse.dok_matrix((1, m), dtype=int)
            for wrd in stpwrds:
                try:
                    line[0, self.WrdKey[wrd]] = 1
                except KeyError:
                    continue
            if LangMat != None:
                LangMat = scipy.sparse.vstack([LangMat, line], format='csr')
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
        maxDF=Mat.shape[1]*maxDF/100
        NMat=[i for i in range(Mat.shape[0]) if len(Mat.rows[i])>=maxDF or len(Mat.rows[i])<=minDF]
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
        PrtList=random.sample(range(self.EpiMat.shape[0]), k)
        for i in range(len(PrtInd)):
            if PrtInd[i] not in PrtList:
                PrtList[i]=PrtInd[i]

        print('Selected rows as prototypes : ', PrtList)
        PrtMat=Mat.subgroups([PrtList])[0]
        OldPrt=PrtMat.copy()
        print('Rows put into matrix format')
        NrmMat=Mat.apply(lambda x:x*x,copy=True).apply(sum,axis=1).apply(sqrt)

        while True:
            Grps=Mat.dot(PrtMat.transpose().tocsr())
            print('Normalisation')

            NrmPrt=PrtMat.apply(lambda x:x*x,copy=True).apply(sum,axis=1).apply(sqrt).transpose()

            Grps=numpy.divide(Grps,NrmMat.dot(NrmPrt.tocsr()))
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
            if min(OldPrt.cossimrowtorow(PrtMat))>0.99:
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
            if LstSri[i].split('  ')[0] in Numbers:
                i+=1
                continue
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
    Test.CleanUpStatsMat(70, 5)
    t=Test.GrpByK(10)
    with open(pathDumps+'/Kmeansdata100.txt','a') as F:
        print([t[0].count([i]) for i in range(10)],file=F)
    print('Done')


if __name__ == '__main__':
    Test = Projet()
    l = os.listdir(pathData)


# tmp : t=sum([sum([Test.StatsMat.getrow(i) for i in p.values()]) for p in m])
