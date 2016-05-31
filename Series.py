# TODO
# + Cleanup
# - dump methode
# - Easy switch faq/pc
# + Traitement du texte
# + Ajout Reversed WrdKey
# + Ajout sélection inverse des paths
# - Refactore AddStrToMat to AddEpiToRow
# - Matrice conversion for stats
# + Gestion des langues
# + Ajout AddSeasonLil
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
from math import sqrt,log
import bisect
import nltk
import numpy as np
import scipy.sparse
from scipy.io import mmwrite, mmread
from collections import Counter
import treetaggerwrapper
import pdb

from My_lil_matrix import My_lil_matrix

if os.environ['COMPUTERNAME'] == 'TIE':
    pathDumps = 'C:/Users/Vivien/PycharmProjects/ProjetL2/dumps'
    pathData = 'E:/Documents/Programmes/addic7ed'
elif os.environ['COMPUTERNAME'] == 'Janice':
    pathDumps = 'C:\projet l2'
    pathData = 'C:\tmp\addic7ed\addic7ed'
else:
    pathDumps = '/tmp'
    pathData = '/tmp/addic7ed'


class Projet():
    def __init__(self,Dumps=pathDumps,Data=pathData):

        # Initialising variables
        self.WrdKey = dict()
        self.SsnKey = dict()
        self.RevSsnKey = dict()
        self.RevWrdKey = dict()
        self.StatsMat=My_lil_matrix((0, 0))

        self.SriData = []
        self.SsnData= [] #list of tuples
        self.EpiData = [] #list of tuples
        self.KGroupes=[] #list of numbers
        self.Prototypes=[] # list of lists

        #Initialising Constants

        self.pathDumps = Dumps
        self.pathData = Data
        self.cur_title = None
        self.Languages = ['english']
        self.Stemmer=nltk.stem.SnowballStemmer('english')
        self.TreeTagger=treetaggerwrapper.TreeTagger(TAGLANG='en')

        # Regular expressions used to process strings

        self.SsnPat = re.compile(r'(\d+)___(\S+)')
        self.EpiPat = re.compile(r'(\d+)__(\S+).txt')
        self.SubPat = re.compile(
            r'(\d+)\n(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)\n(.*?)(?=(\n\d+\n)|\Z)', re.DOTALL)
        self.TrtPat = re.compile(r" +")

        # Logging failures
        self.Skipped = []
        self.ReadErr = []
        self.LangErr = []

    def TxtTrt(self, Text):
        '''Prends en argument une chaine de charactères, retourne une liste de mots'''
        Text = self.TrtPat.sub(' ', Text)
        LstWrd = [i[1]+'_'+i[2] for i in (j.split('\t') for j in self.TreeTagger.tag_text(Text,notagdns=True,notagemail=True,notagip=True,notagurl=True)) ]
        return LstWrd

    def InitStats(self,maxDF=100,minDF=0,TF=True,DF=True,copy=True,Smax=5000):
        self.StatsMat.apply(float)
        print('Matrix values changed to floats')
        self.CleanUpStatsMatLil(maxDF,minDF,Smax)
        def TFnorm(list):
            s=sum(list)
            for i in range(len(list)):
                list[i]=list[i]/s
            return list
        def DFnorm(list):
            D=self.StatsMat.shape[0]
            for i in range(len(list)):
                list[i] *= log(D / len(list))
            return list
        if DF:
            self.StatsMat=self.StatsMat.transpose()
            self.StatsMat.apply(DFnorm,axis=2)
            self.StatsMat=self.StatsMat.transpose()
        if TF:
            self.StatsMat.apply(TFnorm,axis=2)

    def RemoveSeries(self,Series):
        '''

        :param Series: List of indices
        :return:
        '''

        Change=self.StatsMat.removerowsind2(Series)
        for new, old in Change.items():
            self.RevSsnKey[new]=self.RevSsnKey[old]
            del self.RevSsnKey[old]
        self.UpdateDict(FromSsn=False)

    def RemoveWords(self,Words):
        Mat=self.StatsMat.transpose()
        Change=Mat.removerowsind2(Words)
        for new, old in Change.items():
            self.RevWrdKey[new]=self.RevWrdKey[old]
            del self.RevWrdKey[old]
        self.UpdateDict(FromWrd=False)
        self.StatsMat=Mat.transpose()

    def FlagLanguages(self,data=None):

        if not data:
            data=self.StatsMat

        LangMat=My_lil_matrix((0,data.shape[1]))
        for Lang in nltk.corpus.stopwords._fileids:
            stpwrds = nltk.corpus.stopwords.words(Lang)
            C=Counter()
            for i in stpwrds:
                if 'DT_'+i in self.WrdKey:
                    C[self.WrdKey['DT_'+i]]=1
                elif 'IN_'+i in self.WrdKey:
                    C[self.WrdKey['IN_'+i]]=1
                elif 'PP_'+i in self.WrdKey:
                    C[self.WrdKey['PP_'+i]]=1
                elif 'NP_'+i in self.WrdKey:
                    C[self.WrdKey['NP_'+i]]=1
            if C:
                LangMat.resize((LangMat.shape[0]+1,LangMat.shape[1]))
                LangMat.AddToRow(C,LangMat.shape[0]-1)
        LangMat = LangMat.tocsr().transpose()
        LangMat = data.dot(LangMat).toarray()
        LangMat = LangMat.argmax(1)

        return LangMat

    def CleanUpStatsMatLil(self, maxDF=100, minDF=5, Smax=5000):
        self.UpdateDict()
        RowToDel = []
        ColToDel = []
        n = self.StatsMat.shape[0]
        m =self.StatsMat.shape[1]

        LangMat = self.FlagLanguages()

        RowToDel += [i for i in range(n) if nltk.corpus.stopwords._fileids[LangMat[i]] not in self.Languages]

        print('Starting to remove ',len(RowToDel), 'series')
        self.RemoveSeries(RowToDel)

        ##Moving onto columns
        Mat = self.StatsMat.transpose()

        # Filtering columns
        maxDF = int(Mat.shape[1] * maxDF / 100)
        NMat = [(len(Mat.data[i]), i) for i in range(Mat.shape[0])]
        NMat.sort(key=lambda x: x[0])
        r = bisect.bisect_left(NMat, (maxDF, 0))
        l = bisect.bisect(NMat, (minDF, Mat.shape[0]))
        ColToDel += [i[1] for i in NMat[:l]] + [i[1] for i in NMat[r:]]
        if r - l > Smax:
            ColToDel += [i[1] for i in NMat[l:r - Smax]]

        print('Starting to remove ', len(ColToDel), ' words')
        self.RemoveWords(ColToDel)
        print(len(ColToDel), ' words removed')

        # done

    def CleanUpStatsMat(self, maxDF=100, minDF=5, Smax=5000):
        """
Remove lines and rows from self.StatsMat.
Currently removes rows of languages not in self.languages.
Currently removes columns with a Document Frequency DF higher than maxDF% or lower than minDF.(flat amount)
        :param maxDF:
        :param minDF:
        """
        self.UpdateDict()
        RowToDel = []
        ColToDel = []
        n = self.StatsMat.shape[0]
        m = self.StatsMat.shape[1]
        Mat = self.StatsMat.copy()

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
        self.RevSsnKey=Mat.removerowsind(RowToDel, self.RevSsnKey)
        self.UpdateDict(FromSsn=0)


        print(len(RowToDel),' series removed')

        ##Moving onto columns
        Mat=Mat.transpose()

        #Filtering columns
        maxDF=int(Mat.shape[1]*maxDF/100)
        NMat=[(len(Mat.data[i]),i) for i in range(Mat.shape[0])]
        NMat.sort(key=lambda x:x[0])
        r=bisect.bisect_left(NMat,(maxDF,0))
        l=bisect.bisect(NMat,(minDF,Mat.shape[0]))
        ColToDel +=[i[1] for i in NMat[:l]]+[i[1] for i in NMat[r:]]
        if r-l>Smax:
            ColToDel+=[i[1] for i in NMat[l:r-Smax]]

        print('Starting to remove ', len(ColToDel),' words')
        self.RevWrdKey=Mat.removerowsind(ColToDel, self.RevWrdKey)
        self.UpdateDict(FromWrd=0)
        print(len(ColToDel),' words removed')

        #done
        Mat=Mat.transpose()
        self.StatsMat=Mat

    def GrpByK(self,k,PrtInd=()):
        print('Starting GrbByK')
        Mat=self.StatsMat
        PrtList=random.sample(range(self.StatsMat.shape[0]), k)
        for i in range(len(PrtInd)):
            if PrtInd[i] not in PrtList:
                PrtList[i]=PrtInd[i]

        print('Selected rows as prototypes : ', PrtList)
        PrtMat=Mat.subgroups([PrtList])[0]
        OldPrt=PrtMat.copy()
        print('Rows put into matrix format')
        NrmMat=Mat.apply(lambda x:x**2,copy=True).apply(sum,axis=1).apply(sqrt)

        while True:
            Grps=Mat.dot(PrtMat.transpose().tocsr())
            print('Normalisation')

            NrmPrt=PrtMat.apply(lambda x:x*x,copy=True).apply(sum,axis=1).apply(sqrt).transpose()

            Grps= np.divide(Grps, NrmMat.dot(NrmPrt.tocsr()))
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
            #pdb.set_trace()
            if min(OldPrt.cossimrowtorow(PrtMat))>0.99:
                break

        self.KGroupes=[i[0] for i in Grps]
        self.Prototypes=OldPrt

        return [i[0] for i in Grps],OldPrt,PrtList

    def UpdateDict(self,FromWrd=1,FromSsn=1):
        if FromWrd:
            self.RevWrdKey = {key: word for (word, key) in self.WrdKey.items()}
        else:
            self.WrdKey = {key:word for (word,key) in self.RevWrdKey.items()}
        if FromSsn:
            self.RevSsnKey = {key: word for (word, key) in self.SsnKey.items()}
        else:
            self.SsnKey = {key: word for (word,key) in self.RevSsnKey.items()}

    def dumpLil(self,name=None,path=None):
        if not path:
            path=self.pathDumps
        if not name:
            name=self.cur_title
        if not os.path.exists(path+'/'+name):
            os.mkdir(path+'/'+name)
        elif not os.path.isdir(path+'/'+name):
            raise NotADirectoryError
        dirpath=path+'/'+name

        with open(dirpath+'/Epimat.dump','w+b') as f:
            pickle.dump(self.StatsMat, f)

        with open(dirpath+'/MetaData.dump','w+b') as f:
            pickle.dump((self.SriData,self.SsnData,self.EpiData),f)

        with open(dirpath+'/Dicts.dump','w+b') as f:
            pickle.dump((self.SsnKey,self.WrdKey),f)

        if self.KGroupes:
            with open(dirpath+'/K-means.dump','w+b') as f:
                pickle.dump((self.KGroupes, self.Prototypes),f)

        if self.Skipped or self.ReadErr or self.LangErr:
            with open(dirpath+'/Errors.dump','w+b') as f:
                pickle.dump((self.Skipped,self.ReadErr,self.LangErr),f)
        print('Dump success at ',dirpath)

    def loadLil(self,name=None,path=None):
        if not path:
            path=self.pathDumps
        if not name:
            pass
        if not os.path.exists(path+'/'+name) or not os.path.isdir(path+'/'+name):
            raise NotADirectoryError
        dirpath=path+'/'+name

        with open(dirpath+'/EpiMat.dump','r+b') as f:
            self.StatsMat=pickle.load(f)

        with open(dirpath+'/MetaData.dump','r+b') as f:
            self.SriData,self.SsnData,self.EpiData=pickle.load(f)

        with open(dirpath+'/Dicts.dump','r+b') as f:
            self.SsnKey,self.WrdKey=pickle.load(f)

        if os.path.exists(dirpath+'/K-means.dump'):
            with open(dirpath+'/K-means.dump','r+b') as f:
                self.KGroupes,self.Prototypes=pickle.load(f)

        if os.path.exists(dirpath+'/Errors.dump'):
            with open(dirpath+'/Errors.dump','r+b') as f:
                self.Skipped,self.ReadErr,self.LangErr=pickle.load(f)

        print('Load success from ',dirpath)

    def dump(self,name=None,EpiMat=True,WrdKey=True,SsnKey=True,StatsMat=True):
        '''Ecrit self.EpiMat, self.WrdKey, self.SsnKey, self.StatsMat sur le disque sous forme de fichiers .dump.
Le répertoire utilisé est self.pathDumps'''
        if not name:
            if self.StatsMat.shape[0]>2:
                name=str(self.StatsMat.shape[0])
            else:
                name=str(self.StatsMat.shape[0])
        if EpiMat:
            file = open(self.pathDumps + '/EpiMat'+name+'.dump', 'w+b')
            mmwrite(file, self.StatsMat)
            file.close()
        if WrdKey:
            file = open(self.pathDumps + '/WrdKey'+name+'.dump', 'w+b')
            pickle.dump(self.WrdKey, file)
            file.close()
        if SsnKey:
            file = open(self.pathDumps + '/SsnKey'+name+'.dump', 'w+b')
            pickle.dump(self.SsnKey, file)
            file.close()
        if StatsMat:
            file = open(self.pathDumps + '/StatsMat'+name+'.dump', 'w+b')
            pickle.dump(self.StatsMat,file)
            file.close()
        print("Done dumping")

    def load(self,name='100',EpiMat=True, WrdKey=True,SsnKey=True,StatsMat=True):
        '''Charge self.EpiMat, self.WrdKey, self.SsnKey, self.StatsMat depuis le répertoire spécifié par self.pathf.
Les fichiers EpiMat.dump et StatsMat.dump sont au format renvoyé par scipy.io.mmwrite tandis que self.WrdKey et self.SsnKey sont au format utilisé par le protocole par défaut de pickle'''


        if EpiMat:
            file = open(self.pathDumps + '/EpiMat'+name+'.dump', 'r+b')
            self.StatsMat = mmread(file).todok()
            file.close()

        if WrdKey:
            file = open(self.pathDumps + '/WrdKey'+name+'.dump', 'r+b')
            self.WrdKey = pickle.load(file)
            file.close()
        if SsnKey:
            file = open(self.pathDumps + '/SsnKey'+name+'.dump', 'r+b')
            self.SsnKey = pickle.load(file)
            file.close()
        if StatsMat:
            file = open(self.pathDumps + '/StatsMat'+name+'.dump', 'r+b')
            self.StatsMat = pickle.load(file)
            file.close()
        print("Done loading")
        self.UpdateDict()

    def AddEpiToRowLil(self, Text, Row):
        Key = self.WrdKey
        Data=self.SubPat.findall(Text)
        EpiWrds='\n'.join([m[9] for m in Data])
        LstWrd= self.TxtTrt(EpiWrds)

        nbWrds=len(LstWrd)

        if not nbWrds:
            return 0

        for Word in set(LstWrd)-Key.keys():
            Key[Word]=self.StatsMat.shape[1]
            self.StatsMat.resize((self.StatsMat.shape[0], self.StatsMat.shape[1] + 1))

        LstWrd=[Key[i] for i in LstWrd]

        self.StatsMat.AddToRow(Counter(LstWrd), Row)

        return nbWrds

    def AddEpisodeLil(self, SriTitle, numseason, Epi, path=None):
        if not path:
            path=self.pathData

        EpiPath= path + '/' + SriTitle + '/' + numseason + '/' + Epi

        F = open(EpiPath, 'r', encoding="utf8")

        print(SriTitle, '   ', numseason, '  ', Epi)

        try:  # test de l'encoding utf8
            Contents = F.read()

        except UnicodeDecodeError:
            F.close()

            try:  # En cas d'erreur, essai de l'encoding latin-1
                F = open(EpiPath, 'r', encoding="latin-1")
                Contents = F.read()

            except UnicodeDecodeError:  # Si toujours erreur, on passe au suivant en ajoutant cela aux erreurs
                F.close()
                self.ReadErr.append([Epi, numseason, EpiPath])
                print('Erreur de décodage : Epi' + Epi + ' Season ' + numseason + 'Path' + EpiPath)
                return 0

        F.close()

        Row = self.SsnKey[SriTitle]

        res = self.AddEpiToRowLil(Contents,Row)

        if not res:
            return 0

        self.EpiData.append((SriTitle, numseason, Epi, res))

        return res

    def AddSeasonLil(self, SriTitle, numseason, path=None):
        if not path:
            path=self.pathData
        PathSsn= path + '/' + SriTitle + '/' + numseason
        if not os.path.isdir(PathSsn):
            return 0,0

        nbEpi=0
        nbWords=0
        for Epi in sorted(os.listdir(PathSsn)):
            res=self.AddEpisodeLil(SriTitle, numseason, Epi)
            if res:
                nbWords+=res
                nbEpi+=1

        if not nbEpi:
            return 0,0

        self.SsnData.append((SriTitle, numseason, nbEpi,nbWords))

        return nbEpi,nbWords

    def AddSerieLil(self, Title, path=None):

        if not path:
            path=self.pathData
        PathSri = path + '/' + Title

        SriTitle=self.SsnPat.match(PathSri.split('/')[-1]).group(2)
        print('Processing ', SriTitle, 'at', PathSri)
        LstSsn=os.listdir(PathSri + '/')
        LstSsn.sort()

        NbSsn=0
        NbEpi=0

        if sum([len(os.listdir(PathSri+'/'+j)) for j in (i for i in LstSsn if '.txt' not in i)])<=1:
            print("Skipping because of low numbers of episode")
            self.Skipped.append(SriTitle)
            return 0, 0, 0

        self.SsnKey[Title]=self.StatsMat.shape[0]
        self.StatsMat.resize((self.StatsMat.shape[0] + 1, self.StatsMat.shape[1]))

        nbSsn=0
        nbEpi=0
        nbWords=0

        for Season in LstSsn:
            res = self.AddSeasonLil(Title,Season)
            if res[0]:
                nbSsn+=1
                nbEpi+=res[0]
                nbWords+=res[1]
        if not nbSsn:
            return 0, 0, 0
        self.SriData.append((Title,nbSsn,nbEpi,nbWords))

        return nbSsn, nbEpi, nbWords

    def AddSeriesLil(self, Path=None, m=-1, Numbers=()):
        if not Path:
            pathData=self.pathData
        else:
            pathData=Path
        SetSri = {i for i in os.listdir(Path) if os.path.isdir(Path+'/'+i)}

        if m<0:
            finalset=SetSri
        else:
            if m<len(Numbers):
                print('Trop de séries prédéterminées')
                return 0
            if len(Numbers)>0:
                if isinstance(Numbers[0],int):
                    Numbers=[str(i) for i in Numbers]
                    Series={i for i in SetSri if i.split('___')[0] in Numbers}
                    SetSri.difference_update(Series)
                elif isinstance(Numbers[0],str):
                    Series=set(Numbers)
                    SetSri.difference_update(Series)
                else:
                    raise NotImplementedError
            else:
                Series=set()
            Series.update(random.sample(SetSri,m-len(Numbers)))
            finalset=Series
        nbadd=0
        nbwrds=0
        nbssn=0
        nbepi=0
        for Serie,i in zip(finalset,range(1,len(finalset)+1)):
            res=self.AddSerieLil(Serie,path=pathData)
            if res[0]:
                nbadd+=1
                nbssn+=res[0]
                nbepi+=res[1]
                nbwrds+=res[2]
            print(i, 'ème série.')
        print('{:,} mots dans {:,} épisodes dans {:,} saisons dans {:,} séries.'.format(nbwrds,nbepi,nbssn,nbadd))
        print("{} séries ignorées par manque d'épisodes.".format(len(self.Skipped)))


    def AddEpiToRow(self, Text, Row):

        M = self.StatsMat
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

                Key[i] = M.shape[1]
                M.resize((M.shape[0], M.shape[1]+1))

                M[Row, M.shape[1] - 1] = 1

    def AddSeries(self, Path, m=-1, Numbers=()):
        """

        :type Numbers: IntList
        :type m: Int
        :type Path: String
        """
        LstSri = os.listdir(Path)
        LstSri.remove('grab.txt')
        i = 0
        Numbers = [str(m) for m in Numbers]
        Series = [S for S in LstSri if S.split('___')[0] in Numbers]
        if m > 0 and m < len(Series):
            m = 0
        else:
            m -= len(Series)
        while m != 0 and i<len(LstSri):
            if LstSri[i].split('___')[0] in Numbers:
                i+=1
                continue
            Series.append(LstSri[i])
            i += 1
            m -= 1
        nbS=0
        for S in Series:
            nbS+=self.AddSerie(Path + '/' + S)
            print(nbS,' séries ajoutées.')

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
        if NbrToAdd <= 1:
            print('Skipping because of low number of episodes')
            return 0

        NbrSri = self.StatsMat.shape[0]
        self.StatsMat.resize((NbrSri + 1, self.StatsMat.shape[1]))

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


