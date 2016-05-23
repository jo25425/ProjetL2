

import matplotlib.pyplot as plt
from Series import Projet
from My_lil_matrix import My_lil_matrix
import pdb
import matplotlib.mlab as mlab
import numpy as np

class Grapher():
    def __init__(self,Proj=None):
        assert type(Proj)==Projet
        self.Proj=Proj
        self.cmap=plt.get_cmap('Set1')
        self.cur_rep=[]
    def FindBestRows(self,row=None):
        '''
        The goal of the function is to return 2 indexes that best represent the data or the index that best represent the data in conjunction with row if row is provided.
        :param row: index of lines
        :return : 1 value if row is provided, tuple of 2 if row isn't

        '''
        if row:
            L=self.Proj.StatsMat.subgroups([[row]])[0].cossimrowtorow(self.Proj.StatsMat,mode=1)
        else:
            L=self.Proj.StatsMat.cossimrowtorow(self.Proj.StatsMat,mode=1)
        i=np.argmin(L)
        if row:
            return i
        else:
            return i%self.Proj.StatsMat.shape[0],i//self.Proj.StatsMat.shape[0]
    def ComparedToRows(self, n1, n2,GroupBy=None,data=None):
        if not data:
            data=self.Proj.StatsMat

        Mat=data.subgroups([[n1,n2]])[0]
        m=len(data.rows)
        values=Mat.cossimrowtorow(data,mode=1)
        if GroupBy:
            NbGrps=max(GroupBy)+1
            Grps=[list() for i in range(NbGrps)]
            for i in range(len(GroupBy)):
                Grps[GroupBy[i]].append(i)
            colors=[self.cmap(i) for i in np.linspace(0,1,NbGrps)]

            for i in range(NbGrps):
                valx=[values[j] for j in Grps[i]]
                valy=[values[j+m] for j in Grps[i]]
                plt.plot(valx, valy, 'd', color=colors[i])
        else:
            plt.plot(values[:m], values[m:], 'd')
        plt.xlabel('Proximité à {}'.format(self.Proj.RevSsnKey[n1]))
        plt.ylabel('Proximité à {}'.format(self.Proj.RevSsnKey[n2]))
        plt.axis([0, max(values[:m], key=lambda x:0 if x >= 0.99 else x), 0, max(values[m:], key=lambda x:0 if x >= 0.99 else x)])
        plt.show()
    class _TFDFRep():
        def __init__(self,parent,Row=None):
            self.parent=parent
            self.matrix=parent.Proj.StatsMat
            self.fig=plt.figure()
            parent.cur_rep.append(self)
            if Row:
                self.LoadRow(Row)
        def LoadRow(self,Row,Group=None):
            '''

            :param Row: int representing the row of the graphed matrix being represented
            :param Group: either a list of ints or a list of sets
            :return:
            '''

            df=self.matrix.non_zeros(0)
            self.data=[(i[0], df[i[0]], i[1]) for i in zip(self.matrix.rows[Row], self.matrix.data[Row])]
            plt.figure(self.fig.number)
            if not Group:
                plt.plot([i[1] for i in self.data], [i[2] for i in self.data], 'd', color=self.parent.cmap(0))
            else:
                if isinstance(Group[0],int):
                    Groups=[]
                    for i in range(len(Group)):
                        try:
                            Groups[Group[i]].add(i)
                        except IndexError:
                            j=len(Groups)
                            while j<=Group[i]:
                                Groups.append(set())
                else:
                    Groups=Group
                Datatoplot=set(self.data)
                NbGrp=len(Groups)
                colors=[self.parent.cmap(i) for i in np.linspace(0,1,NbGrp+1)]
                for SubGrp,color in zip(Groups,colors):
                    Datatoplot.difference_update(SubGrp)
                    plt.plot([i[1] for i in SubGrp], [i[2] for i in SubGrp], 'd', color=color)
                if len(Datatoplot)>0:
                    plt.plot([i[1] for i in Datatoplot], [i[2] for i in Datatoplot], 'd', color=colors[NbGrp])

            plt.xlabel('DF')
            plt.ylabel('TF')
            self.fig.suptitle(self.parent.Proj.RevSsnKey[Row])

            plt.show()
        def FindWordsatPos(self,xycoord,xymax=None,n=10):
            '''

            :param xycoord: si xymax n'est pas fourni, coordonnées du point dont les n plus proches voisins seront trouvés, sinon, coordonnées minimum de la fenêtre à annoter
            :param xymax: si fourni, coordonnées maximum de la fenêtre à annoter
            :param n: nombre de plus proches voisins à chercher
            :return liste de tuple (nomsérie,xpos,ypos) de résultats:
            '''
            xmin=min(self.data,key=lambda x:x[1])[1]
            xmax=max(self.data,key=lambda x:x[1])[1]
            ymin=min(self.data,key=lambda x:x[2])[2]
            ymax=max(self.data,key=lambda x:x[2])[2]
            def dist(point):
                return ((xycoord[1]-point[2])/(ymax-ymin))**2+((xycoord[0]-point[1])/(xmax-xmin))**2
            if len(self.data)<n:
                return [(self.parent.Proj.RevWrdKey[i[0]],i[1],i[2]) for i in sorted(self.data,key=dist)]
            if not xymax:
                return [(self.parent.Proj.RevWrdKey[i[0]],i[1],i[2]) for i in sorted(self.data,key=dist)[:n]]
            return [(self.parent.Proj.RevWrdKey[i[0]],i[1],i[2]) for i in self.data if xymax[1] >= i[2] >=
                    xycoord[1] and xymax[0] >= i[1] >= xycoord[0]]
        def AnnotateWordsatPos(self,xycoord,xymax=None,n=10):
            plt.figure(self.fig.number)
            data=self.FindWordsatPos(xycoord,xymax,n)
            for i in data:
                plt.annotate(i[0], xy=(i[1], i[2]))
    def SerieTFDF(self, Row, GroupBy=None, data=None):
        '''
        Renvoie un objet permettant de représenter les mots présents dans la série au rang Row.
        :param Row:
        :param GroupBy:
        :param data:
        :return:
        '''
        i=self._TFDFRep(self)
        i.LoadRow(Row,GroupBy)
        return i

    def WordsTF(self,Word):
        '''
        Renvoie une représentation des mots indiqués par Word en fonction des séries où ils apparraissent
        :param Word : liste de mots ou d'entiers
        :return:
        '''
        fig=plt.figure()
        self.cur_rep.append(fig)
        if isinstance(Word[0],str):
            Rows=[self.Proj.WrdKey[i] for i in Word]
            Words=Word
        elif isinstance(Word[0],int):
            Words=[self.Proj.RevWrdKey[i] for i in Word]
            Rows=Word
        else:
            raise NotImplementedError
        Transpose = self.Proj.StatsMat.transpose().subgroups([Rows])[0]
        colors=[self.cmap(i) for i in np.linspace(0,1,len(Rows))]
        plt.xticks(range(Transpose.shape[1]),[i[1] for i in sorted(self.Proj.RevSsnKey.items(),key=lambda x:x[0])])
        for i in range(len(Rows)):
            plt.plot(range(Transpose.shape[1]),[Transpose[i,j] for j in range(Transpose.shape[1])],color=colors[i],label=Words[i])
        plt.legend()
        plt.show()


    def PCA(self,Kmeans=True,k=10):
        assert self.Proj.StatsMat.shape[0]>self.Proj.StatsMat.shape[1]
        if Kmeans:
            self.Proj.GrpByK(k)
        colors=[self.cmap(i) for i in np.linspace(0,1,max(self.Proj.GrpK))]
        p=mlab.PCA(self.Proj.StatsMat.tocsr().toarray())
        l=p.project(p.a,p.fracs[2])
        plt.scatter([i[0] for i in l], [i[1] for i in l], c=[colors[i] for i in self.Proj.GrpK])
        plt.show()



if __name__=='__main__':
    Test=Projet()
    TestG=Grapher(Test)
