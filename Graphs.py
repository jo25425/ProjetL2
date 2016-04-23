

import matplotlib.pyplot as pyplot
from Series import Projet
from My_lil_matrix import My_lil_matrix
import pdb
import matplotlib.mlab as mlab
import numpy as np

class Grapher():
    def __init__(self,Proj=None):
        assert type(Proj)==Projet
        self.Proj=Proj
        self.cmap=pyplot.get_cmap('Set1')
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
                pyplot.plot(valx,valy,'d',color=colors[i])
        else:
            pyplot.plot(values[:m],values[m:],'d')
        pyplot.xlabel('Proximité à {}'.format(self.Proj.RevSsnKey[n1]))
        pyplot.ylabel('Proximité à {}'.format(self.Proj.RevSsnKey[n2]))
        pyplot.axis([0,max(values[:m],key=lambda x:0 if x>=0.99 else x),0,max(values[m:],key=lambda x:0 if x>=0.99 else x)])
        pyplot.show()
    def WordsTFDF(self,Serie,GroupBy=None,data=None):
        if not data:
            data=self.Proj.StatsMat

        assert len(data.data[Serie])==len(data.rows[Serie])
        datat=data.transpose()
        if GroupBy:
            NbGrps=max(GroupBy)+1
            Grps=[list() for i in range(NbGrps)]
            for i in range(len(GroupBy)):
                Grps[GroupBy[i]].append(i)
            colors=[self.cmap(i) for i in np.linspace(0,1,NbGrps)]
        else:
            NbGrps=1
            Grps=[list(range(data.shape[1]))]
            colors=[self.cmap(0)]
        if type(Serie)==str:
            Serie=self.Proj.SsnKey[Serie]
        for i in range(NbGrps):
            valx=[len(datat.data[j]) for j in Grps[i]]
            valy=[data[Serie,j] for j in Grps[i]]
            pyplot.plot(valx,valy,'d',color=colors[i])
        pyplot.xlabel('DF')
        pyplot.ylabel('TF dans {}'.format(self.Proj.RevSsnKey[Serie]))
        pyplot.show()




if __name__=='__main__':
    Test=Projet()
    Test.load()
    #Test.InitStats(70,5,1,1)
    TestG=Grapher(Test)
    #i=TestG.FindBestRows()
    #G=Test.GrpByK(2)
    #p=TestG.ComparedToRows(Test.StatsMat.shape[0],Test.StatsMat.shape[0]+1,GroupBy=G[0]+[2,3],data=G[1].combine([Test.StatsMat]))
    #TestG.WordsTFDF(Serie=63)