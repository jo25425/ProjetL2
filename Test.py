from Series import *
from My_lil_matrix import *
from Graphs import *
import matplotlib.mlab as mlab
from sklearn.decomposition import PCA


def go(n=10, N=[1245, 1235, 334, 558, 4], name=None):  # 1000,53,15,1235]):
    P.enable()
    Test.AddSeries(pathData, m=n, Numbers=N)
    Test.dump(name=name)
    P.disable()

def g():
    Test.load(name='100')
    DFmax=30
    DFmin=5
    TF=1
    DF=1
    nbr=7
    Test.InitStats(DFmax,DFmin,TF,DF)
    t=Test.GrpByK(nbr)#,[1479, 950, 2552, 1556, 2650, 1054, 631])
    with open(pathDumps+'/Kmeansdata10.txt','a') as F:
        print('\n',file=F)
        print([t[0].count([i]) for i in range(nbr)],' DFmax=',DFmax,' DFmin=',DFmin,' TF=',TF,' DF=',DF,file=F)
        m=t[1]
        l=[[Test.RevWrdKey[m.rows[j][m.data[j].index(list(sorted(m.data[j]).__reversed__())[i])]] for i in range(30)] for j in range(nbr)]
        print('\n',file=F)
        p=t[0]
        u=[[Test.RevSsnKey[i] for i in range(len(p)) if p[i]==j] for j in range(nbr)]
        for k in range(nbr):
            print(str(l[k]).encode('utf-8'),file=F)
            print(str(u[k]).encode('utf-8'),file=F)
            print('',file=F)
    print('Done')
    return t

if __name__=='__main__':
    P=cProfile.Profile()
    Test=Projet()