from Series import *
from My_lil_matrix import *
from Graphs import *
import matplotlib.mlab as mlab
from sklearn.decomposition import PCA
import cProfile
import pdb


if os.environ['COMPUTERNAME'] == 'TIE':
    pathProj = 'C:/Users/Vivien/PycharmProjects/ProjetL2'
    pathDumps = 'C:/Users/Vivien/PycharmProjects/ProjetL2/dumps'
    pathData = 'E:/Documents/Programmes/addic7ed'
elif os.environ['COMPUTERNAME'] == 'Janice':
    pathDumps = 'C:\projet l2'
    pathData = 'C:\tmp\addic7ed\addic7ed'
else:
    pathDumps = '/tmp'
    pathData = '/tmp/addic7ed'


def go(n=10, N=(1245, 1235, 334, 558, 4, 1439), name=None):  # 1000,53,15,1235]):
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

def TestInit(DFmax=70,DFmin=5,TF=True,DF=True,Smax=5000):
    P.enable()
    Test.InitStats(DFmax,DFmin,TF,DF,Smax)
    P.disable()

def TestFunc(f,*args):
    P.enable()
    i=f(*args)
    P.disable()
    return i

def names(find=False):
    if not os.path.isfile(pathDumps+'/names.dump'):
        find=True
    if find:
        namesrePat=re.compile('(\w+)([(]\d*[)])?\s+.+$',re.MULTILINE)
        with open(pathProj + '/names.txt', 'r') as f:
            nameslist=[i[0] for i in namesrePat.findall(f.read())]
        with open(pathDumps+'/names.dump','w+b') as f:
            pickle.dump(nameslist,f)
        return nameslist

    else:
        with open(pathDumps+'/names.dump','r+b')as f:
            return pickle.load(f)


if __name__=='__main__':
    P=cProfile.Profile()
    Test=Projet()
    Test.TreeTagger.tag_text('Hello there, what are you doing?')
    #go(100,name='TestLiL100')
    Test.load('TestLiL100')
    i=TestFunc(Test.FlagTags,[('DT','NP'),('NN','VB')])
    TestG=Grapher(Test)
    #Test.InitStats(maxDF=70,minDF=5,Smax=85,DF=False,TF=True)
