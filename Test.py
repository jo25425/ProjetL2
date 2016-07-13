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


def go(n=10, N=(1245, 1235, 334, 558, 4, 1439, 358), name=None):  # 1000,53,15,1235]):
    P.enable()
    Test.AddSeries(pathData, m=n, Numbers=N)
    Test.dump(name=name)
    P.disable()


def g():
    Test.load(name='100')
    DFmax = 30
    DFmin = 5
    TF = 1
    DF = 1
    nbr = 7
    Test.InitStats(DFmax, DFmin, TF, DF)
    t = Test.GrpByK(nbr)  # ,[1479, 950, 2552, 1556, 2650, 1054, 631])
    with open(pathDumps + '/Kmeansdata10.txt', 'a') as F:
        print('\n', file=F)
        print([t[0].count([i]) for i in range(nbr)], ' DFmax=', DFmax, ' DFmin=', DFmin, ' TF=', TF, ' DF=', DF, file=F)
        m = t[1]
        l = [[Test.RevWrdKey[m.rows[j][m.data[j].index(list(sorted(m.data[j]).__reversed__())[i])]] for i in range(30)]
             for j in range(nbr)]
        print('\n', file=F)
        p = t[0]
        u = [[Test.RevSsnKey[i] for i in range(len(p)) if p[i] == j] for j in range(nbr)]
        for k in range(nbr):
            print(str(l[k]).encode('utf-8'), file=F)
            print(str(u[k]).encode('utf-8'), file=F)
            print('', file=F)
    print('Done')
    return t


def TestInit(DFmax=70, DFmin=5, TF=True, DF=True, Smax=5000):
    P.enable()
    Test.InitStats(DFmax, DFmin, TF, DF, Smax)
    P.disable()


def TestFunc(f, *args):
    P.enable()
    i = f(*args)
    P.disable()
    return i


def TestTotal(name='Testfont100'):
    P.enable()
    Test.load(name)

    Test.cur_title=name
    if not os.path.isdir(pathDumps+'/'+Test.cur_title):
        os.mkdir(pathDumps+'/'+Test.cur_title)
    GoT_Key = Test.SsnKey['1245___Game_of_Thrones']
    Buffy_Key = Test.SsnKey['334___Buffy_The_Vampire_Slayer']
    Angel_Key = Test.SsnKey['558___Angel']
    PBreak_Key = Test.SsnKey['4___Prison_Break']

    ims = (
        TestG.SerieTFDF(GoT_Key), TestG.SerieTFDF(Buffy_Key), TestG.SerieTFDF(Angel_Key), TestG.SerieTFDF(PBreak_Key))
    for i in ims:
        i.AnnotateWordsatPos((500, 1000000), n=50)
        plt.figure(i.fig.number)
        plt.savefig(pathDumps + '/' + Test.cur_title + '/Série%dAvantClean.png' % i.fig.number)

    TestG.LangRepartition()
    plt.savefig(pathDumps + '/' + Test.cur_title + '/LangRepAvantClean.png')

    TestG.TagsRepartition()
    plt.savefig(pathDumps + '/' + Test.cur_title + '/TagLenFigAvantClean.png')

    Test.MergeDelTags({('NP',): 'NP_ANY', ('SYM', '(', ')', '#', "''", '$', '``', ':', ',', 'FW'): ''})

    Test.InitStats(maxDF=70, minDF=0, TF=True, DF=True, Smax=10000)

    TestG.TagsRepartition()
    plt.savefig(pathDumps + '/' + Test.cur_title + '/TagLenFigApresClean.png')

    GoT_Key = Test.SsnKey['1245___Game_of_Thrones']
    Buffy_Key = Test.SsnKey['334___Buffy_The_Vampire_Slayer']
    Angel_Key = Test.SsnKey['558___Angel']
    PBreak_Key = Test.SsnKey['4___Prison_Break']
    ims = (
        TestG.SerieTFDF(GoT_Key), TestG.SerieTFDF(Buffy_Key), TestG.SerieTFDF(Angel_Key), TestG.SerieTFDF(PBreak_Key))
    for i in ims:
        i.AnnotateWordsatPos((500, 1000000), n=50)
        plt.figure(i.fig.number)
        plt.savefig(pathDumps + '/' + Test.cur_title + '/Série%dApresClean.png' % i.fig.number)

    P.disable()


def TestLen():
    P.enable()
    TestG.TagsRepartition()
    P.disable()
    plt.savefig(pathDumps + '/' + Test.cur_title + '/TagLenFig.png')


def TestTags():
    P.enable()
    Test.UpdateTags()
    P.disable()


def TestLang():
    Test.load(name='TestLiL1000')
    P.enable()
    TestG.LangRepartition()
    P.disable()
    plt.show()


def TestWordsInPrototypes():
    Test.load(name='TestLiL10')
    Test.InitStats(maxDF=70, minDF=2, TF=True, DF=True, Smax=5000)
    Test.GrpByK(3)
    return Test.GetWordsInPrototypes(NbWords=30)


def names(find=False):
    if not os.path.isfile(pathDumps + '/names.dump'):
        find = True
    if find:
        namesrePat = re.compile('(\w+)([(]\d*[)])?\s+.+$', re.MULTILINE)
        with open(pathProj + '/names.txt', 'r') as f:
            nameslist = [i[0] for i in namesrePat.findall(f.read())]
        with open(pathDumps + '/names.dump', 'w+b') as f:
            pickle.dump(nameslist, f)
        return nameslist

    else:
        with open(pathDumps + '/names.dump', 'r+b')as f:
            return pickle.load(f)


def TestCounter1():
    P = cProfile.Profile()
    k = Counter('aaabbb')
    j = Counter('bbbcccc')
    toadd = [j for i in range(100000)]
    P.enable()
    sum(toadd, Counter())
    P.disable()
    return P


def TestCounter2():
    P = cProfile.Profile()
    k = Counter('aaabbb')
    j = Counter('bbbcccc')
    toadd = (j for i in range(100000))
    P.enable()
    k = Counter()
    for i in toadd:
        k += i
    P.disable()
    return P


if __name__ == '__main__':
    P = cProfile.Profile()
    Test = Projet()
    Test.TreeTagger.tag_text('Hello there, what are you doing?')
    TestG = Grapher(Test)
    # t=TestWordsInPrototypes()
    go(100)#, name='Testfont100')
    # Test.load('TestLiL100')
    # i=TestFunc(Test.MergeDelTags,{('NP',):'NP_ANY',('SYM','(',')','#',"''",'$','``',':',',','FW'):''})
    # P1=TestCounter1()
    # P2=TestCounter2()
    # P1.print_stats(sort='cumtime')
    # P2.print_stats(sort='cumtime')
    #TestTotal()
    # Test.InitStats(maxDF=70,minDF=5,Smax=85,DF=False,TF=True)

    # Tags update
    # repr langs
    # auto naming
    # image management
