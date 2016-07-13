from collections import Counter, defaultdict
from math import sqrt

from scipy.sparse import isspmatrix_lil, isspmatrix_csr, csr_matrix, lil_matrix

class Defaultdictwithkey(defaultdict):
    def __missing__(self,key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

class My_lil_matrix():
    def __init__(self,arg):
        """
        LInked List type of sparse matrix made to replace scipy.sparse.lil_matrix
        Uses the scipy.sparse implementation of sparse matrix product.

        Most operations and functions are only implemented for rows, the matrix should be transposed if there is a need
        to do those operations on columns

        The initialisation arg can be a tuple or list of len 2 to describe the shape of the matrix or another matrix to be used as initial values
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

    def __getitem__(self,item):
        '''
        Return the value of the item at the intersection the item[0]th row and the item[1]th column
        :param item: tuple of len 2
        :return:
        '''
        if type(item)==tuple and len(item)==2:
            if item[1] in self.rows[item[0]]:
                return self.data[item[0]][self.rows[item[0]].index(item[1])]
            else:
                return 0
        else:
            raise NotImplementedError

    def non_zeros(self,axis=2):
        '''
        0 returns number of nonzeros for each column, 1 returns number of nonzeros for each row, 2 returns number of nonzeros for the whole matrix
        :param axis: 0,1 or 2
        :return number of non zeros along given axis:
        '''
        if axis>=1:
            res=[len(i) for i in self.rows]
            if axis==2:
                return sum(res)
            return res
        #case axis=0
        res=[0]*self.shape[1]
        for i in self.rows:
            for j in i:
                res[j]+=1
        return res

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

    def addtorow(self, newData, Row):
        """
        Adds newData to the Rowth row of the matrix.
        newData can be a collections.Counter object
        :param newData: Counter
        :param Row: int
        :return:
        """
        if isinstance(newData, Counter):
            newData = Counter(dict(zip(self.rows[Row],self.data[Row]))) + newData
            self.rows[Row],self.data[Row]=[list(i) for i in zip(*newData.items())]

    def addrows(self, rows, res=None):
        """
        rows must be a list of rows to be added together.
        res is the number of the row receiving the sum as its new value, it defaults to rows[0]
        :param rows: list of ints
        :param res: int
        :return:
        """
        if not res:
            res=rows[0]
        CList=[Counter(dict(zip(self.rows[i],self.data[i]))) for i in rows]
        C=Counter()
        for i in CList:
            C+=i
        self.rows[res],self.data[res]=[list(i) for i in zip(*C.items())]
        return C

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

    def removerowsind(self, Rows, Index):

        Rows.sort(reverse=True)
        NewInd=Index.copy()
        N=self.shape[0]
        for i in range(len(Rows)):
            self.rows[Rows[i]]=self.rows[N-i-1]
            self.data[Rows[i]]=self.data[N-i-1]
            NewInd[Rows[i]]=NewInd[N-i-1]
        for i in range(len(Rows)):
            NewInd.pop(N-i-1)
        self.resize((self.shape[0]-len(Rows),self.shape[1]))
        return NewInd

    def removerowsind2(self, Rows):
        Rows.sort(reverse=True)
        Ind=Defaultdictwithkey(lambda x:x)
        N=self.shape[0]
        for i in range(len(Rows)):
            self.rows[Rows[i]]=self.rows[N-i-1]
            self.data[Rows[i]]=self.data[N-i-1]
            Ind[Rows[i]]=Ind[N-i-1]
        for i in range(len(Rows)):
            del Ind[N-i-1]
        self.resize((self.shape[0]-len(Rows),self.shape[1]))
        return Ind

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

    def transpose(self,copy=True):
        NewMat=My_lil_matrix((self.shape[1],self.shape[0]))
        for i in range(self.shape[0]):
            for j in range(len(self.rows[i])):
                NewMat.rows[self.rows[i][j]].append(i)
                NewMat.data[self.rows[i][j]].append(self.data[i][j])
        if not copy:
            self.rows=NewMat.rows
            self.data=NewMat.data
            self.shape=NewMat.shape
        return NewMat if copy else self

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
        elif axis==2:
            for i in range(Mat.shape[0]):
                Mat.data[i]=f(Mat.data[i])
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

    def cossimrowtorow(self, Mat2,mode=0):
        '''
        Mode 0: Calcul ligne à ligne du cossim
        Mode 1: le cossim de la ième ligne de mat 1 avec la jème ligne de mat 2 est en [i*m+j]
        :param Mat2:
        :param mode:
        :return:
        '''
        if mode==0:
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
        elif mode==1:
            n=self.shape[0]*Mat2.shape[0]
            m=Mat2.shape[0]
            Res=[0]*n
            def tmp(list):
                res=0
                for i in list:
                    res+=i*i
                return sqrt(res)
            NrmMat1=self.apply(tmp,True,1)
            NrmMat2=Mat2.apply(tmp,True,1)
            for i in range(n):
                Res[i]=sum([a*b for a,b in zip(self.data[i//m],Mat2.data[i%m])])/(NrmMat1.data[i//m][0]*NrmMat2.data[i%m][0])
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