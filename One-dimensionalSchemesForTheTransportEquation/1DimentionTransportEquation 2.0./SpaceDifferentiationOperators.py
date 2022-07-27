#-MODULES-AND-LIBRARYS-
import numpy              as np
import scipy

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma
from pylab                import *
#----------------------------------------------------------------------------------------#
#-------------------------------------BEGIN----------------------------------------------#
#----------------------------------------------------------------------------------------#
#-MAIN-CLASS-
class SpaceDiff:
    
    #main methods
    def __init__(self, h, v, size):
        self.h        = h                         #шаг дискретизации по пространству
        self.v        = v                         #скорость
        self.matrix   = self.MatrixGenerate(size) #матрица оператора
        
    def diff(self, f):
        #диффереенцирование
        return self.matrix@f

    def MatrixGenerate(self, size):
        #матрица опертора
        return np.eye(size)
    
#-CHILD-CLASES-
#1. periodic operators
class Left1Periodic(SpaceDiff):
    
    def MatrixGenerate(self, size):
        data  = np.array([np.zeros(size)-1, np.zeros(size)+1])
        diags = np.array([-1, 0])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =  -1 #Внесение переодичности методом виртуальных точек
        return self.v/self.h*D

class Left2Periodic(SpaceDiff):
    
    def MatrixGenerate(self, size):
        data  = np.array([np.ones(size), -4*np.ones(size), 3*np.ones(size)])
        diags = np.array([-2, -1, 0])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =  -4 #Внесение переодичности методом виртуальных точек
        D[0,-2] =   1 #Внесение переодичности методом виртуальных точек
        D[1,-1] =   1 #Внесение переодичности методом виртуальных точек
        return self.v/(2*self.h)*D

class Left21Periodic(SpaceDiff):
    
    def MatrixGenerate(self, size):
        data  = np.array([1/6*np.ones(size), -1*np.ones(size), 1/2*np.ones(size), 1/3*np.ones(size)])
        diags = np.array([-2, -1, 0, 1])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =  -1   #Внесение переодичности методом виртуальных точек
        D[0,-2] =   1/6 #Внесение переодичности методом виртуальных точек
        D[1,-1] =   1/6 #Внесение переодичности методом виртуальных точек
        D[-1,0] =   1/3 #Внесение переодичности методом виртуальных точек
        return self.v/(self.h)*D

class Left31Periodic(SpaceDiff):
    
    def MatrixGenerate(self, size):
        data  = np.array([-1/12*np.ones(size), 1/2*np.ones(size), -3/2*np.ones(size), 5/6*np.ones(size), 1/4*np.ones(size)])
        diags = np.array([-3, -2, -1, 0, 1])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =  -3/2   #Внесение переодичности методом виртуальных точек
        D[0,-2] =   1/2   #Внесение переодичности методом виртуальных точек
        D[0,-3] =  -1/12 #Внесение переодичности методом виртуальных точек
        D[1,-1] =   1/2   #Внесение переодичности методом виртуальных точек
        D[1,-2] =  -1/12 #Внесение переодичности методом виртуальных точек
        D[2,-1] =  -1/12 #Внесение переодичности методом виртуальных точек
        D[-1,0] =   1/4 #Внесение переодичности методом виртуальных точек
        return self.v/(self.h)*D

class Left32Periodic(SpaceDiff):
    
    def MatrixGenerate(self, size):
        data  = np.array([-1/30*np.ones(size), 1/4*np.ones(size), -1*np.ones(size), 3/10*np.ones(size), 1/2*np.ones(size), -1/20*np.ones(size)])
        diags = np.array([-3, -2, -1, 0, 1, 2])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] = -1    #Внесение переодичности методом виртуальных точек
        D[0,-2] =  1/4   #Внесение переодичности методом виртуальных точек
        D[0,-3] = -1/30 #Внесение переодичности методом виртуальных точек
        D[1,-1] =  1/4   #Внесение переодичности методом виртуальных точек
        D[1,-2] = -1/30 #Внесение переодичности методом виртуальных точек
        D[2,-1] = -1/30 #Внесение переодичности методом виртуальных точек
        D[-1,0] =  1/2 #Внесение переодичности методом виртуальных точек
        D[-2,0] = -1/20 #Внесение переодичности методом виртуальных точек
        D[-1,1] = -1/20 #Внесение переодичности методом виртуальных точек
        return self.v/(self.h)*D
        
class Center2Periodic(SpaceDiff):
    
    def MatrixGenerate(self, size):
        data  = 1/2*np.array([np.zeros(size)-1, np.zeros(size)+1])
        diags = np.array([-1, 1])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =   -1/2 #Внесение переодичности методом виртуальных точек
        D[-1,0] =    1/2 #Внесение переодичности методом виртуальных точек
        return self.v/self.h*D
        
class Center4Periodic(SpaceDiff):

    def MatrixGenerate(self, size):
        data  = 1/12*np.array([(np.zeros(size)+1), 8*(np.zeros(size)-1), 8*(np.zeros(size)+1), (np.zeros(size)-1)])
        diags = np.array([-2, -1, 1, 2])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-2] =   1/12 #Внесение переодичности методом виртуальных точек
        D[0,-1] =  -8/12 #Внесение переодичности методом виртуальных точек
        D[1,-1] =   1/12 #Внесение переодичности методом виртуальных точек
        D[-2,0] =  -1/12 #Внесение переодичности методом виртуальных точек
        D[-1,0] =   8/12 #Внесение переодичности методом виртуальных точек
        D[-1,1] =  -1/12 #Внесение переодичности методом виртуальных точек
        return self.v/self.h*D

#2. SBP
class SBP21(SpaceDiff):
    def H(self,size):
        data  = np.array([np.zeros(size)+1])
        diags = np.array([0])
        H    = sparse.spdiags(data, diags, size, size).toarray()
        H[ 0, 0] = 1/2
        H[-1,-1] = 1/2
        return H*self.h
    
    def Q(self,size):
        data  = np.array([-1/2*(np.ones(size)), 1/2*(np.ones(size))])
        diags = np.array([-1, 1])
        Q  = sparse.spdiags(data, diags, size, size).toarray()
        Q[ 0] = np.zeros(size)
        Q[-1] = np.zeros(size)
        Q[ 0, 0] = -1/2
        Q[ 0, 1] =  1/2
        Q[-1,-1] =  1/2
        Q[-1,-2] = -1/2
        return Q

    def MatrixGenerate(self, size):
        return  np.linalg.inv(self.H(size)).dot(self.Q(size))

class SBP42(SpaceDiff):

    def H(self,size):
        data  = np.array([np.ones(size)])
        diags = np.array([0])
        H    = np.array(sparse.spdiags(data, diags, size, size).toarray(), dtype=double)
        H[ 0,0] = 17/48
        H[ 1,1] = 59/48
        H[ 2,2] = 43/48
        H[ 3,3] = 49/48
        H[-1,-1] = H[0,0]
        H[-2,-2] = H[1,1]
        H[-3,-3] = H[2,2]
        H[-4,-4] = H[3,3]
        return H*self.h
    
    def Q(self,size):
        data  = np.array([1/12*(np.zeros(size)+1), 2/3*(np.zeros(size)-1), 2/3*(np.zeros(size)+1), 1/12*(np.zeros(size)-1)])
        diags = np.array([-2, -1, 1, 2])
        Q  = np.array(sparse.spdiags(data, diags, size, size).toarray(), dtype=double)

        Q[0,0] = -1/2
        Q[0,1] =  59/96
        Q[0,2] = -1/12
        Q[0,3] = -1/32
        Q[1,2] =  59/96
        Q[1,3] =  0
        Q[2,3] =  59/96
        Q[1,0] = -Q[0,1]
        Q[2,0] = -Q[0,2]
        Q[3,0] = -Q[0,3]
        Q[2,1] = -Q[1,2]
        Q[3,1] = -Q[1,3]
        Q[3,2] = -Q[2,3]
        Q[-1,-1] = -Q[0,0]
        Q[-1,-2] = -Q[0,1]
        Q[-1,-3] = -Q[0,2]
        Q[-1,-4] = -Q[0,3]
        Q[-2,-3] = -Q[1,2]
        Q[-2,-4] = -Q[1,3]
        Q[-3,-4] = -Q[2,3]
        Q[-2,-1] = -Q[-1,-2]
        Q[-3,-1] = -Q[-1,-3]
        Q[-4,-1] = -Q[-1,-4]
        Q[-3,-2] = -Q[-2,-3]
        Q[-4,-2] = -Q[-2,-4]
        Q[-4,-3] = -Q[-3,-4]
        return Q

    def MatrixGenerate(self, size):
        return np.linalg.inv(self.H(size))@self.Q(size)

#3. SBP-projaction methods
class SBP21PROJ(SBP21):

    def P(self, size):
        A = np.zeros(size)+1
        data  = np.array([A])
        diags = np.array([0])
        P     = sparse.spdiags(data, diags, size, size).toarray()
        P[ 0, 0] = 1/2
        P[ 0,-1] = 1/2
        P[-1,-1] = 1/2
        P[-1, 0] = 1/2
        return P

    def MatrixGenerate(self, size):
        return self.v*(self.P(size)@np.linalg.inv(self.H(size))@self.Q(size))

class SBP42PROJ(SBP42):

    def P(self, size):
        A = np.zeros(size)+1
        data  = np.array([A])
        diags = np.array([0])
        P     = sparse.spdiags(data, diags, size, size).toarray()
        P[ 0, 0] = 1/2
        P[ 0,-1] = 1/2
        P[-1,-1] = 1/2
        P[-1, 0] = 1/2
        return P

    def MatrixGenerate(self, size):
        return self.v*(self.P(size)@np.linalg.inv(self.H(size))@self.Q(size))

#4. SBP-SAT methods
class SBP21SAT(SBP21):

    def SAT(self, f):
        #SAT-добавки в оператор
        e0 = np.zeros(f.size)
        eN = np.zeros(f.size)
        e0[ 0] = 1
        eN[-1] = 1
        return 1/2*(f[0]-f[-1])*(e0+eN)

    def diff(self, f):
        return self.v*(self.matrix@f + np.linalg.solve(self.H(f.size), self.SAT(f)))

class SBP42SAT(SBP42):

    def SAT(self, f):
        #SAT-добавки в оператор
        e0 = np.zeros(f.size)
        eN = np.zeros(f.size)
        e0[ 0] = 1
        eN[-1] = 1
        return 1/2*(f[0]-f[-1])*(e0+eN)

    def diff(self, f):
        return self.v*(self.matrix@f + np.linalg.solve(self.H(f.size),self.SAT(f)))
