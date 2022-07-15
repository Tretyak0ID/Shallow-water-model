#-MODULES-AND-LIBRARYS-
import numpy              as np
import scipy

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma
from pylab                import *

#-MAIN-CLASS-
class SpaceDiff:
    
    #main methods
    def __init__(self, h, v):
        self.h        = h
        self.v        = v
        self.operator = 0
        
    def diff(self, f):
        return self.operator@f
    
    #operatops type
    def LeftPeriodic(self,size):
        A = np.zeros(size)+1
        B = np.zeros(size)-1
        data  = np.array([B, A])
        diags = np.array([-1, 0])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =  -1 #Внесение переодичности методом виртуальных точек
        self.operator = self.v/self.h*D
    
    def Center2Periodic(self,size):
        A = np.zeros(size)+1
        B = np.zeros(size)-1
        data  = 1/2*np.array([B, A])
        diags = np.array([-1, 1])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =   -1/2 #Внесение переодичности методом виртуальных точек
        D[-1,0] =    1/2 #Внесение переодичности методом виртуальных точек
        self.operator = self.v/self.h*D
    
    def Center4Periodic(self,size):
        A = np.zeros(size)+1
        B = np.zeros(size)-1
        data  = 1/12*np.array([A, 8*B, 8*A, B])
        diags = np.array([-2, -1, 1, 2])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-2] =   1/12 #Внесение переодичности методом виртуальных точек
        D[0,-1] =  -8/12 #Внесение переодичности методом виртуальных точек
        D[1,-1] =   1/12 #Внесение переодичности методом виртуальных точек
        D[-2,0] =  -1/12 #Внесение переодичности методом виртуальных точек
        D[-1,0] =   8/12 #Внесение переодичности методом виртуальных точек
        D[-1,1] =  -1/12 #Внесение переодичности методом виртуальных точек
        self.operator = self.v/self.h*D
        
    def H21(self,size):
        A = np.zeros(size)+1
        data  = np.array([A])
        diags = np.array([0])
        H    = sparse.spdiags(data, diags, size, size).toarray()
        H[ 0, 0] = 1/2
        H[-1,-1] = 1/2
        return H
    
    def Q21(self,size):
        A = np.zeros(size)+1
        B = np.zeros(size)-1
        data  = np.array([self.v/(2*self.h)*B, self.v/(2*self.h)*A])
        diags = np.array([-1, 1])
        Q  = sparse.spdiags(data, diags, f.size, f.size).toarray()
        Q[ 0] = zeros(f.size)
        Q[-1] = zeros(f.size)
        Q[ 0, 0] = -self.v/(2*self.h)
        Q[ 0, 1] =  self.v/(2*self.h)
        Q[-1,-1] =  self.v/(2*self.h)
        Q[-1,-2] = -self.v/(2*self.h)
        return Q
    
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
    
    def H42(self,size):
        A = np.zeros(size)+1
        data  = np.array([A])
        diags = np.array([0])
        H    = sparse.spdiags(data, diags, f.size, f.size).toarray()
        H[ 0,0] = 17/48
        H[ 1,1] = 59/48
        H[ 2,2] = 43/48
        H[ 3,3] = 49/48
        H[-1,-1] = H[0,0]
        H[-2,-2] = H[1,1]
        H[-3,-3] = H[2,2]
        H[-4,-4] = H[3.3]
        return H
    
    def Q42(self,size):
        A = np.zeros(size)+1
        B = np.zeros(size)-1
        data  = np.array([self.v/(12*self.h)*A, 8*self.v/(12*self.h)*B, 8*self.v/(12*self.h)*A, self.v/(12*self.h)*B])
        diags = np.array([-2, -1, 1, 2])
        Q_42  = sparse.spdiags(data, diags, f.size, f.size).toarray()
        Q_42[ 0] = zeros(f.size)
        Q_42[ 1] = zeros(f.size
        Q_42[0,0] = -1/(2*self.h)
        Q_42[0,1] =  59/(96*self.h)
        Q_42[0,2] = -1/(12*self.h)
        Q_42[0,3] = -1/(32*self.h)
        Q_42[1,2] =  59/(96*self.h)
        Q_42[1,3] =  0
        Q_42[2,3] =  59/(96*self.h)
        Q_42[1,0] = -Q_42[0,1]
        Q_42[2,0] = -Q_42[0,2]
        Q_42[3,0] = -Q_42[0,3]
        Q_42[2,1] = -Q_42[1,2]
        Q_42[3,1] = -Q_42[1,3]
        Q_42[3,2] = -Q_42[2,3]
        Q_42[-1,-1] = -Q_42[0,0]
        Q_42[-1,-2] = -Q_42[0,1]
        Q_42[-1,-3] = -Q_42[0,2]
        Q_42[-1,-4] = -Q_42[0,3]
        Q_42[-2,-3] = -Q_42[1,2]
        Q_42[-2,-4] = -Q_42[1,3]
        Q_42[-3,-4] = -Q_42[2,3]
        Q_42[-2,-1] = -Q_42[-1,-2]
        Q_42[-3,-1] = -Q_42[-1,-3]
        Q_42[-4,-1] = -Q_42[-1,-4]
        Q_42[-3,-2] = -Q_42[-2,-3]
        Q_42[-4,-2] = -Q_42[-2,-4]
        Q_42[-4,-3] = -Q_42[-3,-4]
        return Q_42
    
    def SAT0():
        return 
    
    def SATN():
        return
    
    def SBP21PROJ(self,size):
        self.operator = P(self,size)@np.linalg.inv(H21(self,size))@Q21(self,size)
        
    def SBP42PROJ(self,size):
        self,operator =  P(self,size)@np.linalg.inv(H42(self,size))@Q42(self,size)
    
    def SBP21SAT(self,size):
        self.operator = np.linalg.inv(H21(self,size))@Q21(self,size)
        
    def SBP42SAT(self,size):
        self.operator = np.linalg.inv(H42(self,size))@Q42(self,size)