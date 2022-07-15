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
    
#-CHILD-CLASES-
class LeftPeriodic(SpaceDiff):
    def __init__(self, h, v, size):
        self.h        = h
        self.v        = v

        data  = np.array([np.zeros(size)-1, np.zeros(size)+1])
        diags = np.array([-1, 0])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =  -1 #Внесение переодичности методом виртуальных точек
        self.operator = self.v/self.h*D
        
class Center2Periodic(SpaceDiff):
    def __init__(self, h, v, size):
        self.h        = h
        self.v        = v

        data  = 1/2*np.array([np.zeros(size)-1, np.zeros(size)+1])
        diags = np.array([-1, 1])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-1] =   -1/2 #Внесение переодичности методом виртуальных точек
        D[-1,0] =    1/2 #Внесение переодичности методом виртуальных точек
        self.operator = self.v/self.h*D
        
class Center4Periodic(SpaceDiff):
    def __init__(self, h, v, size):
        self.h        = h
        self.v        = v

        data  = 1/12*np.array([(np.zeros(size)+1), 8*(np.zeros(size)-1), 8*(np.zeros(size)+1), (np.zeros(size)-1)])
        diags = np.array([-2, -1, 1, 2])
        D       =   sparse.spdiags(data, diags, size, size).toarray()
        D[0,-2] =   1/12 #Внесение переодичности методом виртуальных точек
        D[0,-1] =  -8/12 #Внесение переодичности методом виртуальных точек
        D[1,-1] =   1/12 #Внесение переодичности методом виртуальных точек
        D[-2,0] =  -1/12 #Внесение переодичности методом виртуальных точек
        D[-1,0] =   8/12 #Внесение переодичности методом виртуальных точек
        D[-1,1] =  -1/12 #Внесение переодичности методом виртуальных точек
        self.operator = self.v/self.h*D