#-MODULES-AND-LIBRARYS-
import numpy              as np
import scipy

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma
from pylab                import *

#-MAIN-CLASS-
class TimeDiff:
    
    def __init__(self, tau, v, time_operator): 
        self.tau            = tau
        self.v              = v
        self.time_operator  = time_operator
        
    def make_step(self, f):
        return self.time_operator@f[j,:]
        
    def diff(self, f):
        for j in range(f[:,0].size-1):
            f[j+1,:] = make_step(f[j,:])
    
    
class Euler(TimeDiff):
    def __init__(self, tau, v, space_operarot): 
        self.tau            = tau
        self.v              = v
        self.time_operator = np.eye(space_operator[0,:].size) - tau*space_operator
        
class RK4(TimeDiff):
    def __init__(self, tau, v, space_operator): 
        self.tau            = tau
        self.v              = v
        
        D1 = space_operator
        D2 = D1@D1
        D3 = D1@D1@D1
        D4 = D1@D1@D1@D1
        self.time_operator = np.eye(space_operator[0,:].size) - tau*D1 - tau**2/2*D2 - tau**3/6*D3 - tau**4/24*D4