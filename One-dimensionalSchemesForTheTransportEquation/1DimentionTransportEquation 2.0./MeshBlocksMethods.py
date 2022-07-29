#-MODULES-AND-LIBRARYS-
import numpy              as np
import scipy
import math
import SpaceDifferentiationOperators as SD 
import TimeDifferentiationOperators as TD

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma, exp
from pylab                import *
#----------------------------------------------------------------------------------------#
#-------------------------------------BEGIN----------------------------------------------#
#----------------------------------------------------------------------------------------#

#-BLOCK-STRUCTURE-
class Block1:

    def __init__(self, xmin, xmax, cx, ct):
        self.mesh = np.arange(0 , cx+1, dtype = double)*(xmax-xmin)/cx + xmin
        self.PHI  = np.zeros((ct+1, cx+1))
    
    def DefineIC(self, IC):
        self.PHI[0] = IC

    def DefineICstandart(self, ICname, sin_coeff = 1, gauss_bias = 1, gauss_sigma=0.05):
        if   (ICname == 'sin'):
            self.PHI[0] = sin(sin_coeff*self.mesh)
        elif (ICname == 'gauss'):
            self.PHI[0] = exp(-(self.mesh-gauss_bias)**2/gauss_sigma)

class Block2(Block1):
    
    def __init__(self, cxl, cxr, xmin, xmax, sep, ct):
        self.mesh     = np.concatenate((Block1(xmin, sep, cxl, ct).mesh, Block1(sep, xmax, cxr, ct).mesh))
        self.PHI      = np.zeros((ct+1, cxl+cxr+2))