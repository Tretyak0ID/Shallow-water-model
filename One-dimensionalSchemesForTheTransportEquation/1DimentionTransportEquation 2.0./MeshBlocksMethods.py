#-MODULES-AND-LIBRARYS-
import numpy              as np
import scipy
import math
import SpaceDifferentiationOperators as SD 
import TimeDifferentiationOperators as TD

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma
from pylab                import *
#----------------------------------------------------------------------------------------#
#-------------------------------------BEGIN----------------------------------------------#
#----------------------------------------------------------------------------------------#

#-BLOCK-STRUCTURE-
class Block:

    def __init__(self, xmin, xmax, cx):
        self.mesh           = np.arange(0 , cx+1, dtype=double)*(xmax-xmin)/cx + xmin

class Block2Mesh:
    def __init__(self, cxl, cxr, xmin, xmax, sep, ct):
        self.blockl   = Block(xmin, sep, cxl)
        self.blockr   = Block(sep, xmax, cxr)
        self.PHIl     = np.zeros((ct+1, self.blockl.mesh.size))
        self.PHIr     = np.zeros((ct+1, self.blockr.mesh.size))

    def DefineIC(self, ICl , ICr):
        self.PHIl[0] = ICl
        self.PHIr[0] = ICr

    def GetFullSolution(self):
        PHI_full = np.zeros((self.PHIl[:,0].size, np.concatenate((self.PHIl[0], self.PHIr[0])).size))
        for i in range(self.PHIl[:,0].size):
            PHI_full[i] = np.concatenate((self.PHIl[i], self.PHIr[i]))
        return [np.concatenate((self.blockl.mesh, self.blockr.mesh)), PHI_full]