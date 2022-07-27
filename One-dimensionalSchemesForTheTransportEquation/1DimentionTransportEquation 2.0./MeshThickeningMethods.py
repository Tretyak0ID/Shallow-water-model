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

#-BLOCK-OPERATORS-
class SBP21SAT2BLOCKS(SD.SpaceDiff):

    def __init__(self, hl, hr, v, sizel, sizer):
        self.hl      = hl
        self.hr      = hr
        self.v       = v
        self.sizel   = sizel-1
        self.sizer   = sizer-1
        self.opl  = SD.SBP21(hl, v, sizel)
        self.opr  = SD.SBP21(hr, v, sizer)

    def SATL(self, fl, fr):
        #SAT-добавки в оператор
        e0 = np.zeros(fl.size)
        eN = np.zeros(fl.size)
        e0[ 0] = 1
        eN[-1] = 1
        return 1/2*(fl[0]-fr[-1])*e0 - 1/2 *(fl[-1] - fr[0])*eN

    def SATR(self, fl, fr):
        #SAT-добавки в оператор
        e0 = np.zeros(fr.size)
        eN = np.zeros(fr.size)
        e0[ 0] = 1
        eN[-1] = 1
        return -1/2*(fl[-1]-fr[0])*e0 + 1/2*(fl[0] - fr[-1])*eN

    def diff(self, fl, fr):
        return np.array([self.v*(self.opl.matrix@fl + np.linalg.solve(self.opl.H(fl.size), self.SATL(fl,fr))), self.v*(self.opr.matrix@fr + np.linalg.solve(self.opr.H(fr.size), self.SATR(fl,fr)))])

class SBP42SAT2BLOCKS(SBP21SAT2BLOCKS):

    def __init__(self, hl, hr, v, sizel, sizer):
        self.hl      = hl
        self.hr      = hr
        self.v       = v
        self.sizel   = sizel-1
        self.sizer   = sizer-1
        self.opl  = SD.SBP42(hl, v, sizel)
        self.opr  = SD.SBP42(hr, v, sizer)

class RK4BLOCKS2():
    def __init__(self, tau, space_op): 
        self.tau       = tau
        self.v         = space_op.v
        self.space_op  = space_op

    def make_step(self, fl, fr):
        fl1, fr1 = -self.space_op.diff(fl,fr)
        fl2, fr2 = -self.space_op.diff(fl1, fr1)
        fl3, fr3 = -self.space_op.diff(fl2, fr2)
        fl4, fr4 = -self.space_op.diff(fl3, fr3)
        return np.array([fl + self.tau*fl1 + self.tau**2/2*fl2 + self.tau**3/6*fl3 + self.tau**4/24*fl4, fr + self.tau*fr1 + self.tau**2/2*fr2 + self.tau**3/6*fr3 + self.tau**4/24*fr4])

    def diff(self, fl, fr):
        for j in range(fl[:,0].size-1):
            buff = self.make_step(fl[j], fr[j])
            fl[j+1] = buff[0]
            fr[j+1] = buff[1]

#-BLOCK-STRUCTURE-
class Block:

    def __init__(self, xmin, xmax, cx):
        self.mesh           = np.arange(0 , cx+1, dtype=double)*(xmax-xmin)/cx + xmin

class Block2Mesh:
    def __init__(self, cxl, cxr, xmin, xmax, sep, space_op, time_op, ct):
        self.blockl   = Block(xmin, sep, cxl)
        self.blockr   = Block(sep, xmax, cxr)
        self.space_op = space_op
        self.time_op  = time_op
        self.PHIl     = np.zeros((ct+1, self.blockl.mesh.size))
        self.PHIr     = np.zeros((ct+1, self.blockr.mesh.size))

    def DefineIC(self, ICl , ICr):
        self.PHIl[0] = ICl
        self.PHIr[0] = ICr

    def diff(self):
        self.time_op.diff(self.PHIl , self.PHIr)

    def GetFullSolution(self):
        PHI_full = np.zeros((self.PHIl[:,0].size, np.concatenate((self.PHIl[0], self.PHIr[0])).size))
        for i in range(self.PHIl[:,0].size):
            PHI_full[i] = np.concatenate((self.PHIl[i], self.PHIr[i]))
        return [np.concatenate((self.blockl.mesh, self.blockr.mesh)), PHI_full]