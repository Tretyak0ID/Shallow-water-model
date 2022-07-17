#-MODULES-AND-LIBRARYS-
import numpy              as np
import scipy

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma, sqrt
from pylab                import *

#-MAIN-CLASS-
class TimeDiff:
    
    #main methods
    def __init__(self, tau, space_operator): 
        self.tau            = tau
        self.v              = space_operator.v
        self.space_operator = space_operator
        
    def make_step(self, f):
        #один шаг по времени
        return f
        
    def diff(self, f):
        #полное дифференцирование по времени
        for j in range(f[:,0].size-1):
            f[j+1,:] = self.make_step(f[j,:])

    #Методы-анализа-ошибок
    def CourantNumber(self):
        print("CourantNumber: " + str(self.v*self.tau/self.space_operator.h))

    def l2Norm(self, f):
        #отслеживание l2-нормы (закон сохранения энергии)
        l2 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def l2HNorm(self, f):
        #отслеживание l2-H-нормы (закон сохранения энергии)
        l2 = np.zeros(f[:,0].size)
        M = self.space_operator.H(f[0,:].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@M@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def l1Norm(self, f):
        #отслеживание сохранения интеграла (закон сохранения массы)
        l1 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@self.space_operator.diff(f[j,:]))
        return l1

    def l1HNorm(self, f):
        #отслеживание сохранения интеграла с H (закон сохранения массы)
        l1 = np.zeros(f[:,0].size)
        M = self.space_operator.H(f[0,:].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@M@self.space_operator.diff(f[j,:]))
        return l1

#-CHILD-CLASES-
#   
class Euler(TimeDiff):

    def make_step(self, f):
        return f + self.tau*self.space_operator.diff(f)
        
class RK4(TimeDiff):

    def make_step(self, f):
        f1 = -self.space_operator.diff(f)
        f2 = -self.space_operator.diff(f1)
        f3 = -self.space_operator.diff(f2)
        f4 = -self.space_operator.diff(f3)
        return f + + self.tau*f1 + self.tau**2/2*f2 + self.tau**3/6*f3 + self.tau**4/24*f4