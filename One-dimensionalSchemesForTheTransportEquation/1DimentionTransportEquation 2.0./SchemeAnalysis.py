import numpy              as np
import scipy

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma, sqrt
from pylab                import *

class SchemeAnalyzer:
#Обычный анализатор
    def __init__(self, time_operator, space_operator): 
        if (time_operator.space_operator == space_operator):
            self.time_operator = time_operator
            self.space_operator = space_operator
        else:
            print("The operators don't match")

    def PrintCourantNumber(self):
        #печатает число Куранта для данной схемы
        print("Courant number: " + str(self.space_opetaror.v*self.time_operator.tau/self.space_operator.h))

    def l2Norm(self, f):
        #отслеживание l2-нормы (закон сохранения энергии)
        l2 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def l1Norm(self, f):
        #отслеживание сохранения интеграла (закон сохранения массы)
        l1 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@self.space_operator.diff(f[j,:]))
        return l1



class SBPAnalyzer(SchemeAnalyzer):
#Анализатор SBP-схем
    def l2Norm(self, f):
        #отслеживание l2-H-нормы (закон сохранения энергии)
        l2 = np.zeros(f[:,0].size)
        M = self.space_operator.H(f[0,:].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@M@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def l1Norm(self, f):
        #отслеживание сохранения интеграла с H (закон сохранения массы)
        l1 = np.zeros(f[:,0].size)
        M = self.space_operator.H(f[0,:].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@M@self.space_operator.diff(f[j,:]))
        return l1
