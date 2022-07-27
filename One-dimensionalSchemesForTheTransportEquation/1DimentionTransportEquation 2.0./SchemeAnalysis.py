import numpy              as np
import scipy
import math
import SpaceDifferentiationOperators as SD 
import TimeDifferentiationOperators as TD

from scipy                import sparse
from scipy.sparse         import linalg
from numpy                import pi, sin, cos, ma, sqrt, exp
from pylab                import *
#----------------------------------------------------------------------------------------#
#-------------------------------------BEGIN----------------------------------------------#
#----------------------------------------------------------------------------------------#
class SchemeAnalyzer:
#Обычный анализатор cхемы
    def __init__(self, time_operator, space_operator): 
        self.time_operator = time_operator
        self.space_operator = space_operator

    def PrintCourantNumber(self):
        #печатает число Куранта для данной схемы
        print("Courant number: " + str(self.space_operator.v*self.time_operator.tau/self.space_operator.h))

    def l2norm(self, f):
        #l2-норма функции
        return sqrt(f@f*self.space_operator.h)

    def l1norm(self, f):
        #l1-норма функции
        return abs(np.ones(f.size)@f)

    def Maxnorm(self, f):
        #Равномерная норма функции
        return max(abs(f))

    def EnergyCons(self, f):
        #отслеживание изменения l2-нормы (закон сохранения энергии)
        l2 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def MassCons(self, f):
        #отслеживание сохранения интеграла (закон сохранения массы)
        l1 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@self.space_operator.diff(f[j,:]))
        return l1



class SBPAnalyzer(SchemeAnalyzer):
#Анализатор SBP-схем
    def EnergyCons(self, f):
        l2 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@self.space_operator.H(f[0,:].size)@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def MassCons(self, f):
        l1 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@self.space_operator.H(f[0,:].size)@self.space_operator.diff(f[j,:]))
        return l1

    def l2norm(self, f):
        return sqrt(f@self.space_operator.H(f.size)@f*self.space_operator.h)

    def l1norm(self, f):
        return abs(np.ones(f.size)@self.space_operator.H(f.size)@f)



class ConvAnalyzer:
#Анализатор сходимости схем
    def __init__(self, time_operator, space_operator, v): 
        self.time_operator  = time_operator
        self.space_operator = space_operator
        self.v              = v

    def SpaceDet(self, cx, xmax):
        #левые направленные разности
        if  (self.space_operator == 'Left1Periodic'):
            return SD.Left1Periodic(xmax/cx, self.v, cx)
        elif(self.space_operator == 'Left2Periodic'):
            return SD.Left2Periodic(xmax/cx, self.v, cx)
        elif(self.space_operator == 'Left21Periodic'):
            return SD.Left21Periodic(xmax/cx, self.v, cx)
        elif(self.space_operator == 'Left31Periodic'):
            return SD.Left31Periodic(xmax/cx, self.v, cx)
        elif(self.space_operator == 'Left32Periodic'):
            return SD.Left32Periodic(xmax/cx, self.v, cx)

        #Центральные схемы
        elif(self.space_operator == 'Center2Periodic'):
            return SD.Center2Periodic(xmax/cx, self.v, cx)
        elif(self.space_operator == 'Center4Periodic'):
            return SD.Center4Periodic(xmax/cx, self.v, cx)

        #SBP-SAT
        elif(self.space_operator == 'SBP21PROJ'):
            return SD.SBP21PROJ(xmax/cx, self.v, cx+1)
        elif(self.space_operator == 'SBP42PROJ'):
            return SD.SBP42PROJ(xmax/cx, self.v, cx+1)
        elif(self.space_operator == 'SBP21SAT'):
            return SD.SBP21SAT(xmax/cx, self.v, cx+1)
        elif(self.space_operator == 'SBP42SAT'):
            return SD.SBP42SAT(xmax/cx, self.v, cx+1)

    def TimeDef(self, ct, tmax, D):
        if   (self.time_operator == 'Euler'):
            return TD.Euler(tmax/ct, D)
        elif(self.time_operator == 'RK2'):
            return TD.RK2(tmax/ct, D)
        elif(self.time_operator == 'RK4'):
            return TD.RK4(tmax/ct, D)
        elif(self.time_operator == 'RK6'):
            return TD.RK6(tmax/ct, D)

    def SinTestConv(self, base, fcx, fct, tnum):
        #Запускает счет на исходной и сгущенной в base раз сетки с синусом на tnum периодов, возвращает отношение l1,l2,max норм ошибок, стороит их графики
        scx = fcx*base
        sct = fct*base

        D1 = self.SpaceDet(fcx, 2*pi)
        D2 = self.SpaceDet(scx, 2*pi)
        T1 = self.TimeDef(fct, 2*pi*tnum, D1)
        T2 = self.TimeDef(sct, 2*pi*tnum, D2)
        
        if('SBP' in self.space_operator):
            x1      = np.arange(0 , fcx + 1, dtype=double)*2*pi/fcx
            x2      = np.arange(0 , scx + 1, dtype=double)*2*pi/scx
            PHI1    = np.zeros((fct + 1, fcx + 1))
            PHI2    = np.zeros((sct + 1, scx + 1))
            M1      = D1.H(x1.size)
            M2      = D2.H(x2.size)
        else:
            x1      = np.arange(0 , fcx, dtype=double)*2*pi/fcx
            x2      = np.arange(0 , scx, dtype=double)*2*pi/scx
            PHI1    = np.zeros((fct+1, fcx))
            PHI2    = np.zeros((sct+1, scx))
            M1      = np.eye(x1.size)
            M2      = np.eye(x2.size)

        PHI1[0] = sin(x1)
        PHI2[0] = sin(x2)
        for i in range(fct+1):
            T1.diff(PHI1)
            T2.diff(PHI2)
        
        Error1 = PHI1[-1] - sin(x1)
        Error2 = PHI2[-1] - sin(x2)

        fig,ax = plt.subplots(1,2)
        fig.set_size_inches(15,5)
        ax[0].plot(x1,Error1)
        ax[1].plot(x2,Error2)

        print("Order of Convergence: " + str(math.log(sqrt(2*pi/fcx*Error1@Error1)/sqrt(2*pi/scx*Error2@Error2),base)))
        return sqrt(2*pi/fcx*Error1@Error1)/sqrt(2*pi/scx*Error2@Error2)

    def GaussTestConv(self, base, fcx, fct, tnum):
        #Запускает счет на исходной и сгущенной в base раз сетки с синусом на tnum периодов, возвращает отношение l1,l2,max норм ошибок, стороит их графики
        scx = fcx*base
        sct = fct*base

        D1 = self.SpaceDet(fcx, 2)
        D2 = self.SpaceDet(scx, 2)
        T1 = self.TimeDef(fct, 2*tnum, D1)
        T2 = self.TimeDef(sct, 2*tnum, D2)
        
        if('SBP' in self.space_operator):
            x1      = np.arange(0 , fcx + 1, dtype=double)*2/fcx
            x2      = np.arange(0 , scx + 1, dtype=double)*2/scx
            PHI1    = np.zeros((fct + 1, fcx + 1))
            PHI2    = np.zeros((sct + 1, scx + 1))
            M1      = D1.H(x1.size)
            M2      = D2.H(x2.size)
        else:
            x1      = np.arange(0 , fcx, dtype=double)*2/fcx
            x2      = np.arange(0 , scx, dtype=double)*2/scx
            PHI1    = np.zeros((fct+1, fcx))
            PHI2    = np.zeros((sct+1, scx))
            M1      = np.eye(x1.size)
            M2      = np.eye(x2.size)

        PHI1[0] = exp(-(x1-1)**2/0.5)
        PHI2[0] = exp(-(x2-1)**2/0.5)
        for i in range(fct+1):
            T1.diff(PHI1)
            T2.diff(PHI2)
        
        Error1 = PHI1[-1] - exp(-(x1-1)**2/0.5)
        Error2 = PHI2[-1] - exp(-(x2-1)**2/0.5)

        fig,ax = plt.subplots(1,2)
        fig.set_size_inches(15,5)
        ax[0].plot(x1,Error1)
        ax[1].plot(x2,Error2)

        print("Order of Convergence: " + str(math.log(sqrt(2/fcx*Error1@Error1)/sqrt(2/scx*Error2@Error2),base)))
        return sqrt(2/fcx*Error1@Error1)/sqrt(2/scx*Error2@Error2)

