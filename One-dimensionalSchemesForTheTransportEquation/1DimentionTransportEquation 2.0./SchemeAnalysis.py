import numpy              as np
import scipy
import SpaceDifferentiationOperators as SD 
import TimeDifferentiationOperators as TD

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
        print("Courant number: " + str(self.space_operator.v*self.time_operator.tau/self.space_operator.h))

    def l2Norm(self, f):
        #отслеживание l2-нормы (закон сохранения энергии) для всего решения
        l2 = np.zeros(f[:,0].size)
        for j in range(f[:,0].size):
            l2[j] = abs(f[j,:]@self.space_operator.diff(f[j,:]))*self.space_operator.h
        return l2

    def l2norm(self, f):
        #l2-норма функции
        return sqrt(f@f*self.space_operator.h)

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

    def l2norm(self, f):
        #l2-норма функции
        M = self.space_operator.H(f.size)
        return sqrt(abs(f@M@f)*self.space_operator.h)

    def l1Norm(self, f):
        #отслеживание сохранения интеграла с H (закон сохранения массы)
        l1 = np.zeros(f[:,0].size)
        M = self.space_operator.H(f[0,:].size)
        for j in range(f[:,0].size):
            l1[j] = abs(np.ones(f[j,:].size)@M@self.space_operator.diff(f[j,:]))
        return l1

class ApproxAnalyzer:
    def __init__(self, time_operator, space_operator): 
        self.time_operator = time_operator
        self.space_operator = space_operator

    def SinApprox(self, basis, num):
        l2Error = np.zeros((num))

        for p in range(num):
            cx  = 10*(basis**p)
            ct  = 20*(basis**p)
            h   = 2*pi/cx
            tau = 2*pi/ct
            #Coose scheme
            if (self.time_operator == 'Euler'):
                if  (self.space_operator == 'LeftPeriodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.LeftPeriodic(h, 1, cx)
                    T = TD.Euler(tau, D)
                    A = SchemeAnalyzer(T,D)
                    A.PrintCourantNumber()

                elif(self.space_operator == 'Center2Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center2Periodic(h, 1, cx)
                    T = TD.Euler(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center4Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center4Periodic(h, 1, cx)
                    T = TD.Euler(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'SBP21PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21PROJ(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42PROJ(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP21SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21SAT(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42SAT(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

            elif(self.time_operator == 'RK4'):
                if  (self.space_operator == 'LeftPeriodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.LeftPeriodic(h, 1, cx)
                    T = TD.RK4(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center2Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center2Periodic(h, 1, cx)
                    T = TD.RK4(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center4Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center4Periodic(h, 1, cx)
                    T = TD.RK4(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'SBP21PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21PROJ(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42PROJ(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP21SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21SAT(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42SAT(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

            analit = sin(x)
            PHI = np.zeros((ct+1,x.size))
            PHI[0] = analit
            T.diff(PHI)
            l2Error[p] = A.l2norm(PHI[-1]-analit)
        
        return l2Error


    
    def GaussApprox(self, basis, num):
        l2Error = np.zeros((num))

        for p in range(num):
            cx  = 10*(basis**p)
            ct  = 20*(basis**p)
            h   = 2/cx
            tau = 2/ct
            #Coose scheme
            if (self.time_operator == 'Euler'):
                if  (self.space_operator == 'LeftPeriodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.LeftPeriodic(h, 1, cx)
                    T = TD.Euler(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center2Periodic'):
                    x  = np.arange(-1 , cx, dtype=double)*h
                    D = SD.Center2Periodic(h, 1, cx)
                    T = TD.Euler(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center4Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center4Periodic(h, 1, cx)
                    T = TD.Euler(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'SBP21PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21PROJ(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42PROJ(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP21SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21SAT(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42SAT(h, 1, cx+1)
                    T = TD.Euler(tau, D)
                    A = SBPAnalyzer(T,D)

            elif(self.time_operator == 'RK4'):
                if  (self.space_operator == 'LeftPeriodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.LeftPeriodic(h, 1, cx)
                    T = TD.RK4(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center2Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center2Periodic(h, 1, cx)
                    T = TD.RK4(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'Center4Periodic'):
                    x  = np.arange(0 , cx, dtype=double)*h
                    D = SD.Center4Periodic(h, 1, cx)
                    T = TD.RK4(tau, D)
                    A = SchemeAnalyzer(T,D)

                elif(self.space_operator == 'SBP21PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21PROJ(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42PROJ'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42PROJ(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP21SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP21SAT(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

                elif(self.space_operator == 'SBP42SAT'):
                    x  = np.arange(0 , cx + 1, dtype=double)*h
                    D = SD.SBP42SAT(h, 1, cx+1)
                    T = TD.RK4(tau, D)
                    A = SBPAnalyzer(T,D)

            analit = exp((-(x-1)**2)/(0.05))
            PHI = np.zeros((ct+1,x.size))
            PHI[0] = analit
            T.diff(PHI)
            l2Error[p] = A.l2norm(PHI[-1]-analit)
        
        return l2Error
        


    