import numpy as np


class State:
    def __init__(self, u, v, h, q):
        self.u = u  #x-velocity (core)
        self.v = v  #y-velocity (core)
        self.h = h  #hight (core)
        self.q = q  #evaporation (addition)

    @classmethod
    def empty(cls, nx, ny):
        return cls(np.empty((ny + 1, nx + 1)), np.empty((ny + 1, nx + 1)), np.empty((ny + 1, nx + 1)), np.empty((ny + 1, nx + 1)))

    @classmethod
    def zeros(cls, nx, ny):
        return cls(np.zeros((ny + 1, nx + 1)), np.zeros((ny + 1, nx + 1)), np.zeros((ny + 1, nx + 1)), np.zeros((ny + 1, nx + 1)))

    def __add__(self, other):
        return State(self.u + other.u, self.v + other.v, self.h + other.h, self.q + other.q)

    def __mul__(self, other):
        return State(other*self.u, other*self.v, other*self.h, other*self.q)

    def __rmul__(self, other):
        return State(other*self.u, other*self.v, other*self.h, other*self.q)