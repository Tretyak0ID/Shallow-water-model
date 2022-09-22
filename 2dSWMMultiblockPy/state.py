import numpy as np


class GridField:
    """
    GridField class implements list of numpy arrays.
    Each array can be of different shape.
    GridField provides summation and multiplication by scalar methods
    """
    def __init__(self, f):
        self.f = f

    def __getitem__(self, item):
        return self.f[item]

    def __setitem__(self, key, value):
        self.f[key] = value

    @classmethod
    def empty(cls, domains):
        return cls([np.empty((domain.ny + 1, domain.nx + 1)) for domain in domains])

    @classmethod
    def zeros(cls, domains):
        return cls([np.zeros((domain.ny + 1, domain.nx + 1)) for domain in domains])

    @classmethod
    def ones(cls, domains):
        return cls([100.0*np.ones((domain.ny + 1, domain.nx + 1)) for domain in domains])

    def __add__(self, other):
        if type(other) == GridField:
            return GridField([self[k] + other[k] for k in range(len(self.f))])
        else:
            return GridField([self[k] + other for k in range(len(self.f))])

    def __sub__(self, other):
        return GridField([self[k] - other[k] for k in range(len(self.f))])

    def __mul__(self, other):
        if type(other) == GridField:
            return GridField([self[k]*other[k] for k in range(len(self.f))])
        else:
            return GridField([other*f for f in self.f])

    __rmul__ = __mul__


class State:
    def __init__(self, u, v, h):
        self.u = u
        self.v = v
        self.h = h

    @classmethod
    def empty(cls, domains):
        return cls(GridField.empty(domains), GridField.empty(domains), GridField.empty(domains))

    @classmethod
    def zeros(cls, domains):
        return cls(GridField.zeros(domains), GridField.zeros(domains), GridField.zeros(domains))

    @classmethod
    def ones(cls, domains):
        return cls(GridField.ones(domains), GridField.ones(domains), GridField.zeros(domains))

    def __add__(self, other):
        return State(self.u + other.u, self.v + other.v, self.h + other.h)

    def __sub__(self, other):
        State(self.u - other.u, self.v - other.v, self.h - other.h)

    def __mul__(self, scalar):
        return State(scalar * self.u, scalar * self.v, scalar * self.h)

    __rmul__ = __mul__
