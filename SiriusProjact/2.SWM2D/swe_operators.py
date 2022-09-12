import operators as op
import numpy as np
from state import State


class SweLinearOperator:

    def __init__(self, g, H, pcori, diff_method):
        self.g = g
        self.H = H
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domain):
        gx, gy = op.calc_grad(state.h, domain, self.diff_method)
        div = op.calc_div(state.u, state.v, domain, self.diff_method)
        return State(- self.g * gx + self.pcori * state.v,
                     - self.g * gy - self.pcori * state.u,
                     - self.H * div)


class SweVecInvFormOperator:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domain):
        div = op.calc_div(state.h * state.u, state.h * state.v, domain, self.diff_method)

        kin_energy = (state.u ** 2 + state.v ** 2) / 2
        gx, gy = op.calc_grad(self.g * state.h + kin_energy, domain, self.diff_method)

        curl = op.calc_curl(state.u, state.v, domain, self.diff_method)

        return State(- gx + (self.pcori + curl) * state.v,
                     - gy - (self.pcori + curl) * state.u,
                     - div)


class SweAdvectiveFormOperator:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domain):
        gx, gy = op.calc_grad(state.h, domain, self.diff_method)
        div = op.calc_div(state.h * state.u, state.h * state.v, domain, self.diff_method)

        du_dx, du_dy = op.calc_grad(state.u, domain, self.diff_method)
        dv_dx, dv_dy = op.calc_grad(state.v, domain, self.diff_method)

        return State(- self.g * gx + self.pcori * state.v - state.u * du_dx - state.v * du_dy,
                     - self.g * gy - self.pcori * state.u - state.u * dv_dx - state.v * dv_dy,
                     - div)

class SweAdvectionOnlyFormOperator:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domain):
        gx, gy = op.calc_grad(state.h, state.u, state.v, domain, self.diff_method)
        out = - state.u*gx - state.v*gy

        return State(0,
                     0,
                     out)
