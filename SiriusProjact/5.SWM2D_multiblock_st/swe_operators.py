import operators as op
from state import State, GridField


class SweLinearOperator:

    def __init__(self, g, H, pcori, diff_method):
        self.g = g
        self.H = H
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domains):
        gx, gy = op.calc_grad(state.h, domains, self.diff_method)
        div = op.calc_div(state.u, state.v, domains, self.diff_method)
        return State(- self.g * gx + self.pcori * state.v,
                     - self.g * gy - self.pcori * state.u,
                     - self.H * div)


class SweVecInvFormOperator:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domains):
        div = op.calc_div(state.h * state.u, state.h * state.v, domains, self.diff_method)

        kin_energy = (state.u * state.u + state.v * state.v) * 0.5
        gx, gy = op.calc_grad(self.g * state.h + kin_energy, domains, self.diff_method)

        curl = op.calc_curl(state.u, state.v, domains, self.diff_method)

        return State(-1.0*gx + (curl + self.pcori) * state.v,
                     -1.0*gy - (curl + self.pcori) * state.u,
                     -1.0*div)


class SweAdvectiveFormOperator:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domains):
        gx, gy = op.calc_grad(state.h, domains, self.diff_method)
        div = op.calc_div(state.h * state.u, state.h * state.v, domains, self.diff_method)

        du_dx, du_dy = op.calc_grad(state.u, domains, self.diff_method)
        dv_dx, dv_dy = op.calc_grad(state.v, domains, self.diff_method)

        return State(-1.0 * self.g * gx + self.pcori * state.v - du_dx * state.u - du_dy * state.v,
                     -1.0 * self.g * gy - self.pcori * state.u - dv_dx * state.u - dv_dy * state.v,
                     -1.0 * div)


class SweOnlyAdvection:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domains):
        gx, gy = op.calc_grad(state.h, domains, self.diff_method)

        return State(GridField.zeros(domains),
                     GridField.zeros(domains),
                     - 1.0 * state.u * gx - 1.0 * state.v * gy)
