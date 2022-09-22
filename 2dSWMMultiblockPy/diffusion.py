from operators import *
from state import State


class DiffusionOperator:

    def __init__(self, coefs, diff2_method):
        self.coefs = coefs
        self.diff2_method = diff2_method

    def calc_rhs(self, state, domains):
        lap_u = calc_laplacian(calc_laplacian(state.u, domains, self.diff2_method, self.coefs),
                               domains, self.diff2_method, self.coefs)
        lap_v = calc_laplacian(calc_laplacian(state.v, domains, self.diff2_method, self.coefs),
                               domains, self.diff2_method, self.coefs)
        lap_h = calc_laplacian(calc_laplacian(state.h, domains, self.diff2_method, self.coefs),
                               domains, self.diff2_method, self.coefs)
        return -1.0*State(lap_u, lap_v, lap_h)


def calc_laplacian(f, domains, diff2_method, coefs):

    diff = GridField.empty(domains)
    for ind, domain in enumerate(domains):
        diff[ind] = coefs[ind] * (diff2_method(f[ind], 'x', domain) + diff2_method(f[ind], 'y', domain))
    return sbp_SAT_penalty_two_block_diffusion(diff, f, coefs, domains, diff2_method.__name__)


def diff2_sbp21(f, direction, domain):
    out = np.empty_like(f)
    if direction == 'y':
        out[0, :] = (f[-2, :] - 2.0 * f[0, :] + f[1, :])
        for j in range(1, domain.ny):
            out[j, :] = (f[j - 1, :] - 2.0 * f[j, :] + f[j + 1, :])
        out[-1, :] = (f[-2, :] - 2.0 * f[-1, :] + f[1, :])

        return out / domain.dy ** 2

    elif direction == 'x':
        out[:, 0] = (f[:, 0] - 2.0 * f[:, 1] + f[:, 2])
        for i in range(1, domain.nx):
            out[:, i] = (f[:, i - 1] - 2.0 * f[:, i] + f[:, i + 1])
        out[:, -1] = (f[:, -3] - 2.0 * f[:, -2] + f[:, -1])

        return out / domain.dx ** 2
    else:
        raise Exception(f"Error in diff_sbp21. Wrong direction value {direction}!")


def diff2_sbp21_boundary(f, domain):
    fl = (3.0 * f[:,  0] - 4.0 * f[:,  1] + f[:,  2]) / 2 / domain.dx
    fr = (3.0 * f[:, -1] - 4.0 * f[:, -2] + f[:, -3]) / 2 / domain.dx
    return fl, fr


def sbp_SAT_penalty_two_block_diffusion(tend, f, coefs, domains, diff2_method_name):

    if diff2_method_name == "diff2_sbp21":
        h0 = 1.0 / 2
        interp_method = lambda x, y: x
        boundary_method = diff2_sbp21_boundary
        if domains[0].ny == 2 * domains[1].ny:
            interp_method = interp_1d_sbp21_2to1_ratio
    else:
        raise Exception(f"Error in sbp_SAT_penalty_two_block_diffusion. Diff method {diff2_method_name} "
                        f"is not supported!")

    fl0, fr0 = boundary_method(f[0], domains[0])
    fl1, fr1 = boundary_method(f[1], domains[1])

    # SAT in x direction (assuming two blocks in x direction)
    ff = interp_method(fl1, "coarse2fine")
    df = (coefs[0] * fr0 + coefs[1] * ff) / (domains[0].dx * h0)
    tend[0][:, -1] += -df / 2

    ff = interp_method(fr0, "fine2coarse")
    df = (coefs[1]*fl1 + coefs[0]*ff) / (domains[1].dx * h0)
    tend[1][:, 0] += -df / 2

    ff = interp_method(fr1, "coarse2fine")
    df = (coefs[1] * ff + coefs[0] * fl0) / (domains[0].dx * h0)
    tend[0][:, 0] += -df / 2

    ff = interp_method(fl0, "fine2coarse")
    df = (coefs[1] * fr1 + coefs[0] * ff) / (domains[1].dx * h0)
    tend[1][:, -1] += -df / 2

    return tend