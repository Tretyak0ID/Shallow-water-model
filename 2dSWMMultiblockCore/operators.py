import numpy as np
from state import GridField


def calc_grad(f, domains, diff_method):
    gx = GridField.empty(domains)
    gy = GridField.empty(domains)
    for ind, domain in enumerate(domains):
        gx[ind] = diff_method(f[ind], 'x', domain)
        gy[ind] = diff_method(f[ind], 'y', domain)
    return sbp_SAT_penalty_two_block(gx, f, 'x', domains, diff_method.__name__), \
           sbp_SAT_penalty_two_block(gy, f, 'y', domains, diff_method.__name__)


def calc_div(u, v, domains, diff_method):
    div_x = GridField.empty(domains)
    div_y = GridField.empty(domains)
    for ind, domain in enumerate(domains):
        div_x[ind] = diff_method(u[ind], 'x', domain)
        div_y[ind] = diff_method(v[ind], 'y', domain)
    return sbp_SAT_penalty_two_block(div_x, u, 'x', domains, diff_method.__name__) + \
           sbp_SAT_penalty_two_block(div_y, v, 'y', domains, diff_method.__name__)


def calc_curl(u, v, domains, diff_method):
    curl_x = GridField.empty(domains)
    curl_y = GridField.empty(domains)
    for ind, domain in enumerate(domains):
        curl_x[ind] = diff_method(v[ind], 'x', domain)
        curl_y[ind] = diff_method(u[ind], 'y', domain)
    return sbp_SAT_penalty_two_block(curl_x, v, 'x', domains, diff_method.__name__) - \
           sbp_SAT_penalty_two_block(curl_y, u, 'y', domains, diff_method.__name__)


def diff_sbp21(f, direction, domain):
    out = np.empty_like(f)
    if direction == 'y':
        for j in range(1, domain.ny):
            out[j, :] = (f[j + 1, :] - f[j - 1, :]) / 2.0 / domain.dy
        out[-1, :] = (f[1, :] - f[-2, :]) / 2.0 / domain.dy
        out[0, :] = out[-1, :]
    elif direction == 'x':
        for i in range(1, domain.nx):
            out[:, i] = (f[:, i + 1] - f[:, i - 1]) / 2.0 / domain.dx
        out[:, -1] = (f[:, -1] - f[:, -2]) / domain.dx
        out[:, 0] = (f[:, 1] - f[:, 0]) / domain.dx
    else:
        raise Exception(f"Error in diff_sbp21. Wrong direction value {direction}!")
    return out


def diff_sbp42(f, direction, domain):
    diff_f = np.empty_like(f)
    if direction == 'y':
        for i in range(2, (domain.ny - 1)):
            diff_f[i, :] = (f[i - 2, :] - 8 * f[i - 1, :] + 8 * f[i + 1, :] - f[i + 2, :]) / (12 * domain.dy)
        diff_f[0, :] = (f[-3, :] - 8 * f[-2, :] + 8 * f[1, :] - f[2, :]) / (12 * domain.dy)
        diff_f[1, :] = (f[-2, :] - 8 * f[0, :] + 8 * f[2, :] - f[3, :]) / (12 * domain.dy)
        diff_f[-2, :] = (f[-4, :] - 8 * f[-3, :] + 8 * f[0, :] - f[1, :]) / (12 * domain.dy)
        diff_f[-1, :] = diff_f[0, :]
    elif direction == 'x':
        for i in range(4, (domain.nx - 3)):
            diff_f[:, i] = (f[:, i - 2] - 8 * f[:, i - 1] + 8 * f[:, i + 1] - f[:, i + 2]) / (12 * domain.dx)
        diff_f[:, 0] = (-24 / 17 * f[:, 0] + 59 / 34 * f[:, 1] - 4 / 17 * f[:, 2] - 3 / 34 * f[:, 3]) / (domain.dx)
        diff_f[:, 1] = (-f[:, 0] + f[:, 2]) / (2 * domain.dx)
        diff_f[:, 2] = (4 / 43 * f[:, 0] - 59 / 86 * f[:, 1] + 59 / 86 * f[:, 3] - 4 / 43 * f[:, 4]) / (domain.dx)
        diff_f[:, 3] = (3 / 98 * f[:, 0] - 59 / 98 * f[:, 2] + 32 / 49 * f[:, 4] - 4 / 49 * f[:, 5]) / (domain.dx)

        diff_f[:, -1] = (24 / 17 * f[:, -1] - 59 / 34 * f[:, -2] + 4 / 17 * f[:, -3] + 3 / 34 * f[:, -4]) / (domain.dx)
        diff_f[:, -2] = (f[:, -1] - f[:, -3]) / (2 * domain.dx)
        diff_f[:, -3] = (-4 / 43 * f[:, -1] + 59 / 86 * f[:, -2] - 59 / 86 * f[:, -4] + 4 / 43 * f[:, -5]) / (domain.dx)
        diff_f[:, -4] = (-3 / 98 * f[:, -1] + 59 / 98 * f[:, -3] - 32 / 49 * f[:, -5] + 4 / 49 * f[:, -6]) / (domain.dx)
    else:
        raise Exception(f"Error in diff_sbp42. Wrong direction value {direction}!")
    return diff_f


def sbp_SAT_penalty_two_block(tend, f, direction, domains, diff_method_name):
    """
    This function implements SAT penalty boundary synchronization
    within two block domain. Blocks aligned in x-direction.
    Step size in the y-direction is considered the same for both blocks
    :param f: GridField class instance (list of numpy arrays)
    :param domains: list of domains. domain[0] - left, domain[1] - right
    :return:
    """

    if diff_method_name == "diff_sbp21":
        h0 = 1.0 / 2
        if domains[0].ny == domains[1].ny:
            interp_method = lambda x, y: x
        elif domains[0].ny == 2 * domains[1].ny:
            interp_method = interp_1d_sbp21_2to1_ratio
        else:
            raise Exception(f"Error in sbp_SAT_penalty_two_block. Unknown resolution ratio!")
    elif diff_method_name == "diff_sbp42":
        h0 = 17.0 / 48
        if domains[0].ny == domains[1].ny:
            interp_method = lambda x, y: x
        elif domains[0].ny == 2 * domains[1].ny:
            interp_method = interp_1d_sbp42_2to1_ratio
        else:
            raise Exception(f"Error in sbp_SAT_penalty_two_block. Unknown resolution ratio!")

    if direction == 'x':

        # SAT in x direction (assuming two blocks in x direction)
        ff = interp_method(f[1][:, 0], "coarse2fine")
        df = (f[0][:, -1] - ff) / (domains[0].dx * h0)
        tend[0][:, -1] += -df / 2

        ff = interp_method(f[0][:, -1], "fine2coarse")
        df = (-f[1][:, 0] + ff) / (domains[1].dx * h0)
        tend[1][:, 0] += -df / 2

        ff = interp_method(f[1][:, -1], "coarse2fine")
        df = (ff - f[0][:, 0]) / (domains[0].dx * h0)
        tend[0][:, 0] += -df / 2

        ff = interp_method(f[0][:, 0], "fine2coarse")
        df = (f[1][:, -1] - ff) / (domains[1].dx * h0)
        tend[1][:, -1] += -df / 2

    elif direction == 'y':
        pass
    else:
        raise Exception(f"Error in sbp_SAT_penalty_two_block. Wrong direction value {direction}!")

    return tend


def interp_1d_sbp21_2to1_ratio(f, direction):
    if direction == "coarse2fine":
        out = np.empty(2 * f.size - 1)
        for i in range(f.size - 1):
            out[2*i] = f[i]
            out[2*i+1] = (f[i] + f[i+1])/2.0
        out[-1] = f[-1]
    elif direction == "fine2coarse":
        out = np.empty((f.size + 1) // 2)
        for i in range(1,out.size-1):
            out[i] = (f[2*i-1] + 2*f[2*i] +f[2*i+1])/4.0
        out[0] = (f[-2]  + 2*f[0]  + f[1])/4.0
        out[-1] = (f[-2] + 2*f[-1] + f[1])/4.0
    else:
        raise Exception(f"Error in interp_1d_sbp21_2to1_ratio. Wrong direction value {direction}!")
    return out

def interp_1d_sbp42_2to1_ratio(f, direction):
    if direction == "coarse2fine":
        out = np.empty(2 * f.size - 1)
        for i in range(1,f.size - 2):
            out[2*i] = f[i]
            out[2*i+1] = (-f[i-1] + 9*f[i] + 9*f[i+1] - f[i+2])/16.0
        out[0] = f[0]
        out[1] = (-f[-2] + 9*f[0] + 9*f[1] - f[2])/16.0
        out[-2] = (-f[-3] + 9*f[-2] + 9*f[-1] - f[1])/16.0
        out[-3] = f[-2]
        out[-1] = f[-1]
    elif direction == "fine2coarse":
        out = np.empty((f.size + 1) // 2)
        for i in range(3,out.size-3):
            out[i] = (-f[2*i-3] +9*f[2*i-1] + 16*f[2*i]+ 9*f[2*i+1] - f[2*i+3])/32.0
        out[0]  = (-f[-4] +9*f[-2] + 16*f[0]+ 9*f[1] - f[3])/32.0
        out[1]  = (-f[-2] +9*f[1] + 16*f[2]+ 9*f[3] - f[5])/32.0
        out[2]  = (-f[1] +9*f[3] + 16*f[4]+ 9*f[5] - f[7])/32.0
        out[-3] = (-f[-8] +9*f[-6] + 16*f[-5]+ 9*f[-4] - f[-2])/32.0
        out[-2] = (-f[-6] +9*f[-4] + 16*f[-3]+ 9*f[-2] - f[1])/32.0
        out[-1] = (-f[-4] +9*f[-2] + 16*f[-1]+ 9*f[1] - f[3])/32.0
    else:
        raise Exception(f"Error in interp_1d_sbp42_2to1_ratio. Wrong direction value {direction}!")
    return out