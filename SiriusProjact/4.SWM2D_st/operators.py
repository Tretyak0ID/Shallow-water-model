import numpy as np


def calc_grad(f, domain, diff_method):
    return diff_method(f, 'x', domain), diff_method(f, 'y', domain)


def calc_div(u, v, domain, diff_method):
    return diff_method(u, 'x', domain) + diff_method(v, 'y', domain)


def calc_curl(u, v, domain, diff_method):
    return -diff_method(u, 'y', domain) + diff_method(v, 'x', domain)


def central_diff2(f, direction, domain, u=0, v=0):
    out = np.empty_like(f)
    if direction == 'y':
        for j in range(1, domain.ny):
            out[j, :] = (f[j + 1, :] - f[j - 1, :]) / 2.0 / domain.dy
        out[-1, :] = (f[1, :] - f[-2, :]) / 2.0 / domain.dy
        out[0, :] = out[-1, :]
    elif direction == 'x':
        for i in range(1, domain.nx):
            out[:, i] = (f[:, i + 1] - f[:, i - 1]) / 2.0 / domain.dx
        out[:, -1] = (f[:, 1] - f[:, -2]) / 2.0 / domain.dx
        out[:, 0] = out[:, -1]
    else:
        raise Exception(f"Error in central_diff2. Wrong direction value {direction}!")
    return out

def central_diff4(f, direction, domain, u=0, v=0):
    out = np.empty_like(f)
    if direction == 'y':
        for j in range(2, domain.ny - 1):
            out[j, :] = (-f[j + 2, :] + 8.0*f[j + 1, :] - 8.0*f[j - 1, :] + f[j - 2, :]) / 12.0 / domain.dy
        out[-1, :] = (-f[2, :] + 8.0*f[ 1, :] - 8.0*f[-2, :] + f[-3, :]) / 12.0 / domain.dy
        out[-2, :] = (-f[1, :] + 8.0*f[-1, :] - 8.0*f[-3, :] + f[-4, :]) / 12.0 / domain.dy
        out[1, :]  = (-f[3, :] + 8.0*f[ 2, :] - 8.0*f[ 0, :] + f[-2, :]) / 12.0 / domain.dy
        out[0, :]  = out[-1, :]
    elif direction == 'x':
        for i in range(2, domain.nx - 1):
            out[:, i] = (-f[:, i + 2] + 8.0*f[:, i + 1] - 8.0*f[:, i - 1] + f[:, i - 2]) / 12.0 / domain.dx
        out[:, -1] = (-f[:, 2] + 8.0*f[ :, 1] - 8.0*f[ :,-2] + f[ :,-3]) / 12.0 / domain.dx
        out[:, -2] = (-f[:, 1] + 8.0*f[ :,-1] - 8.0*f[ :,-3] + f[ :,-4]) / 12.0 / domain.dx
        out[:, 1]  = (-f[:, 3] + 8.0*f[ :, 2] - 8.0*f[ :, 0] + f[ :,-2]) / 12.0 / domain.dx
        out[:, 0]  = out[:,-1]
    else:
        raise Exception(f"Error in central_diff4. Wrong direction value {direction}!")
    return out

def upstream1(f, direction, domain, u, v):
    out = np.empty_like(f)

    if direction == 'x':
        for i in range(1, domain.nx):
            for j in range(domain.ny + 1):
                if u[j, i] >= 0 :
                    out[j, i] = (f[j, i] - f[j, i - 1])  / domain.dx
                else : 
                    out[j, i] = (f[j, i + 1] - f[j, i])  / domain.dx

        for j in range(domain.ny + 1):
            if u[j, -1] >= 0 :
                out[j, -1] = (f[j, -1] - f[j, - 2])  / domain.dx
            else :
                out[j, -1] = (f[j,  1] - f[j, - 1])  / domain.dx
        
        out[:, 0] = out[:, -1]

    elif direction == 'y':
        for i in range(domain.nx + 1):
            for j in range(1, domain.ny):
                if v[j, i] >= 0 :
                    out[j, i] = (f[j, i] - f[j - 1, i])  / domain.dy
                else : 
                    out[j, i] = (f[j + 1, i] - f[j, i])  / domain.dy

        for i in range(domain.nx + 1):
            if v[-1, i] >= 0 :
                out[-1, i] = (f[-1, i] - f[-2, i])  / domain.dy
            else :
                out[-1, i] = (f[1,  i] - f[-1, i])  / domain.dy
        
        out[0, :] = out[-1, :]

    return out