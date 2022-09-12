import numpy as np


def calc_grad(f, u, v, domain, diff_method):
    return diff_method(f, u, v, 'x', domain), diff_method(f, u, v, 'y', domain)


def calc_div(f, u, v, domain, diff_method):
    return diff_method(u, u, v, 'x', domain) + diff_method(v, u, v, 'y', domain)


def calc_curl(u, v, domain, diff_method):
    return -diff_method(u, u, v, 'y', domain) + diff_method(v, u, v, 'x', domain)


def central_diff2(f, u, v, direction, domain):
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

def left_diff1(f, u, v, direction, domain):
    out = np.empty_like(f)
    if direction == 'y':
        for j in range(1, domain.ny+1):
            out[j, :] = (f[j, :] - f[j - 1, :])  / domain.dy
        out[0, :] = (f[0, :] - f[-1, :])  / domain.dy
    elif direction == 'x':
        for i in range(1, domain.nx+1):
            out[:, i] = (f[:, i] - f[:, i - 1])  / domain.dx
        out[:, 0] = (f[:, 0] - f[:, - 1])  / domain.dx
    else:
        raise Exception(f"Error in left_diff1. Wrong direction value {direction}!")
    return out

def upstream1(f, u, v, direction, domain):
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

def upstream4(f, u, v, direction, domain):
    out = np.empty_like(f)

    if direction == 'x':
        for i in range(3, domain.nx-2):
            for j in range(domain.ny + 1):
                if u[j, i] >= 0 :
                    out[j, i] = (3 * f[j, i + 1] + 10 * f[j, i] - 18 * f[j, i - 1] + 6 * f[j, i - 2] - f[j, i - 3]) / 12.0  / domain.dx
                else : 
                    out[j, i] = (-f[j, i + 3] + 6 * f[j, i + 2] - 18 * f[j, i + 1] + 10 * f[j, i] + 3 * f[j, i - 1]) / 12.0  / domain.dx

        for j in range(domain.ny + 1):

            if u[j, -1] >= 0 :
                out[j, -1] = (3 * f[j, 1] + 10 * f[j, -1] - 18 * f[j, -2] + 6 * f[j, -3] - f[j, -4]) / 12.0  / domain.dx
            else :
                out[j, -1] = (-f[j, 3] + 6 * f[j, 2] - 18 * f[j, 1] + 10 * f[j, -1] + 3 * f[j, -2]) / 12.0  / domain.dx

            if u[j, -2] >= 0 :
                out[j, -1] = (3 * f[j, -1] + 10 * f[j, -2] - 18 * f[j, -3] + 6 * f[j, -4] - f[j, -5]) / 12.0  / domain.dx
            else :
                out[j, -1] = (- f[j, 2] + 6 * f[j, 1] - 18 * f[j, -1] + 10 * f[j, -2] + 3 * f[j, -3]) / 12.0  / domain.dx

            if u[j, -3] >= 0 :
                out[j, -1] = (3 * f[j, -2] + 10 * f[j, -3] - 18 * f[j, -4] + 6 * f[j, -5] - f[j, -6]) / 12.0  / domain.dx
            else :
                out[j, -1] = (- f[j, 1] + 6 * f[j, -1] - 18 * f[j, -2] + 10 * f[j, -3] + 3 * f[j, -4]) / 12.0  / domain.dx

            if u[j, 1] >= 0 :
                out[j, -1] = (3 * f[j, 2] + 10 * f[j, 1] - 18 * f[j, 0] + 6 * f[j, -2] - f[j, -3]) / 12.0  / domain.dx
            else :
                out[j, -1] = (- f[j, 0] + 6 * f[j, 1] - 18 * f[j, 2] + 10 * f[j, 3] + 3 * f[j, 4]) / 12.0  / domain.dx

            if u[j, 2] >= 0 :
                out[j, -1] = (3 * f[j, 3] + 10 * f[j, 2] - 18 * f[j, 1] + 6 * f[j, 0] - f[j, -2]) / 12.0  / domain.dx
            else :
                out[j, -1] = (- f[j, 1] + 6 * f[j, 2] - 18 * f[j, 3] + 10 * f[j, 4] + 3 * f[j, 5]) / 12.0  / domain.dx
        
        out[:, 0] = out[:, -1]

    elif direction == 'y':
        for i in range(domain.nx+1):
            for j in range(3, domain.ny - 2):
                if v[j, i] >= 0 :
                    out[j, i] = (3 * f[j + 1, i] + 10 * f[j, i] - 18 * f[j - 1, i] + 6 * f[j - 2, i] - f[j - 3, i]) / 12.0  / domain.dx
                else : 
                    out[j, i] = (-f[j + 3, i] + 6 * f[j + 2, i] - 18 * f[j + 1, i] + 10 * f[j, i] + 3 * f[j - 1, i]) / 12.0  / domain.dx

        for i in range(domain.nx + 1):

            if v[-1, i] >= 0 :
                out[-1, i] = (3 * f[1, i] + 10 * f[-1, i] - 18 * f[-2, i] + 6 * f[-3, i] - f[-4, i]) / 12.0  / domain.dx
            else :
                out[-1, i] = (-f[3, i] + 6 * f[2, i] - 18 * f[1, i] + 10 * f[-1, i] + 3 * f[-2, i]) / 12.0  / domain.dx

            if v[-2, i] >= 0 :
                out[-2, i] = (3 * f[-1, i] + 10 * f[-2, i] - 18 * f[-3, i] + 6 * f[-4, i] - f[-5, i]) / 12.0  / domain.dx
            else :
                out[-1, i] = (- f[2, i] + 6 * f[1, i] - 18 * f[-1, i] + 10 * f[-2, i] + 3 * f[-3, i]) / 12.0  / domain.dx

            if v[-3, i] >= 0 :
                out[-1, i] = (3 * f[-2, i] + 10 * f[-3, i] - 18 * f[-4, i] + 6 * f[-5, i] - f[-6, i]) / 12.0  / domain.dx
            else :
                out[-1, i] = (- f[1, i] + 6 * f[-1, i] - 18 * f[-2, i] + 10 * f[-3, i] + 3 * f[-4, i]) / 12.0  / domain.dx

            if v[1, i] >= 0 :
                out[-1, i] = (3 * f[2, i] + 10 * f[1, i] - 18 * f[0, i] + 6 * f[-2, i] - f[-3, i]) / 12.0  / domain.dx
            else :
                out[-1, i] = (- f[0, i] + 6 * f[1, i] - 18 * f[2, i] + 10 * f[3, i] + 3 * f[4, i]) / 12.0  / domain.dx

            if v[2, i] >= 0 :
                out[-1, i] = (3 * f[3, i] + 10 * f[2, i] - 18 * f[1, i] + 6 * f[0, i] - f[-2, i]) / 12.0  / domain.dx
            else :
                out[-1, i] = (- f[1, i] + 6 * f[2, i] - 18 * f[3, i] + 10 * f[4, i] + 3 * f[5, i]) / 12.0  / domain.dx
        
        out[0, :] = out[-1, :]

    return out

def diff_sbp42(f, u, v, direction, domain):
    out = np.empty_like(f)
    if direction == 'y':
        out[0, :] = -24. / 17 * f[0, :] + 59. / 34 * f[1, :] - 4. / 17 * f[2, :] - 3. / 34 * f[3, :]
        out[1, :] = - 1. / 2 * f[0, :] + 1. / 2 * f[2, :]
        out[2, :] = 4. / 43 * f[0, :] - 59. / 86 * f[1, :] + 59. / 86 * f[3, :] - 4. / 43 * f[4, :]
        out[3, :] = 3. / 98 * f[0, :] - 59. / 98 * f[2, :] + 32. / 49 * f[4, :] - 4. / 49 * f[5, :]

        for j in range(4, domain.ny - 3):
            out[j, :] = (f[j - 2, :] - 8.0 * f[j - 1, :] + 8.0 * f[j + 1, :] - f[j + 2, :]) / 12.0

        out[-1, :] = 24. / 17 * f[-1, :] - 59. / 34 * f[-2, :] + 4. / 17 * f[-3, :] + 3. / 34 * f[-4, :]
        out[-2, :] = 1. / 2 * f[-1, :] - 1. / 2 * f[-3, :]
        out[-3, :] = -4. / 43 * f[-1, :] + 59. / 86 * f[-2, :] - 59. / 86 * f[-4, :] + 4. / 43 * f[-5, :]
        out[-4, :] = -3. / 98 * f[-1, :] + 59. / 98 * f[-3, :] - 32. / 49 * f[-5, :] + 4. / 49 * f[-6, :]

        out[0, :] = 0.5 * (out[0, :] + out[-1, :])
        out[-1, :] = out[0, :]

        out = out / domain.dy

    elif direction == 'x':
        out[:, 0] = -24. / 17 * f[:, 0] + 59. / 34 * f[:, 1] - 4. / 17 * f[:, 2] - 3. / 34 * f[:, 3]
        out[:, 1] = - 1. / 2 * f[:, 0] + 1. / 2 * f[:, 2]
        out[:, 2] = 4. / 43 * f[:, 0] - 59. / 86 * f[:, 1] + 59. / 86 * f[:, 3] - 4. / 43 * f[:, 4]
        out[:, 3] = 3. / 98 * f[:, 0] - 59. / 98 * f[:, 2] + 32. / 49 * f[:, 4] - 4. / 49 * f[:, 5]

        for i in range(4, domain.nx - 3):
            out[:, i] = (f[:, i - 2] - 8.0 * f[:, i - 1] + 8.0 * f[:, i + 1] - f[:, i + 2]) / 12.0

        out[:, -1] = 24. / 17 * f[:, -1] - 59. / 34 * f[:, -2] + 4. / 17 * f[:, -3] + 3. / 34 * f[:, -4]
        out[:, -2] = 1. / 2 * f[:, -1] - 1. / 2 * f[:, -3]
        out[:, -3] = -4. / 43 * f[:, -1] + 59. / 86 * f[:, -2] - 59. / 86 * f[:, -4] + 4. / 43 * f[:, -5]
        out[:, -4] = -3. / 98 * f[:, -1] + 59. / 98 * f[:, -3] - 32. / 49 * f[:, -5] + 4. / 49 * f[:, -6]

        out[:, 0] = 0.5 * (out[:, 0] + out[:, -1])
        out[:, -1] = out[:, 0]

        out = out / domain.dx
    else:
        raise Exception(f"Error in diff_sbp21. Wrong direction value {direction}!")
    return out

def diff_sbp21(f, u, v, direction, domain):
    out = np.empty_like(f)
    if direction == 'y':
        for j in range(1, domain.ny):
            out[j, :] = (f[j + 1, :] - f[j - 1, :]) / 2.0 / domain.dy
        out[-1, :] = (f[-1, :] - f[-2, :]) / domain.dy
        out[0, :] = (f[1, :] - f[0, :]) / domain.dy

        #SAT method
        df = (f[-1, :]-f[0,:]) / (domain.dy / 2)
        out[0 , :] = out[0 , :] - df / 2
        out[-1, :] = out[-1, :] - df / 2
        # projection method
        # out[0, :] = 0.5 * (out[0, :] + out[-1, :])
        # out[-1, :] = out[0, :]

    elif direction == 'x':
        for i in range(1, domain.nx):
            out[:, i] = (f[:, i + 1] - f[:, i - 1]) / 2.0 / domain.dx
        out[:, -1] = (f[:, -1] - f[:, -2]) / domain.dx
        out[:, 0] = (f[:, 1] - f[:, 0]) / domain.dx

        #SAT method
        df = (f[:, -1]-f[:, 0]) / (domain.dx / 2)
        out[:, 0] = out[:, 0] - df / 2
        out[:, -1] = out[:, -1] - df / 2        
        
        # projection method
        # out[:, 0] = 0.5 * (out[:, 0] + out[:, -1])
        # out[:, -1] = out[:, 0]
    else:
        raise Exception(f"Error in diff_sbp21. Wrong direction value {direction}!")
    return out
