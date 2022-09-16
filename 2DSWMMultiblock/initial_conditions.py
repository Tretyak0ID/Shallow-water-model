from state import State
import numpy as np
import sympy as sm
import math


def random_h(domains, h_mean):
    state = State.zeros(domains)
    for ind, domain in enumerate(domains):
        state.h[ind] = h_mean + 0.1 * h_mean * np.random.rand(domain.ny+1, domain.nx+1)
    return state


def gaussian_hill(domains, h_mean, xc=None, yc=None, sigma_x=100.0, sigma_y=100.0):
    state = State.zeros(domains)
    xc = xc if xc is not None else domains[0].lx / 2
    yc = yc if yc is not None else domains[0].ly / 2
    for ind, domain in enumerate(domains):
        dx = (domain.xx - 0.8 * xc) / domain.lx
        dy = (domain.yy - 0.8 * yc) / domain.ly
        state.h[ind] = h_mean + np.exp(-sigma_x * dx ** 2 - sigma_y * dy ** 2)
    return state


def gaussian_hill_1rotor(domains, h_mean, xc=None, yc=None, sigma_x=100.0, sigma_y=100.0):
    state = State.zeros(domains)

    xc = xc if xc is not None else domains[0].lx / 2
    yc = yc if yc is not None else domains[0].ly / 2
    for ind, domain in enumerate(domains):
        for i in range(domain.nx + 1):
            for j in range(domain.ny + 1):
                state.u[ind][j,i] =  100*np.sin(np.pi*domain.x[i]/(2 * np.pi * 6371.22 * 1000.0))*np.cos(np.pi*domain.y[j]/(2 * np.pi * 6371.22 * 1000.0))
                state.v[ind][j,i] = -100*np.cos(np.pi*domain.x[i]/(2 * np.pi * 6371.22 * 1000.0))*np.sin(np.pi*domain.y[j]/(2 * np.pi * 6371.22 * 1000.0))

        dx = (domain.xx - 0.5 * xc) / domain.lx
        dy = (domain.yy - 1.0 * yc) / domain.ly
        state.h[ind] = h_mean + np.exp(-sigma_x * dx ** 2 - sigma_y * dy ** 2)

        dx = (domain.xx - 1.5 * xc) / domain.lx
        dy = (domain.yy - 1.0 * yc) / domain.ly
        state.h[ind] += np.exp(-sigma_x * dx ** 2 - sigma_y * dy ** 2)
    return state


def gaussian_hill_2rotor(domains, h_mean, xc=None, yc=None, sigma_x=200.0, sigma_y=200.0):
    state = State.zeros(domains)

    xc = xc if xc is not None else domains[0].lx / 2
    yc = yc if yc is not None else domains[0].ly / 2
    for ind, domain in enumerate(domains):
        state.u[ind] =  100*np.sin(np.pi*domain.xx/(domain.lx))**2*np.sin(2*np.pi*domain.yy/(domain.ly))
        state.v[ind] = -100*np.sin(np.pi*domain.yy/(domain.ly))**2*np.sin(2*np.pi*domain.xx/(domain.lx))

        dx = (domain.xx - 0.5 * xc) / domain.lx
        dy = (domain.yy - 1.0 * yc) / domain.ly
        state.h[ind] = h_mean + np.exp(-sigma_x * dx ** 2 - sigma_y * dy ** 2)

        dx = (domain.xx - 1.5 * xc) / domain.lx
        dy = (domain.yy - 1.0 * yc) / domain.ly
        state.h[ind] += np.exp(-sigma_x * dx ** 2 - sigma_y * dy ** 2)
    return state


def stream_geostrophic_balance(domains, pcori, g, h_mean):

    state = State.zeros(domains)
    u0 = 50.0

    for ind, domain in enumerate(domains):

        xxn = 2 * np.pi * domain.xx / domain.lx
        yyn = 2 * np.pi * domain.yy / domain.ly

        Ax = domain.lx / np.sqrt(domain.lx ** 2 + domain.ly ** 2)
        Ay = domain.ly / np.sqrt(domain.lx ** 2 + domain.ly ** 2)

        state.u[ind] = u0 * Ax * ( np.cos(xxn) * np.cos(yyn) - np.sin(xxn) * np.sin(yyn))
        state.v[ind] = u0 * Ay * (-np.cos(xxn) * np.cos(yyn) + np.sin(xxn) * np.sin(yyn))

        state.h[ind] = h_mean - u0 * pcori / g * domain.ly * Ax / 2.0 / np.pi * \
                      ( np.sin(xxn) * np.cos(yyn) + np.sin(yyn) * np.cos(xxn))

    return state

def eddy_geostrophic_balance(domains, pcori, g, h_mean, scale_h, scale_sigma, xc=None, yc=None):
    state = State.zeros(domains)
    xc = xc if xc is not None else domains[0].lx / 2
    yc = yc if yc is not None else domains[0].ly / 2

    for ind, domain in enumerate(domains):
        
        dx = (domain.xx - xc) / scale_sigma
        dy = (domain.yy - yc) / scale_sigma
        state.h[ind] = h_mean - (h_mean * np.exp( - dx ** 2 - dy ** 2)) * scale_h
        state.v[ind] =   g / pcori * h_mean * scale_h * np.exp( - dx ** 2 - dy ** 2) * 2 * dx / scale_sigma
        state.u[ind] = - g / pcori * h_mean * scale_h * np.exp( - dx ** 2 - dy ** 2) * 2 * dy / scale_sigma

    return state


def eddy_full_geostrophic_balance(domains, pcori, g, h_mean, scale_h, scale_sigma, xc=None, yc=None):
    state = State.zeros(domains)
    xc = xc if xc is not None else domains[0].lx / 2
    yc = yc if yc is not None else domains[0].ly / 2

    for ind, domain in enumerate(domains):

        r = np.sqrt((domain.xx - xc) ** 2 + (domain.yy - yc) ** 2)
        phi = np.zeros_like(r)
        for i in range(domain.nx):
            for j in range(domain.ny):
                phi[j,i] = math.atan2((domain.yy[j,i] - np.mean(domain.y)), (domain.xx[j,i] - np.mean(domain.x)))

        dhdr = h_mean * np.exp( - (r / scale_sigma) ** 2) * scale_h * 2.0 * r / scale_sigma ** 2
        vtan = (- r * pcori + np.sqrt((r * pcori) ** 2 + 4.0 * g * r * dhdr)) / 2.0

        state.h[ind] = h_mean - (h_mean * np.exp( - (r / scale_sigma) ** 2)) * scale_h
        state.u[ind] = - vtan * np.cos(np.pi / 2.0 - phi)
        state.v[ind] =   vtan * np.sin(np.pi / 2.0 - phi) 

    return state


def eddy_and_velocity_geostrophic_balance(domains, pcori, g, h_mean, scale_h, scale_sigma, xc=None, yc=None):
    state = State.zeros(domains)
    xc = xc if xc is not None else domains[0].lx / 2
    yc = yc if yc is not None else domains[0].ly / 2

    for ind, domain in enumerate(domains):

        r = np.sqrt((domain.xx - xc) ** 2 + (domain.yy - yc) ** 2)
        phi = np.zeros_like(r)
        for i in range(domain.nx):
            for j in range(domain.ny):
                phi[j,i] = math.atan2((domain.yy[j,i] - np.mean(domain.y)), (domain.xx[j,i] - np.mean(domain.x)))

        dhdr = h_mean * np.exp( - (r / scale_sigma) ** 2) * scale_h * 2.0 * r / scale_sigma ** 2
        vtan = (- r * pcori + np.sqrt((r * pcori) ** 2 + 4.0 * g * r * dhdr)) / 2.0

        h0 = pcori  / g * np.cos((domain.yy - 10.0 ** 7) / (6371.22 * 1000.0)) * 10.0 * 6371.22 * 1000.0
        u0 = np.sin((domain.yy - 10.0 ** 7) / (6371.22 * 1000.0)) * 10.0
        v0 = 0

        state.h[ind] = h_mean - (h_mean * np.exp( - (r / scale_sigma) ** 2)) * scale_h + h0
        state.u[ind] = - vtan * np.cos(np.pi / 2.0 - phi) + u0
        state.v[ind] =   vtan * np.sin(np.pi / 2.0 - phi) + v0

    return state


def barotropic_instability(domains, pcori, g, h_mean):

    state = State.zeros(domains)

    y = sm.Symbol('x')
    s = sm.Symbol('s')
    ly = sm.Symbol('ly')
    expr = sm.lambdify(s, sm.integrate((sm.sin(2 * sm.pi * y / ly)) ** 81,
                                       (y, 0, s)).subs(ly, domains[0].ly), "numpy")

    u0 = 50.0

    for ind, domain in enumerate(domains):
        state.u[ind] = u0 * (np.sin(2*np.pi*domain.yy/domain.ly))**81
        h_prof = u0 * expr(domain.y)

        def dist(xc, yc):
            return ((domain.xx - xc) / domain.lx) ** 2 + ((domain.yy - yc) / domain.ly) ** 2

        xc1 = 0.85 * domain.lx
        yc1 = 0.75 * domain.ly

        xc2 = 0.15 * domain.lx
        yc2 = 0.25 * domain.ly

        k = 1000

        h_pert = 0.01 * h_mean * (np.exp(-k * dist(xc1, yc1)) + np.exp(-k * dist(xc2, yc2)))

        for i in range(0, domain.nx + 1):
            state.h[ind][:, i] = h_mean - pcori / g * h_prof + h_pert[:, i]

    return state
