from state import State
import numpy as np
import sympy as sm
import math


def gaussian_hill(domain, h_mean, k_x=50.0, k_y=50.0):
    state = State.zeros(domain.nx, domain.ny)
    dx = (domain.xx - 0.8 * np.mean(domain.x)) / domain.nx
    dy = (domain.yy - 0.8 * np.mean(domain.y)) / domain.ny
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state


def stream_geostrophic_balance(domain, pcori, g, h_mean):

    state = State.zeros(domain.nx, domain.ny)
    u0 = 50.0

    xxn = 2 * np.pi * domain.xx / domain.xe
    yyn = 2 * np.pi * domain.yy / domain.ye

    Ax = domain.xe / np.sqrt(domain.xe ** 2 + domain.ye ** 2)
    Ay = domain.ye / np.sqrt(domain.xe ** 2 + domain.ye ** 2)

    state.u = u0 * Ax * ( np.cos(xxn) * np.cos(yyn) - np.sin(xxn) * np.sin(yyn))
    state.v = u0 * Ay * (-np.cos(xxn) * np.cos(yyn) + np.sin(xxn) * np.sin(yyn))

    state.h = h_mean - u0 * pcori / g * domain.ye * Ax / 2.0 / np.pi * \
              ( np.sin(xxn) * np.cos(yyn) + np.sin(yyn) * np.cos(xxn))

    return state


def eddy_geostrophic_balance(domain, pcori, g, h_mean, scale_h, scale_sigma):
    state = State.zeros(domain.nx, domain.ny)
    dx = (domain.xx - np.mean(domain.x)) / scale_sigma
    dy = (domain.yy - np.mean(domain.y)) / scale_sigma
    state.h = h_mean - (h_mean * np.exp( - dx ** 2 - dy ** 2)) * scale_h
    state.v =   g / pcori * h_mean * scale_h * np.exp( - dx ** 2 - dy ** 2) * 2 * dx / scale_sigma
    state.u = - g / pcori * h_mean * scale_h * np.exp( - dx ** 2 - dy ** 2) * 2 * dy / scale_sigma
    return state

def eddy_full_geostrophic_balance(domain, pcori, g, h_mean, scale_h, scale_sigma):
    state = State.zeros(domain.nx, domain.ny)
    r = np.sqrt((domain.xx - np.mean(domain.x)) ** 2 + (domain.yy - np.mean(domain.y)) ** 2)
    phi = np.zeros_like(r)
    for i in range(domain.nx):
        for j in range(domain.ny):
            phi[j,i] = math.atan2((domain.yy[j,i] - np.mean(domain.y)), (domain.xx[j,i] - np.mean(domain.x)))
    dhdr = h_mean * np.exp( - (r / scale_sigma) ** 2) * scale_h * 2.0 * r / scale_sigma ** 2
    vtan = (- r * pcori + np.sqrt((r * pcori) ** 2 + 4.0 * g * r * dhdr)) / 2.0

    state.h = h_mean - (h_mean * np.exp( - (r / scale_sigma) ** 2)) * scale_h
    state.u = - vtan * np.cos(np.pi / 2.0 - phi)
    state.v =   vtan * np.sin(np.pi / 2.0 - phi) 
    return state

def eddy_and_velocity_geostrophic_balance(domain, pcori, g, h_mean, scale_h, scale_sigma):
    state = State.zeros(domain.nx, domain.ny)

    r = np.sqrt((domain.xx - np.mean(domain.x) / 2) ** 2 + (domain.yy - np.mean(domain.y)) ** 2)
    phi = np.zeros_like(r)
    for i in range(domain.nx):
        for j in range(domain.ny):
            phi[j,i] = math.atan2((domain.yy[j,i] - np.mean(domain.y)), (domain.xx[j,i] - np.mean(domain.x) / 2))

    dhdr = h_mean * np.exp( - (r / scale_sigma) ** 2) * scale_h * 2.0 * r / scale_sigma ** 2
    vtan = (- r * pcori + np.sqrt((r * pcori) ** 2 + 4.0 * g * r * dhdr)) / 2.0

    h0 = pcori  / g * np.cos((domain.yy - 10.0 ** 7) / (6371.22 * 1000.0)) * 10.0 * 6371.22 * 1000.0
    u0 = np.sin((domain.yy - 10.0 ** 7) / (6371.22 * 1000.0)) * 10.0
    v0 = 0

    state.h = h_mean - (h_mean * np.exp( - (r / scale_sigma) ** 2)) * scale_h + h0
    state.u = - vtan * np.cos(np.pi / 2.0 - phi) + u0
    state.v =   vtan * np.sin(np.pi / 2.0 - phi) + v0
    return state


def barotropic_instability(domain, pcori, g, h_mean):

    state = State.zeros(domain.nx, domain.ny)

    y = sm.Symbol('x')
    s = sm.Symbol('s')
    ly = sm.Symbol('ly')
    expr = sm.lambdify(s, sm.integrate((sm.sin(2*sm.pi*y/ly)) ** 81, (y, 0, s)).subs(ly, domain.ye), "numpy")

    u0 = 50.0

    state.u = u0 * (np.sin(2*np.pi*domain.yy/domain.ye))**81
    h_prof = u0 * expr(domain.y)

    def dist(xc, yc):
        return ((domain.xx - xc)/domain.xe)**2 + ((domain.yy - yc)/domain.ye)**2

    xc1 = 0.85 * domain.xe
    yc1 = 0.75 * domain.ye

    xc2 = 0.15 * domain.xe
    yc2 = 0.25 * domain.ye

    k = 1000

    h_pert = 0.01 * h_mean * (np.exp(-k * dist(xc1, yc1)) + np.exp(-k * dist(xc2, yc2)))

    for i in range(0, domain.nx+1):
        state.h[:, i] = h_mean - pcori / g * h_prof

    state.h = state.h + h_pert

    return state
