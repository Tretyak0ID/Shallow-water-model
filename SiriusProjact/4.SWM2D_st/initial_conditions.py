from state import State
import numpy as np
import sympy as sm


def gaussian_hill(domain, h_mean, k_x=50.0, k_y=50.0):
    state = State.zeros(domain.nx, domain.ny)
    dx = (domain.xx - 0.8 * np.mean(domain.x)) / domain.x
    dy = (domain.yy - 0.8 * np.mean(domain.y)) / domain.y
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state


def geostrophic_balance(domain, pcori, g, h_mean):

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

