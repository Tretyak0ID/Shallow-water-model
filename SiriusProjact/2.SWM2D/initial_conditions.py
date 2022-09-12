from state import State
import numpy as np
import sympy as sm


def gaussian_hill(domain, h_mean, k_x=50.0, k_y=50.0):
    state = State.zeros(domain.nx, domain.ny)
    dx = (domain.xx - np.mean(domain.x)) / domain.x[-1]
    dy = (domain.yy - np.mean(domain.y)) / domain.y[-1]
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state

def gaussian_hill_linexy(domain, h_mean, k_x=50.0, k_y=50.0, k_u = 100, k_v = 100):
    state = State.zeros(domain.nx, domain.ny)
    state.u = np.ones((domain.ny + 1, domain.nx + 1))*k_u
    state.v = np.ones((domain.ny + 1, domain.nx + 1))*k_v
    dx = (domain.xx - np.mean(domain.x)) / domain.x[-1]
    dy = (domain.yy - np.mean(domain.y)) / domain.y[-1]
    state.u = state.u
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state

def gaussian_hill_linex(domain, h_mean, k_x=50.0, k_y=50.0, k_u = 100):
    state = State.zeros(domain.nx, domain.ny)
    state.u = np.ones((domain.ny + 1, domain.nx + 1))*k_u
    dx = (domain.xx - np.mean(domain.x)) / domain.x[-1]
    dy = (domain.yy - np.mean(domain.y)) / domain.y[-1]
    state.u = state.u
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state

def gaussian_hill_liney(domain, h_mean, k_x=50.0, k_y=50.0, k_v = 100):
    state = State.zeros(domain.nx, domain.ny)
    state.v = np.ones((domain.ny + 1, domain.nx + 1))*k_v
    dx = (domain.xx - np.mean(domain.x)) / domain.x[-1]
    dy = (domain.yy - np.mean(domain.y)) / domain.y[-1]
    state.u = state.u
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state

def gaussian_hill_circle(domain, h_mean, k_x=50.0, k_y=50.0):
    state = State.zeros(domain.nx, domain.ny)

    for i in range(domain.nx+1):
        for j in range(domain.ny+1):
            state.u[j,i] =  100*np.sin(np.pi*domain.x[i]/(2 * np.pi * 6371.22 * 1000.0))*np.cos(np.pi*domain.y[j]/(2 * np.pi * 6371.22 * 1000.0))
            state.v[j,i] = -100*np.cos(np.pi*domain.x[i]/(2 * np.pi * 6371.22 * 1000.0))*np.sin(np.pi*domain.y[j]/(2 * np.pi * 6371.22 * 1000.0))

    dx = (domain.xx - np.mean(domain.x)/2) / domain.x[-1]
    dy = (domain.yy - np.mean(domain.y)) / domain.y[-1]
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state

def gaussian_hill_2circle(domain, h_mean, k_x=50.0, k_y=50.0):
    state = State.zeros(domain.nx, domain.ny)

    for i in range(domain.nx+1):
        for j in range(domain.ny+1):
            state.u[j,i] =  100*np.sin(np.pi*domain.x[i]/(2 * np.pi * 6371.22 * 1000.0)*2)*np.cos(np.pi*domain.y[j]/(2 * np.pi * 6371.22 * 1000.0))
            state.v[j,i] = -100*np.cos(np.pi*domain.x[i]/(2 * np.pi * 6371.22 * 1000.0)*2)*np.sin(np.pi*domain.y[j]/(2 * np.pi * 6371.22 * 1000.0))

    dx = (domain.xx - np.mean(domain.x)) / domain.x[-1]
    dy = (domain.yy - np.mean(domain.y)) / domain.y[-1]
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state


def barotropic_instability(domain, pcori, g, h_mean):

    state = State.zeros(domain.nx, domain.ny)

    y = sm.Symbol('x')
    s = sm.Symbol('s')
    ly = sm.Symbol('ly')
    expr = sm.integrate((sm.sin(2*sm.pi*y/ly)) ** 81, (y, 0, s)).subs(ly, domain.ye)

    u0 = 50.0

    state.u = u0 * (np.sin(2*np.pi*domain.yy/domain.ye))**81
    h_prof = np.zeros_like(domain.y)
    for j, y in enumerate(domain.y):
        h_prof[j] = u0*float(expr.subs(s, y))

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