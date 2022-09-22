import numpy as np
from state import GridField


def get_standard_weights(domain):
    weights_x = np.ones(domain.nx + 1) * domain.dx
    weights_x[0] = weights_x[-1] = 0.5 * domain.dx

    weights_y = np.ones(domain.ny + 1) * domain.dy
    weights_y[0] = weights_y[-1] = 0.5 * domain.dy

    return weights_y.reshape(domain.ny + 1, 1) @ weights_x.reshape(1, domain.nx + 1)


def get_sbp42_weights(domain):
    weights_x = np.ones(domain.nx + 1) * domain.dx
    weights_x[0:4] = weights_x[-1:-5:-1] = np.array((17.0, 59.0, 43.0, 49.0)) / 48.0 * domain.dx

    weights_y = np.ones(domain.ny + 1) * domain.dy
    weights_y[0:4] = weights_y[-1:-5:-1] = np.array((17.0, 59.0, 43.0, 49.0)) / 48.0 * domain.dy

    return weights_y.reshape(domain.ny + 1, 1) @ weights_x.reshape(1, domain.nx + 1)


def calc_dot_prod(f, g, domains, weights=get_standard_weights):
    return sum([(weights(domains[k]) * f[k] * g[k]).sum() for k in range(len(domains))])


def calc_l2_norm(f, domains, weights=get_standard_weights):
    return np.sqrt(calc_dot_prod(f, f, domains, weights))


def calc_c_norm(f, domains):
    return max([abs(f[k]).max() for k in range(len(domains))])


def calc_mass(f, domains, weights=get_standard_weights):
    return calc_dot_prod(f, GridField.ones(domains), domains, weights)
