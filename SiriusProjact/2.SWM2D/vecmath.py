import numpy as np
from scipy import linalg


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


def calc_dot_prod(f, g, domain, weights=get_standard_weights):
    return (weights(domain) * f * g).sum()


def calc_l2_norm(f, domain, weights=get_standard_weights):
    return np.sqrt(calc_dot_prod(f, f, domain, weights))


def calc_c_norm(f, domain):
    return abs(f).max()


def calc_mass(f, domain, weights=get_standard_weights):
    return calc_dot_prod(f, np.ones((domain.ny + 1, domain.nx + 1)), domain, weights)
