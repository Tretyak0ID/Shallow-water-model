import numpy as np
from   scipy import linalg


def get_standard_weights(domain):
    weights_x = np.ones(domain.nx + 1) * domain.dx
    weights_x[0] = weights_x[-1] = 0.5 * domain.dx

    weights_y = np.ones(domain.ny + 1) * domain.dy
    weights_y[0] = weights_y[-1] = 0.5 * domain.dy

    return weights_y.reshape(domain.ny + 1, 1) @ weights_x.reshape(1, domain.nx + 1)


def calc_dot_prod(f, g, domain):
    return (get_standard_weights(domain) * f * g).sum()


def calc_l2_norm(f, domain):
    return np.sqrt(calc_dot_prod(f, f, domain))


def calc_c_norm(f, domain):
    return abs(f).max()


def calc_mass(f, domain):
    return calc_dot_prod(f, np.ones((domain.ny + 1, domain.nx + 1)), domain)