def explicit_euler(state, operator, dt, domain):
    return state + dt * operator.calc_rhs(state, domain)


def rk4(state, operator, dt, domain):
    k1 = operator.calc_rhs(state, domain)
    k2 = operator.calc_rhs(state + dt / 2 * k1, domain)
    k3 = operator.calc_rhs(state + dt / 2 * k2, domain)
    k4 = operator.calc_rhs(state + dt * k3, domain)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
