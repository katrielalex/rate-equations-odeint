#!/usr/bin/env python

"""rate_equations.py: Solves a particular chemical rate equation ODE."""

__author__      = "Max Shepherd"
__copyright__   = "Max Shepherd, 2009"

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import argparse
import logging
import numpy as np
import os
import pandas as pd
import redirect_std
import scipy.integrate
import scipy.optimize
import sys
import warnings

# Chemistry magic
X = 0.5

# Take statespace as: (B,C,D,E,F,G)
initial_conditions = (0.1, 0, 0, 1, 1.5, 0)

# Time points to solve for concentrations
t_0 = 0
t_max = 100
dt = 0.1


# Given particular reaction rates, solve for the concentrations of each reagent over time.
def concentrations(rates):
    def d_by_dt(t, y, b, c, d):
        B, C, D, E, F, G = y
        B_prime = -b * B * E
        G_prime = d * D * E
        F_prime = -c * C * (E + C + X)
        C_prime = -B_prime + G_prime - F_prime
        D_prime = -F_prime - G_prime
        E_prime = -B_prime - G_prime
        return [B_prime, C_prime, D_prime, E_prime, F_prime, G_prime]

    r = scipy.integrate.ode(d_by_dt)
    # vode, method={adams (non-stiff), bdf (stiff)}; lsoda
    # r.set_integrator('vode', method='adams')  # bdf if stiff, also try lsoda integrator
    r.set_integrator('lsoda')
    r.set_initial_value(initial_conditions, 0)  # y0, t0
    r.set_f_params(*rates)

    height = int((t_max - t_0) / dt)
    solution = np.zeros((height, len(initial_conditions)))

    # The solver will warn noisily if it looks like it will fail, both to stdout and with
    # warnings. We are checking for success on each step and abort if it failed, so we're OK with
    # the warnings and just want to squelch them. Hence the next two lines.
    with redirect_std.stdout_redirected(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in xrange(height):
            next = r.integrate(r.t + dt)
            if r.successful():
                solution[i] = next
            else:
                # If the solver crashed, don't keep pushing---just quit and stick with zeros.
                break

    return solution


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Do some rate equation computations")
    parser.add_argument("--rates", help="comma-separated coefficients for b, c, d")
    args = parser.parse_args()

    logging.info("Using fake experimental data")
    real_data = concentrations((0.64166667, 0.01, 0.24666667))

    logging.info("Minimising L2 distance between approximated values and real data")
    initial_rates = tuple(float(rate) for rate in args.rates.split(",")) if args.rates else (0.5, 0.3, 0.2)
    score = lambda rates: np.linalg.norm(concentrations(rates) - real_data)
    score = lambda rates: 6000 - np.count_nonzero(concentrations(rates))
    result = scipy.optimize.minimize(score, initial_rates, method='Nelder-Mead', options=dict(disp=True))

    logging.info("Plotting")
    # rates = result.x
    # solution = pd.DataFrame(concentrations(rates), columns=list('BCDEFG'))
    # solution.plot()
    # plt.show()

    logging.info("Done")
    import ipdb; ipdb.set_trace()


# Wrap solver using scipy.integrate.ode and do the timesteps manually until it blows up
# Take the L2 norm(s) between the results and the actual data
# Do some monte carlo for the parameters to get a good idea of what the state space is like

