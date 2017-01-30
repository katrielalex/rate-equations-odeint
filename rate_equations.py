#!/usr/bin/env python
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import argparse
import logging
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Chemistry magic
X = 0.5

# Reaction rates are b, c, d
rates = (0.1, 0.2, 0.3)

# Take statespace as: (B,C,D,E,F,G)
initial_conditions = (0.1, 0, 0, 1, 1.5, 0)

# Time points to solve for concentrations; let's use 100 steps from 0 to 10
time_points = np.linspace(0, 10, 100)


# Given particular reaction rates, solve for the concentrations of each reagent over time.
def concentrations(rates):
    def d_by_dt(x, t, b, c, d):
        B, C, D, E, F, G = x
        B_prime = -b * B * E
        G_prime = d * D * E
        F_prime = -c * C * (E + C + X)
        C_prime = -B_prime + G_prime - F_prime
        D_prime = -F_prime - G_prime
        E_prime = -B_prime - G_prime
        return (B_prime, C_prime, D_prime, E_prime, F_prime, G_prime)

    return odeint(d_by_dt, initial_conditions, time_points, rates)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Do some rate equation computations")
        args = parser.parse_args()

        logging.basicConfig(level=logging.DEBUG)

        solution = pd.DataFrame(concentrations(rates), columns=list('BCDEFG'))
        solution.plot()
        plt.show()
