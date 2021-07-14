
"""
Generate a given number of peaks (simulated in [0, 1]x[0, 1])
"""

import numpy as np


def generate_peaks(n_peaks=10, mu = 0, sigma = 1):
    peaks = mu + np.random.randn(2, n_peaks)*sigma
    return peaks

def generate_shifts(n_peaks):
    # per ogni punto vanno generati nuovi punti nelle vicinanze
    # possiamo generare un vettore degli spostamenti
    shifts = 0 + np.random.randn(2, n_peaks) * 0.2
    return shifts


