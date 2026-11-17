import numpy as np


def estimate_aberration_angle(r_hel, v_sw=400):
    au = 1.496e8  # km
    semimajor = 57.91e6  # km
    Gv = 6.674e-11  # N m^2 / kg^-2
    Smass = 1.989e30  # kg
    # v_sw = 400
    nu = np.sqrt(Gv * Smass * (2 / (r_hel * au) - 1 / semimajor) * 1e-9)
    aberration_angle = np.arctan2(nu, v_sw)

    return np.round(aberration_angle / 2 / np.pi * 360, 4)
