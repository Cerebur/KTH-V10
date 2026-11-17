import numpy as np


def current_profile_ringcurrent_v10(rho, control_params):
    # this function calculates the current profile of the ring current

    d = control_params["d"]
    e = control_params["e"]
    f = control_params["f"]

    current = -(d / f * np.sqrt(2 * np.pi)) * np.exp(-((rho - e) ** 2) / (2 * f**2))

    return current
