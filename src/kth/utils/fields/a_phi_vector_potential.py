import numpy as np
import scipy.special as special
from scipy.integrate import simpson


def a_phi_hankel_v10(H_current, rho_z_in, phi, z, lambda_arr, d_0):
    # This function calculates the vector potential a_phi with the results from the Hankel transformation of
    # the neutral sheet current.

    sheet_thickness = d_0

    integrand = (
        H_current
        * special.j1(lambda_arr * rho_z_in)
        * np.exp(-lambda_arr * np.sqrt(z**2 + sheet_thickness**2))
    )

    result_a_phi_hankel = simpson(integrand, x=lambda_arr)

    return result_a_phi_hankel
