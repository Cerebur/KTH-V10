import numpy as np
import scipy.special as special
from scipy.integrate import simpson

from ..currents.ringcurrent_profile import current_profile_ringcurrent_v10
from .a_phi_vector_potential import a_phi_hankel_v10


def tail_field_ringcurrent_v10(x_msm, y_msm, z_msm, di, control_params):
    # this function calculates the magnetic field of the eastward flowing ring shaped current close to the planet

    rho = np.sqrt(x_msm**2 + y_msm**2)
    phi = np.arctan2(y_msm, x_msm)

    d_0 = control_params["d_0"]

    mu_0 = 1.0
    steps = 100
    rho_min = 0.5  # these values are adapted for the specific current profile for the ring current
    rho_max = 2
    h_steps = 100  # This value is from experience. When you change the current profile this should be checked again for sufficient convergence.

    rho_hankel = np.arange(steps) / float(steps - 1.0) * (rho_max - rho_min) + rho_min

    current = current_profile_ringcurrent_v10(rho_hankel, control_params)

    lambda_max = 20  # std value
    lambda_min = 10 ** (-1)  # std value

    lambda_out = 10 ** (
        np.divide(range(h_steps), (float(h_steps) - 1))
        * (np.log10(lambda_max) - np.log10(lambda_min))
        + np.log10(lambda_min)
    )

    result_hankel_trafo = np.zeros(h_steps)
    for i in range(h_steps):
        # special.j1 = Bessel function of the first kind of order 1
        result_hankel_trafo[i] = simpson(
            special.j1(lambda_out[i] * rho_hankel) * current * rho_hankel,
            x=rho_hankel,
            axis=-1,
        )

    H_current = mu_0 / 2.0 * result_hankel_trafo

    ###############################################################

    n_vec = len(np.atleast_1d(x_msm))
    b_rc_x = np.zeros(n_vec)
    b_rc_y = np.zeros(n_vec)
    b_rc_z = np.zeros(n_vec)
    b_rc_rho = np.zeros(n_vec)

    for i in range(n_vec):
        a_phi = a_phi_hankel_v10(H_current, rho[i], phi[i], z_msm[i], lambda_out, d_0)

        # numerically approximate the derivatives
        # MB:Numerical integration probably used to ensure nuerical stability,
        # as the analytical derivatives of the Hankel transform might be difficult to calculate and could lead to numerical issues.
        # And with a numerical implementation the kernal could be changed without having to change the analytical derivatives.

        delta_z = 10 ** (-5)

        d_a_phi_d_z = (
            a_phi_hankel_v10(
                H_current, rho[i], phi[i], z_msm[i] + delta_z, lambda_out, d_0
            )
            - a_phi_hankel_v10(
                H_current, rho[i], phi[i], z_msm[i] - delta_z, lambda_out, d_0
            )
        ) / (2 * delta_z)

        delta_rho = 10 ** (-5)
        d_a_phi_d_rho = (
            a_phi_hankel_v10(
                H_current, rho[i] + delta_rho, phi[i], z_msm[i], lambda_out, d_0
            )
            - a_phi_hankel_v10(
                H_current, rho[i] - delta_rho, phi[i], z_msm[i], lambda_out, d_0
            )
        ) / (2 * delta_rho)

        # MB: Calculating curl in cylindrical coordinates.
        b_rc_rho[i] = -d_a_phi_d_z

        # MB: Here there seems to be singularity control, so that a_phi/rho does not blow up near
        # the axis of symmetry
        if rho[i] <= 10 ** (-4):
            b_rc_z[i] = 1.0 + d_a_phi_d_rho

        # MB: Calculating curl in cylindrical coordinates.
        else:
            b_rc_z[i] = a_phi / rho[i] + d_a_phi_d_rho

        # rotate back to cartesian
        b_rc_x[i] = b_rc_rho[i] * np.cos(phi[i])
        b_rc_y[i] = b_rc_rho[i] * np.sin(phi[i])

    # z_sheet_thickness =   1 * np.exp(-0.5*((z_msm)/2)**2 )
    # rho_sheet_thickness =  1 * np.exp(-0.5*((rho-1)/2)**2 )
    z_sheet_thickness = 1

    b_rc_z = b_rc_z * z_sheet_thickness
    b_rc_x = b_rc_x * z_sheet_thickness
    b_rc_y = b_rc_y * z_sheet_thickness

    return np.array([b_rc_x, b_rc_y, b_rc_z])
