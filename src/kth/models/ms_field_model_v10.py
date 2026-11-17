# -*- coding: utf-8 -*-
"""Created on 17 Nov 2026 15:52.

@author: kristin
Modified: Jonathan
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from kth.utils.fields.cf_field_v10 import cf_field_v10
from kth.utils.fields.internal_field_v10 import internal_field_v10
from kth.utils.fields.ringcurrent_bfield import tail_field_ringcurrent_v10
from kth.utils.fields.tail_bfield import tail_field_ns_bs_v10

FloatArray = npt.NDArray[np.floating[Any]]


def model_field_v10(
    x_msm: FloatArray,
    y_msm: FloatArray,
    z_msm: FloatArray,
    di: FloatArray,
    aberration: FloatArray,
    dipole: bool,
    neutralsheet: bool,
    ringcurrent: bool,
    internal: bool,
    external: bool,
    control_params: dict,
    R_SS: float,
) -> FloatArray:
    """Models a magnetospheric field at a given position.

    One can optionally enable or disable contributions from the neutralsheet current, ring current, the internal dipole and corresponding chapman-ferraro currents

    Types:
        FloatArray is a numpy array with float values

    Args:
        x_msm: x-position in msm coordinates in km
        y_msm: x-position in msm coordinates in km
        z_msm: z-position in msm coordinates in km
        di: Disturbance index (0-100)
        aberration: aberration angle in deg
        dipole: enables dipole field if True
        neutralsheet: enables neutralsheet current field if True
        ringcurrent: enables ringcurrent field if True
        internal: considers the internal contributions of the different fields if True
        external: considers the external contributions (i.e. chapman-ferraro current induced fields) of the different fields if True
        control_params: parameters used for the calculation of the different field contributions
        R_SS: subsolar standoff distance in km??

    Returns:
        magnetic field at specified location in order Bx, By, Bz
    """
    # This function calls the different modules (Dipole, Ring Current,
    # Neutral Sheet Current) and their shielding fields
    # (external fields ). These return the magnetic field of the corresponding
    #

    g10_int_ind = control_params["g10_int_ind"]
    kappa3 = control_params["kappa3"]
    g10_int = control_params["g10_int"]

    t_nsc = control_params["t_a"] + control_params["t_b"] * di
    # t_rc = control_params['t_a'] - control_params['t_b'] * di
    t_rc = 0.51

    n_points = x_msm.size

    # multiply with kappa
    x_msm_k = x_msm * control_params["kappa"]
    y_msm_k = y_msm * control_params["kappa"]
    z_msm_k = z_msm * control_params["kappa"]

    if n_points == 1:
        B_total = np.zeros([3, 1])
    else:
        B_total = np.zeros([3, n_points])

    ##################################################
    # calculate fields
    #################################################

    if dipole:
        if internal:
            B_int = kappa3 * internal_field_v10(
                x_msm_k, y_msm_k, z_msm_k, control_params
            )
            B_total = B_total + B_int

        if external:
            # This was intended to include any induced internal dipole fields.
            # The coefficient was based on empirical work.

            induction_scale_fac = -0.0052631579 * (g10_int + g10_int_ind)
            B_cf_int = (
                kappa3
                * induction_scale_fac
                * cf_field_v10(
                    x_msm_k,
                    y_msm_k,
                    z_msm_k,
                    control_params["lin_coeff_int"],
                    control_params["p_i_int"],
                    control_params["q_i_int"],
                    control_params["x_shift_int"],
                )
            )
            # add
            B_total = B_total + B_cf_int

    if neutralsheet:
        if internal:
            z_offset = 0.0

            B_nsc = tail_field_ns_bs_v10(x_msm, y_msm, z_msm, z_offset)
            B_total = B_total + t_nsc * B_nsc

        if external:
            """
            #add image sheets at twice the asymptotic tail radius with the opposite current direction
            
            R_tail_asymptotic = R_SS * 2. / control_params['RPL']
            
            z_offset = 2. * R_tail_asymptotic
            B_image1_ns = -tail_field_ns_bs_v10(x_msm, y_msm, z_msm, z_offset)

            z_offset = -2. * R_tail_asymptotic
            B_image2_ns = -tail_field_ns_bs_v10(x_msm, y_msm, z_msm, z_offset)
            """

            # calculate the Chapman-Ferraro currents (shielding)

            B_cf_ns = 0.05 * cf_field_v10(
                x_msm_k,
                y_msm_k,
                z_msm_k,
                control_params["lin_coeff_disk"],
                control_params["p_i_disk"],
                control_params["q_i_disk"],
                control_params["x_shift_disk"],
            )

            # add
            B_total = B_total + (t_nsc * B_cf_ns)  # + B_image1_ns + B_image2_ns

    if ringcurrent:
        if internal:
            scale_fac = 5
            B_rc = scale_fac * tail_field_ringcurrent_v10(
                x_msm_k, y_msm_k, z_msm_k, di, control_params
            )
            B_total = B_total + t_rc * B_rc

        if external:
            # the ring current field is so small, so that a shielding is not necessarily required. This
            # is why the cf_disk part is deactivated (at the moment).

            B_total = B_total

    ##################################################
    # rotate magnetic field back to MSO base
    #################################################

    if x_msm.size == 1:
        bx = B_total[0]
        by = B_total[1]
        bz = B_total[2]
    else:
        bx = B_total[0, :]
        by = B_total[1, :]
        bz = B_total[2, :]

    b_x = bx * np.cos(aberration) - by * np.sin(aberration)
    b_y = bx * np.sin(aberration) + by * np.cos(aberration)
    b_z = bz

    return np.array([b_x, b_y, b_z])
