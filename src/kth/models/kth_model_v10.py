# -*- coding: utf-8 -*-
"""Created on 17 Nov 2026 15:52.

@author: kristin
Modified: Jonathan
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from kth.models.ms_field_model_v10 import model_field_v10
from kth.utils.mp_dist_shue import shue_mp_calc_r_mp

FloatArray = npt.NDArray[np.floating[Any]]
ScalarOrArray = float | FloatArray


# Directory containing the currently executed Python file
script_dir = Path(__file__).resolve().parent

# Directory pointing to the default parameters
default_parameter_dir = script_dir.parent / "parameters"

# Create temp/ if it doesn't exist
temp_dir = script_dir / "temp"
temp_dir.mkdir(exist_ok=True)


def kth_model_for_mercury_v10(
    x_mso: ScalarOrArray,
    y_mso: ScalarOrArray,
    z_mso: ScalarOrArray,
    r_hel: ScalarOrArray,
    di: ScalarOrArray,
    aberration: ScalarOrArray,
    *,  # adding an * makes sure that the keyword switches must be passed explicitly instead of simply by order
    control_param_path=default_parameter_dir / "control_params_v10.json",
    fit_param_path=default_parameter_dir / "kth_own_cf_fit_parameters_v10.dat",
    imf_bx=0,
    imf_by=0,
    imf_bz=0,
    dipole=True,
    neutralsheet=True,
    ringcurrent=True,
    internal=True,
    external=True,
) -> npt.NDArray:
    """Calculate the KTH22 magnetic field model for Mercury.

    Calculates the magnetic field at specified locations using MSO coordinates,
    heliocentric distance, disturbance index, and aberration angle. The model
    uses the specified control and fitting parameters and allows individual
    magnetic field contributions to be enabled or disabled.

    Types:
        ScalarOrArray: Either a single floating-point value or a NumPy array
                        of floating-point values. All floats are internally
                        converted to np.float64

    Args:
        x_mso: X coordinates in MSO coordinates, in km.
        y_mso: Y coordinates in MSO coordinates, in km??.
        z_mso: Z coordinates in MSO coordinates, in km??.
        r_hel: Heliocentric distance, in AU.
        di: Disturbance index values.
        aberration: Aberration angle, in degrees.
        control_param_path: Path to the control parameter file.
        fit_param_path: Path to the fitting parameter file.
        imf_bx: IMF Bx component. Not yet implemented.
        imf_by: IMF By component. Not yet implemented.
        imf_bz: IMF Bz component. Not yet implemented.
        dipole: If True, include the planetary dipole field.
        neutralsheet: If True, include the neutral sheet contribution.
        ringcurrent: If True, include the ring current contribution.
        internal: If True, include internal magnetospheric field contributions.
        external: If True, include external Chapman-Ferraro field contributions.

    Returns:
        np.ndarray: Magnetic field in MSO coordinates, in nT??, with
            shape (3,N)??. The components are ordered as Bx, By, and Bz.
    """
    print_feedback = True  # if True: print output with number of positions which
    # are inside the planet or outside the magnetosphere
    # if False: No print output

    save_temp_data = True  # if True: saves temporary data files to a temp directory.

    # print('calculating KTH Magnetic Field Version 10b')

    # Convert the input parameters to the correct types.
    # Floats need to be converted to numpy.float values for the .size comparison to work
    x_mso, y_mso, z_mso, r_hel, di, aberration = (
        np.asarray(v, dtype=np.float64)
        for v in (x_mso, y_mso, z_mso, r_hel, di, aberration)
    )

    # Check that the input variables seem sensible
    nan_return = np.empty(3) * np.nan

    if x_mso.ndim > 1:
        print("Positions must be float or 1D arrays! Aborting... ")
        # if you would like to have the program abort, you could use
        # sys.exit()
        return nan_return

    input_length = x_mso.size  # moved the flattening after the dimension check.
    x_mso = np.array(x_mso).flatten()  # check would have passed trivially otherwise
    y_mso = np.array(y_mso).flatten()
    z_mso = np.array(z_mso).flatten()
    nan_return = np.empty((3, input_length)) * np.nan

    if x_mso.size != y_mso.size:
        print("Number of positions (x,y,z) do not match")
        return nan_return

    if x_mso.size != z_mso.size:
        print("Number of positions (x,y,z) do not match")
        return nan_return

    if x_mso.size != r_hel.size:
        print(
            "Length of heliocentric distance (r_hel) does not match. All input array must have the same size."
        )
        return nan_return

    if x_mso.size != di.size:
        print(
            "Length of disturbance index (di) does not match. All input array must have the same size."
        )
        return nan_return

    if x_mso.size != aberration.size:
        print(
            "Length of aberration does not match. All input array must have the same size."
        )
        return nan_return

    if (not internal) and (not external):
        print(
            'Internal and external field are both set "False". Set at least one True.'
        )
        return nan_return

    if (not dipole) and (not neutralsheet) and (not ringcurrent):
        print(
            'Dipole, neutralsheet and ringcurrent are set "False". Set at least one True.'
        )
        return nan_return

    ############################################################################################
    #        Reading control parameters and shielding coefficients from file                   #
    ############################################################################################

    with open(control_param_path, "r") as file:
        control_params = json.load(file)

    shielding_input = fit_param_path
    shielding_par_file = open(shielding_input, "r")
    shielding_params = np.loadtxt(shielding_par_file)
    shielding_par_file.close()

    # defining the lengths of the following arrays
    n_lin_int = shielding_params[0].astype(int)  # 16
    n_non_lin_int = shielding_params[1].astype(int)  # 4
    n_lin_neutralcurrent = shielding_params[2].astype(int)  # 0
    n_non_lin_neutralcurrent = shielding_params[3].astype(int)  # 0

    # length check
    length_check = (
        4
        + n_lin_int
        + 3 * n_non_lin_int
        + n_lin_neutralcurrent
        + 3 * n_non_lin_neutralcurrent
    )

    if len(shielding_params) != length_check:
        print(
            "Wrong shielding coefficients file length. Length has to be "
            + str(length_check)
            + ". Length is "
            + str(len(shielding_params))
            + " at the moment."
        )
        return nan_return

    # define coefficient arrays

    low = 4
    high = low + n_lin_int
    lin_coeff_int = shielding_params[low:high]
    control_params["lin_coeff_int"] = lin_coeff_int

    low = high
    high = low + n_lin_neutralcurrent
    lin_coeff_disk = shielding_params[low:high]
    control_params["lin_coeff_disk"] = lin_coeff_disk

    low = high
    high = low + n_non_lin_int
    p_i_int = shielding_params[low:high]
    control_params["p_i_int"] = p_i_int

    low = high
    high = low + n_non_lin_neutralcurrent
    p_i_disk = shielding_params[low:high]
    control_params["p_i_disk"] = p_i_disk

    low = high
    high = low + n_non_lin_int
    q_i_int = shielding_params[low:high]
    control_params["q_i_int"] = q_i_int

    low = high
    high = low + n_non_lin_neutralcurrent
    q_i_disk = shielding_params[low:high]
    control_params["q_i_disk"] = q_i_disk

    low = high
    high = low + n_non_lin_int
    x_shift_int = shielding_params[low:high]
    control_params["x_shift_int"] = x_shift_int

    low = high
    high = low + n_non_lin_neutralcurrent
    x_shift_disk = shielding_params[low:high]
    control_params["x_shift_disk"] = x_shift_disk

    #######################################################################################
    # DI-Scaling
    #######################################################################################
    if len(np.atleast_1d(di)) > 1:
        if any(t > 100 for t in di):
            print(
                "At least one element in DI is greater than 100. DI must be between 0 and 100. If you don't know the "
                "exact value, use 50."
            )
            return nan_return

        if any(t < 0 for t in di):
            print(
                "At least one element in DI is negative. DI must be between 0 and 100. If you don't know the exact value, use 50."
            )
            return nan_return
    elif len(np.atleast_1d(di)) == 1:
        if di < 0:
            print(
                "Disturbance index di must be between 0 and 100. If you don't know the exact value, use 50."
            )
            return nan_return
        if di > 100:
            print(
                "Disturbance index di must be between 0 and 100. If you don't know the exact value, use 50."
            )
            return nan_return

    f = control_params["f_a"] + (
        control_params["f_b"] * di
    )  # f is a factor for the scaling for R_SS (subsolar standoff distance)

    #######################################################################################
    # RMP-Scaling
    #######################################################################################

    if len(np.atleast_1d(r_hel)) > 1:
        if any(r_hel > 0.47):
            print(
                "Please use r_hel (heliocentric distance) in AU, not in km. r_hel should be between 0.3 and 0.47. "
            )
            return nan_return
        if any(r_hel < 0.3):
            print(
                "Please use r_hel (heliocentric distance) in AU, not in km. r_hel should be between 0.3 and 0.47. "
            )
            return nan_return
    if len(np.atleast_1d(r_hel)) == 1:
        if r_hel > 0.47:
            print(
                "Please use r_hel (heliocentric distance) in AU, not in km.r_hel should be between 0.3 and 0.47."
            )
            return nan_return
        if r_hel < 0.3:
            print(
                "Please use r_hel (heliocentric distance) in AU, not in km.r_hel should be between 0.3 and 0.47."
            )
            return nan_return

    R_SS = f * (r_hel ** (1 / 3)) * control_params["RPL"]

    # print('Achtung, line 244, R_SS2 eingeführt')
    # f2 = 2.14 + (-0.00368 * di)
    # R_SS = f2 * (r_hel ** (1 / 3)) * control_params['RPL']

    control_params["kappa"] = control_params["RMP"] / R_SS
    control_params["kappa3"] = (control_params["kappa"]) ** 3

    ################################################################################################################
    # Application of the offset: MSO->MSM coordinate system
    # Scaling to planetary radius
    # Scaling with heliocentric distance
    ################################################################################################################

    dipole_offset = (
        479 / control_params["RPL"]
    )  # value of offset by Anderson et al. (2012)
    x_msm_in = x_mso / control_params["RPL"]
    y_msm_in = y_mso / control_params["RPL"]
    z_msm_in = z_mso / control_params["RPL"] - dipole_offset
    aberration = (
        aberration / (360) * 2 * np.pi
    )  # from degree to rad, aberration angle is negative

    # application of the aberration to the coordinates
    x_msm = x_msm_in * np.cos(aberration) - y_msm_in * np.sin(aberration)
    y_msm = x_msm_in * np.sin(aberration) + y_msm_in * np.cos(aberration)
    z_msm = z_msm_in

    ################################################################################################################
    # Check for points lying outside the magnetopause. The magnetic field for these will be set to zero.
    # Also give a warning for calculation within the planet
    ################################################################################################################

    r_mso = np.sqrt((x_mso) ** 2 + (y_mso) ** 2 + (z_mso) ** 2)
    r_msm = np.sqrt((x_msm) ** 2 + (y_msm) ** 2 + (z_msm) ** 2)

    indices_inside_planet = np.where(r_mso < 0.99 * control_params["RPL"])

    if indices_inside_planet[0].size > 1:
        if print_feedback:
            print(
                "Warning: "
                + str(indices_inside_planet[0].size)
                + " point(s) are located inside the planet."
            )

    r_mp_check = (
        shue_mp_calc_r_mp(x_msm, y_msm, z_msm, R_SS, control_params["alpha"])
        / control_params["RPL"]
    )

    usable_indices = np.where(r_msm <= r_mp_check)

    if save_temp_data:
        np.savetxt(temp_dir / "usable_indices.txt", usable_indices)

    n_points = x_mso.size

    if usable_indices[0].size == 0:
        if print_feedback:
            print("No points within the magnetopause! Setting result to NaN...")

        return nan_return
    elif len([usable_indices[0]]) < n_points:
        # restrict to points within magnetopause
        x_msm = x_msm[usable_indices]
        y_msm = y_msm[usable_indices]
        z_msm = z_msm[usable_indices]

        control_params["kappa"] = control_params["kappa"][usable_indices]
        control_params["kappa3"] = control_params["kappa3"][usable_indices]
        di = di[usable_indices]

        if aberration.ndim > 0:
            # testing whether it is a full on numpy array or just an array with dimension 0 from the asarray operation earlier
            aberration = aberration[usable_indices]

        points_outside = n_points - usable_indices[0].size

        if points_outside > 0:
            if print_feedback:
                print(
                    "Warning: "
                    + str(n_points - usable_indices[0].size)
                    + " point(s) are located outside the magnetopause."
                )

    x_msm = x_msm.flatten()
    y_msm = y_msm.flatten()
    z_msm = z_msm.flatten()

    ##############################################################
    # Calculation of the model field
    #############################################################
    result = model_field_v10(
        x_msm,
        y_msm,
        z_msm,
        di,
        aberration,
        dipole,
        neutralsheet,
        ringcurrent,
        internal,
        external,
        control_params,
        R_SS,
    )

    # set all elements to nan which are located outside the magnetosphere or inside the planet
    result_with_nan = np.empty((3, input_length)) * np.nan

    result_with_nan[0, usable_indices] = result[0]
    result_with_nan[1, usable_indices] = result[1]
    result_with_nan[2, usable_indices] = result[2]

    result_with_nan[0, indices_inside_planet] = (
        np.empty(indices_inside_planet[0].size) * np.nan
    )
    result_with_nan[1, indices_inside_planet] = (
        np.empty(indices_inside_planet[0].size) * np.nan
    )
    result_with_nan[2, indices_inside_planet] = (
        np.empty(indices_inside_planet[0].size) * np.nan
    )

    return result_with_nan


def load_usable_indices() -> npt.NDArray:
    """Loads the usable indices as calculated by the kth_model_for_mercury_v10 function.

    Args:
        None

    Returns:
        numpy array of usable_indices

    """
    return np.loadtxt(temp_dir / "usable_indices.txt")
