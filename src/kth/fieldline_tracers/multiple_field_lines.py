# -*- coding: utf-8 -*-
"""Created on 17 Nov 2026 15:52.

@author: kristin
Modified: Jonathan
"""

# Proposed fix for trace_fieldline_v10:
# - Keep magnetic arrays empty before the while loop.
# - In each iteration, append B at the current point once, then compute the next coordinates.
# - After the loop, compare len(x_trace) with len(Bx) and append the final B value if the coordinate
#   arrays are longer.
# - This avoids the off-by-one mismatch where x_trace grows one step ahead of Bx.
#
# The mismatch occurs because x/y/z are appended after the RK step while B values are appended for
# the current point. If the loop exits right after appending the last coordinate, the B arrays can be
# one element shorter than the coordinate arrays.
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt

from kth.models.kth_model_v10 import (
    ScalarOrArray,
    kth_model_for_mercury_v10,
    load_usable_indices,
)
from kth.utils import clean_array

# Directory containing the currently executed Python file
script_dir = Path(__file__).resolve().parent

# Directory pointing to the default parameters
default_parameter_dir = script_dir.parent / "parameters"


def trace_fieldline_v10(
    x_start: ScalarOrArray,
    y_start: ScalarOrArray,
    z_start: ScalarOrArray,
    r_hel: ScalarOrArray,
    di: ScalarOrArray,
    aberration: ScalarOrArray,
    *,
    delta_t_in=0.9,
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
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Traces the magnetic field line from a given start location.

    Uses the kth model to calculate the magnetic field at a specific location and uses that to develop the field line according to 4th order Runge-Kutta.

    Types:
        ScalarOrArray: Either a single floating-point value or a NumPy array
                        of floating-point values. All floats are internally
                        converted to np.float64



    Args:
        x_start: x-location of the start point of the field line in km
        y_start: y-location of the start point of the field line in km
        z_start: z-location of the start point of the field line in km
        r_hel: heliocentric distance in AU
        di: Disturbance Index (0-100)
        aberration: Aberration angle in deg
        delta_t_in: delta_t_in controles the sign and the stepsize in the Runge Kutta Alogrithm for fieldlinetracing. Small delta_t_in (eg 0.1 or 0.3) lead to more precise results buthave longer computing times. Large delta_t_in (e.g. 3) lead to less precise results and less computing time. Good value to start with: 0.9
        control_param_path: gives the path to the control parameters needed by the kth model
        fit_param_path: gives the path to the fitting parameters needed by the kth model
        imf_bx: IMF Bx component. Not yet implemented.
        imf_by: IMF By component. Not yet implemented.
        imf_bz: IMF Bz component. Not yet implemented.
        dipole: If True, include the planetary dipole field.
        neutralsheet: If True, include the neutral sheet contribution.
        ringcurrent: If True, include the ring current contribution.
        internal: If True, include internal magnetospheric field contributions.
        external: If True, include external Chapman-Ferraro field contributions.

    Returns:
       Returns a tuple with the following elements:
       1. Array with x-locations of field line
       2. Array with y-locations of field line
       3. Array with z-locations of field line
       4. Array with magnitude of B-field along the line
       5. Array with Bx along the line
       6. Array with By along the line
       7. Array with Bz along the line
       If the returned arrays contain nans, the values could not be determined for these positions

    """
    # for opposite direction choose delta_t = -0.9 (or smaller/higher)

    # print('start calculating fieldline')

    nan_return = (
        np.array(np.nan),
        np.array(np.nan),
        np.array(np.nan),
        np.array(np.nan),
        np.array(np.nan),
        np.array(np.nan),
        np.array(np.nan),
    )

    R_M = 2440  # Radius of Mercury in km
    r = np.sqrt((x_start / R_M) ** 2 + (y_start / R_M) ** 2 + (z_start / R_M) ** 2)
    # print('r = ', r)

    if delta_t_in > 0:
        sign = 1
    elif delta_t_in < 0:
        sign = -1
    else:
        print("Delta_t stepsize must not be 0! Returning nan values")
        return nan_return

    if r < 1:
        print(
            "Radius of start point is smaller than 1. You start inside the planet! Radius = ",
            r,
        )
        # sys.exit()
        # Return an array of nans instead of exiting. This way it is easier to use a Cartesian grid of
        # #starting points for field line tracing, where some points are inside the planet. MB 29.4.2026
        return nan_return

    def f(x, y, z):
        return kth_model_for_mercury_v10(
            x,
            y,
            z,
            r_hel,
            di,
            aberration,
            control_param_path=control_param_path,
            fit_param_path=fit_param_path,
            dipole=dipole,
            neutralsheet=neutralsheet,
            ringcurrent=ringcurrent,
            internal=internal,
            external=external,
            imf_bx=imf_bx,
            imf_by=imf_by,
            imf_bz=imf_bz,
        )

    x_trace: list[npt.NDArray[np.float64]] = [
        np.asarray(x_start, dtype=np.float64).reshape(())
    ]
    y_trace: list[npt.NDArray[np.float64]] = [
        np.asarray(y_start, dtype=np.float64).reshape(())
    ]
    z_trace: list[npt.NDArray[np.float64]] = [
        np.asarray(z_start, dtype=np.float64).reshape(())
    ]

    mag_B_trace = []

    # Save values of the magnetic field along the field line trace for later use. This is important for example for the calculation of the plasma transport along the field lines. MB 15.4.2026
    Bx = []
    By = []
    Bz = []

    i = 0

    while r > 1 and i < 1000:
        if i % 10 == 0:
            print("i = ", i)

        r = np.sqrt(
            (x_trace[i] / R_M) ** 2 + (y_trace[i] / R_M) ** 2 + (z_trace[i] / R_M) ** 2
        )
        if r < 1.00:
            print("r < 1RM")
            break

        B = f(x_trace[i], y_trace[i], z_trace[i])

        # Numpy does not directly convert an array to scalar, so picking the first array value below
        # gives the scalar as a result
        # mag_B = float(np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)) added by MB on 15.4.2026
        mag_B = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)[0]

        if r > 1.5:  # far away from Mercury
            delta_t = 150 / mag_B
            if delta_t < 0.3:
                delta_t = 0.3
            elif delta_t > 8:
                delta_t = 8
            delta_t = sign * delta_t
        else:
            delta_t = delta_t_in

        if np.isnan(B[0]):
            break
        if np.isnan(x_trace[i]):
            break

        Bx.append(B[0])
        By.append(B[1])
        Bz.append(B[2])
        mag_B_trace.append(mag_B)

        k1 = delta_t * B
        if np.isnan(k1[0]):
            # print('k1 is nan')
            break
        try:
            k2 = delta_t * f(
                x_trace[i] + 0.5 * k1[0],
                y_trace[i] + 0.5 * k1[1],
                z_trace[i] + 0.5 * k1[2],
            )
        except Exception as e:
            print("Exception has been raised, breaking current loop:\n", e)
            break
        if np.isnan(k2[0]):
            break
        try:
            k3 = delta_t * f(
                x_trace[i] + 0.5 * k2[0],
                y_trace[i] + 0.5 * k2[1],
                z_trace[i] + 0.5 * k2[2],
            )
        except Exception as e:
            print("Exception has been raised, breaking current loop:\n", e)
            break
        if np.isnan(k3[0]):
            # print('k3 is nan')
            break
        try:
            k4 = delta_t * f(x_trace[i] + k3[0], y_trace[i] + k3[1], z_trace[i] + k3[2])
        except Exception as e:
            print("Exception has been raised, breaking current loop:\n", e)
            break

        if np.isnan(k4[0]):
            # print('k4 is nan')
            break

        # print('k1: ', k1)
        # print('k2: ', k2)
        # print('k3: ', k3)
        # print('k4: ', k4)

        x_new = np.asarray(x_trace[i]).reshape(()) + (1 / 6) * np.asarray(
            k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]
        ).reshape(())
        y_new = np.asarray(y_trace[i]).reshape(()) + (1 / 6) * np.asarray(
            k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]
        ).reshape(())
        z_new = np.asarray(z_trace[i]).reshape(()) + (1 / 6) * np.asarray(
            k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]
        ).reshape(())
        x_trace.append(np.asarray(x_new))
        y_trace.append(np.asarray(y_new))
        z_trace.append(np.asarray(z_new))

        if np.isnan(x_trace[-1]):
            # print('last element is nan')
            break

        i = i + 1

        usable_indices = load_usable_indices()
        if len(np.atleast_1d(usable_indices)) == 0:
            break

        r = np.sqrt(
            (x_trace[-1] / R_M) ** 2
            + (y_trace[-1] / R_M) ** 2
            + (z_trace[-1] / R_M) ** 2
        )

        if r < 1.00:
            break
        if (
            x_trace[-1] <= -6 * R_M
        ):  # abortion of fieldline calculation if position if further on nithside than -4 RM. Change this value if you want to go further to nightside.
            break
    # Ensure final magnetic field arrays match coordinate lengths.
    # If the loop exited before any magnetic field samples were stored, fill the remaining slots with NaNs.
    if len(x_trace) > 0:
        if len(Bx) < len(x_trace):
            Bx.extend([np.nan] * (len(x_trace) - len(Bx)))
            By.extend([np.nan] * (len(y_trace) - len(By)))
            Bz.extend([np.nan] * (len(z_trace) - len(Bz)))
        if len(mag_B_trace) < len(x_trace):
            mag_B_trace.extend([np.nan] * (len(x_trace) - len(mag_B_trace)))
    if len(Bx) > len(x_trace):
        Bx = Bx[: len(x_trace)]
        By = By[: len(y_trace)]
        Bz = Bz[: len(z_trace)]
    if len(mag_B_trace) > len(x_trace):
        mag_B_trace = mag_B_trace[: len(x_trace)]

    # Build final coordinate and magnetic field arrays after the loop so lengths are consistent.
    x_array = np.asarray(x_trace, dtype=np.float64)
    y_array = np.asarray(y_trace, dtype=np.float64)
    z_array = np.asarray(z_trace, dtype=np.float64)
    mag_B_list = np.asarray(mag_B_trace, dtype=np.float64)
    try:
        # Added by MB on 16.4.2026 so that the matplotlib plotting works for the field lines
        x_array = clean_array(x_array).astype(np.float64)
        y_array = clean_array(y_array).astype(np.float64)
        z_array = clean_array(z_array).astype(np.float64)
        mag_B_list = clean_array(mag_B_list).astype(np.float64)
        Bx = clean_array(Bx).astype(np.float64)
        By = clean_array(By).astype(np.float64)
        Bz = clean_array(Bz).astype(np.float64)

        # Check that arrays are of the same size
        if np.shape(x_array) != np.shape(Bx):
            print(
                "Error: x_array and Bx have different sizes. x_array size: ",
                np.shape(x_array),
                "Bx size: ",
                np.shape(Bx),
            )

        return x_array, y_array, z_array, mag_B_list, Bx, By, Bz
    except Exception as e:
        print("Exception raised. Returning nans:\n", e)
        return nan_return
