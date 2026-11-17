import numpy as np


def calc_L_shell_v10(
    x_start,
    y_start,
    z_start,
    r_hel,
    di,
    aberration,
    control_param_path,
    fit_param_path,
    imf_bx=0,
    imf_by=0,
    imf_bz=0,
):
    """This function calculates the L shell parameter.

    Args:
        x_start : position where the l shell parameter should be calculated. x in MSO in km
        y_start : position where the l shell parameter should be calculated. y in MSO in km
        z_start : position where the l shell parameter should be calculated. z in MSO in km

        r_hel : heliocentric distance in AU (between 0.3 and 0.47)
        di : Disturbance Index (further information see Anderson et al. 2013) (value between 0 and 100, default = 50 )
        aberration : aberration in degrees (average = 8° )

        control_param_path : like in KTH, do not change
        fit_param_path : like in KTH, do not change
        bx_imf : Bx component of IMF, optional
             The default is 0.
        by_imf : By component of IMF, optional
             The default is 0.
        bz_imf : Bz component of IMF, optional
            The default is 0.

    Returns:
        L Shell parameter: Here, the L Shell value/ L Shell parameter is defined
        as the radius (in Mercury Radii) of the location where the magnetic fieldline crosses the
        magnetic equatior. See: https://en.wikipedia.org/wiki/L-shell
    """
    # print('start calculating L-Shell')

    l_shell_parameter = np.nan

    if z_start > 479:
        delta_t = -0.9
        starting_hemisphere = 1  # northern hemisphere, z_msm positive
    if z_start < 479:
        delta_t = 0.9
        starting_hemisphere = -1  # southern hemisphere, z_msm negative

    # for opposite direction choose delta_t = -0.3 (or smaller/higher)

    if (x_start.size * y_start.size * z_start.size) != 1:
        print("Please enter only one starting position for the fieldline")
        return np.nan

    R_M = 2440  # Radius of Mercury in km
    r = np.sqrt((x_start / R_M) ** 2 + (y_start / R_M) ** 2 + (z_start / R_M) ** 2)
    if delta_t > 0:
        sign = 1
    if delta_t < 0:
        sign = -1

    if r < 1:
        print(
            "Radius of start point is smaller than 1. You start inside the planet! Radius = ",
            r,
        )
        return np.nan
    if x_start > 1.8 * R_M:
        print(
            "Your start point is outside the magnetosphere. Please choose a startposition closer to the planet. "
        )
        return np.nan

    def f(x, y, z):
        return kth_model_for_mercury_v10(
            x,
            y,
            z,
            r_hel,
            di,
            aberration,
            control_param_path,
            fit_param_path,
            neutralsheet=True,
            dipole=True,
            ringcurrent=True,
            internal=True,
            external=True,
        )

    x_trace = [x_start]
    y_trace = [y_start]
    z_trace = [z_start]
    mag_B_start = f(x_start, y_start, z_start)
    mag_B_value = np.sqrt(
        mag_B_start[0] ** 2 + mag_B_start[1] ** 2 + mag_B_start[2] ** 2
    )
    mag_B_trace = [mag_B_value]

    i = 0

    while r > 1 and i < 1000:
        # if i%10 == 0:
        # print('i = ', i)

        r = np.sqrt(
            (x_trace[i] / R_M) ** 2 + (y_trace[i] / R_M) ** 2 + (z_trace[i] / R_M) ** 2
        )
        if r < 1.00:
            # print('r < 1RM')
            break

        B = f(x_trace[i], y_trace[i], z_trace[i])
        # Numpy does not directly convert an array to scalar, so picking the first array value below
        # gives the scalar as a result
        # mag_B = float(np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)) added by MB on 15.4.2026
        mag_B = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)[0]

        if r > 1.5:
            delta_t = 150 / mag_B
            if delta_t < 0.3:
                delta_t = 0.3
            elif delta_t > 8:
                delta_t = 8

            delta_t = sign * delta_t
        elif r < 1.5:
            delta_t = sign * 0.9

        if np.isnan(B[0]) == True:
            break

        if np.isnan(x_trace[i]) == True:
            break

        k1 = delta_t * B

        if np.isnan(k1[0]) == True:
            # print('k1 is nan')
            break

        try:
            k2 = delta_t * f(
                x_trace[i] + 0.5 * k1[0],
                y_trace[i] + 0.5 * k1[1],
                z_trace[i] + 0.5 * k1[2],
            )
        except:
            # print('k2 is nan')
            break

        if np.isnan(k2[0]) == True:
            break

        try:
            k3 = delta_t * f(
                x_trace[i] + 0.5 * k2[0],
                y_trace[i] + 0.5 * k2[1],
                z_trace[i] + 0.5 * k2[2],
            )
        except:
            # print('k3 is nan')
            break
        if np.isnan(k3[0]) == True:
            # print('k3 is nan')
            break

        try:
            k4 = delta_t * f(x_trace[i] + k3[0], y_trace[i] + k3[1], z_trace[i] + k3[2])
        except:
            # print('k4 is nan')
            break

        if np.isnan(k4[0]) == True:
            # print('k4 is nan')
            break

        # print('k1: ', k1)
        # print('k2: ', k2)
        # print('k3: ', k3)
        # print('k4: ', k4)

        x_trace.append(x_trace[i] + (1 / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
        y_trace.append(y_trace[i] + (1 / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        z_trace.append(z_trace[i] + (1 / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))
        mag_B_trace.append(mag_B)

        if np.isnan(x_trace[-1]) == True:
            # print('last element is nan')
            return np.array([np.nan])
            break

        rho = np.sqrt(
            np.asarray(x_trace, dtype=float) ** 2
            + np.asarray(y_trace, dtype=float) ** 2
        )

        if starting_hemisphere > 0 and z_trace[-1] < 479:
            return np.max(rho) / R_M

        if starting_hemisphere < 0 and z_trace[-1] > 479:
            return np.max(rho) / R_M

        i = i + 1

        x_array = np.asarray(x_trace, dtype=object)
        y_array = np.asarray(y_trace, dtype=object)
        z_array = np.asarray(z_trace, dtype=object)
        mag_B_list = np.asarray(mag_B_trace, dtype=object)

        r = np.sqrt(
            x_array[-1].astype(float) ** 2
            + y_array[-1].astype(float) ** 2
            + z_array[-1].astype(float) ** 2
        )
        # print(r)

        if r < 1.00 * R_M:
            break

        if x_array[-1] <= -8 * R_M:
            break

    return np.round(l_shell_parameter, 4)
