import numpy as np


def calc_M_shell_v10(
    x_start,
    y_start,
    z_start,
    r_hel,
    di,
    aberration,
    control_param_path,
    fit_param_path,
    delta_t=0.3,
    imf_bx=0,
    imf_by=0,
    imf_bz=0,
):
    """This function calculates the M shell parameter.
    Here, the M Shell value/ M Shell parameter is defined
    as the location of the minimum magnetic
    field value along the fieldline. Multiple positions are possible.
    M Shell (radius of the return of this function) and L Shell should be similar on nightside and different on dayside.


    Parameters
    ----------
    x_start : position where the m shell parameter should be calculated. x in MSO in km
    y_start : position where the m shell parameter should be calculated. y in MSO in km
    z_start : position where the m shell parameter should be calculated. z in MSO in km

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
    -------
    M Shell (position of magnetic field minimum value along fieldline in x_mso, y_mso, z_mso)


    """
    delta_t_in = np.abs(delta_t)
    delta_t_sign_change = -delta_t_in

    x_at_min_B = np.nan
    y_at_min_B = np.nan
    z_at_min_B = np.nan

    fieldline1 = trace_fieldline_v10(
        x_start,
        y_start,
        z_start,
        r_hel,
        di,
        aberration,
        control_param_path,
        fit_param_path,
        delta_t_in,
        imf_bx,
        imf_by,
        imf_bz,
    )
    fieldline2 = trace_fieldline_v10(
        x_start,
        y_start,
        z_start,
        r_hel,
        di,
        aberration,
        control_param_path,
        fit_param_path,
        delta_t_sign_change,
        imf_bx,
        imf_by,
        imf_bz,
    )

    # print('fieldline1', fieldline1)
    # print('fieldline2', fieldline2)

    if fieldline1[0].size > 1:
        x_mso_fieldline = np.concatenate(
            (np.flip(fieldline1[0].astype(float)), fieldline2[0].astype(float))
        )
        y_mso_fieldline = np.concatenate(
            (np.flip(fieldline1[1].astype(float)), fieldline2[1].astype(float))
        )
        z_mso_fieldline = np.concatenate(
            (np.flip(fieldline1[2].astype(float)), fieldline2[2].astype(float))
        )
        B_total_nT_fieldline = np.round(
            np.concatenate(
                (np.flip(fieldline1[3].astype(float)), fieldline2[3].astype(float))
            ),
            5,
        )

        # minimum_B = np.nanmin(B_total_nT_fieldline)
        # print('minimum B: ', minimum_B)
        index_min_B = np.where(B_total_nT_fieldline == np.nanmin(B_total_nT_fieldline))
        # print('Indices at minimum: ', index_min_B)
        x_at_min_B = x_mso_fieldline[index_min_B]
        y_at_min_B = y_mso_fieldline[index_min_B]
        z_at_min_B = z_mso_fieldline[index_min_B]

    return np.array([x_at_min_B, y_at_min_B, z_at_min_B])
