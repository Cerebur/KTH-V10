# -*- coding: utf-8 -*-
"""created on 17 nov 2026 15:52.

@author: kristin
modified: jonathan
"""

import numpy as np


def fx(xj):
    """Calculates the"""
    # xj is considered in units of planetary radii
    # result is in nA/m^2
    # new in Version 10: NSC is stretched out on the night side, parameters: TBD in more detail,
    # problem: no messenger Data to compare with

    plateau_length = 3  # R_M
    max_curr_loc = -2.59
    plateau_hight = (
        100 * (max_curr_loc + 1.0) ** 2 * np.exp(-0.39 * (max_curr_loc + 1.0) ** 2)
    )

    x_1D = np.reshape(xj, xj.size)

    result_1D = 100 * (x_1D + 1.0) ** 2 * np.exp(-0.39 * (x_1D + 1.0) ** 2)
    result_1D[np.where(x_1D > -1)] = 0  # no current in planet or on dayside

    plateau_end = -2.59 - plateau_length

    indices_plateau = np.where((x_1D < max_curr_loc) & (x_1D > plateau_end))
    indices_far_tail = np.where(x_1D < plateau_end)

    result_1D[indices_plateau] = plateau_hight
    result_1D[indices_far_tail] = (
        100
        * (x_1D[indices_far_tail] + plateau_length + 1.0) ** 2
        * np.exp(-0.39 * (x_1D[indices_far_tail] + plateau_length + 1.0) ** 2)
    )

    return np.reshape(result_1D, xj.shape)
