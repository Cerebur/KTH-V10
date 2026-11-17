# -*- coding: utf-8 -*-
"""Created on 17 Nov 2026 15:52.

@author: kristin
Modified: Jonathan
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from kth.fieldline_tracers.multiple_field_lines import trace_fieldline_v10


def trace_field_line_single_v10():
    """Example function of how to plot field lines.

    Creates a plot with a fieldline trace.

    Args:
        None

    Returns: None
    """
    mercury1 = patches.Circle((0, (479 / 2440)), 1, color="0.75")

    R_M = 2440.0
    x_start = -1.2 * R_M
    y_start = 0
    z_start = 479
    r_hel = 0.37
    di = 50

    fieldlinetrace = trace_fieldline_v10(
        x_start, y_start, z_start, r_hel, di, aberration=0, delta_t_in=0.3
    )
    fieldlinetrace = trace_fieldline_v10(
        x_start, y_start, z_start, r_hel, di, aberration=0, delta_t_in=-0.3
    )
    # print(fieldlinetrace)
    x = fieldlinetrace[0]  # x
    # y = fieldlinetrace[1]  # y
    z = fieldlinetrace[2]  # z

    fig, ax1 = plt.subplots()
    ax1.add_artist(mercury1)
    plt.plot(x / R_M, z / R_M)
    ax1.axis("square")
    plt.xlim((-5, 3))
    plt.ylim((-3, 3))
    ax1.grid()
    ax1.invert_xaxis()
    plt.show()
