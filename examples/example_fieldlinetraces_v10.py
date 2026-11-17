# -*- coding: utf-8 -*-
"""Created on Mon Sep 18 22:55:24 2023.

@author: kristin
Modified: Jonathan
"""
# =============================================================================
# example for how to use the fieldline tracer in KTH V10b

# 1. example: single fieldline
# 2. example: tracing and plotting multiple fieldlines
# =============================================================================

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge

import kth

# ================================================================================
# Defining path variables
# ================================================================================

# System Path to the KTH22-model-main folder
your_System_Path = "/Users/jonathan/Documents/PhD/KTH/"  # adapt this local path

sys.path.append(your_System_Path)
control_param_path = os.path.join(your_System_Path, "control_params_v10.json")
fit_param_path = os.path.join(your_System_Path, "kth_own_cf_fit_parameters_v10.dat")

# ================================================================================
# Updating plotting parameters
# ================================================================================

plt.rcParams.update({"font.size": 16})

# ================================================================================
# Defining constants
# ================================================================================
R_M = 2440  # in km, Mercury Radius

r_hel = np.array([0.39])  # Subsolar standoff distance??
di = np.array([50])  # Disturbance Index
aberration = 0  # What is the aberration again??
imf_bx = imf_by = imf_bz = 0  # IMF
delta_t_in = 0.9  # delta_t_in controles the sign and the stepsize in the Runge Kutta
# Alogrithm for fieldlinetracing. Small delta_t_in (eg 0.1 or 0.3)
# lead to more precise results buthave longer computing times.
# Large delta_t_in (e.g. 3) lead to less precise results and
# less computing time.
# Good value to start with: 0.9


# choose example
example_for_single_fieldline = False
example_for_multiple_fieldlines = True
number_of_fieldlines = 5

# =============================================================================
# 1. example: define contants and input parameters for one fieldline
# =============================================================================

if example_for_single_fieldline:
    x_start = np.array([1.1]) * R_M
    y_start = np.array([0.0])
    z_start = np.array([-0.5]) * R_M

    # =============================================================================
    # calculate fieldline
    # =============================================================================

    fieldline = kth.trace_fieldline_v10(
        x_start,
        y_start,
        z_start,
        r_hel,
        di,
        aberration,
        delta_t_in=delta_t_in,
    )

    # =============================================================================
    # explenation of return:

    # fieldline[0] -> x in mso in km
    # fieldline[1] -> y in mso in km
    # fieldline[2] -> z in mso in km
    # fieldline[3] -> |B| in nT
    # =============================================================================

    print("footpoint /end of calculated fieldline: ")
    print("x_mso (km): ", (fieldline[0])[-1])
    print("y_mso (km): ", (fieldline[1])[-1])
    print("z_mso (km): ", (fieldline[2])[-1])

    # =============================================================================
    # plot single fieldline
    # =============================================================================
    theta1, theta2 = 90, 90 + 180
    radius = 1
    center = (0, 0)
    m_day1 = Wedge(center, radius, theta1, theta2, fc="0.4", edgecolor="0.4")
    m_night1 = Wedge(center, radius, theta2, theta1, fc="0.8", edgecolor="0.8")

    lim = 3
    title = "fieldlinetrace"

    fig = plt.figure(figsize=(7, 7))
    fig.suptitle(title, fontsize=16)

    # x-z
    ax = plt.subplot(1, 1, 1)
    ax.axis("square")  # creates square ax
    plt.xlim((lim, -lim))  # set axis limits
    plt.ylim((-lim, lim))
    plt.plot(fieldline[0] / R_M, fieldline[2] / R_M, color="0.1")
    ax.add_artist(m_day1)
    ax.add_artist(m_night1)  # creats grey circle (mercury)
    plt.xlabel(r"$X_{MSM}$" + " in " + "$R_M$")
    plt.ylabel(r"$Z_{MSM}$" + " in " + "$R_M$")
    plt.grid()


###############################################################################


# =============================================================================
# 2. example: define contants and input parameters for multiple fieldlines
# =============================================================================

if example_for_multiple_fieldlines:
    alpha = np.linspace(-np.pi + 0.1, np.pi - 0.1, number_of_fieldlines)
    x_start = np.sin(alpha) * (R_M + 100)
    y_start = np.zeros(len(alpha))
    z_start = np.cos(alpha) * (R_M + 100)

    # =============================================================================
    # plot multiple fieldlines
    # =============================================================================

    # (ideally you split this routine into one calculation routine and one plotting routine.
    # This routine is just to show how it works. )

    theta1, theta2 = 90, 90 + 180
    radius = 1
    center = (0, 0)
    m_day1 = Wedge(center, radius, theta1, theta2, fc="0.4", edgecolor="0.4")
    m_night1 = Wedge(center, radius, theta2, theta1, fc="0.8", edgecolor="0.8")

    lim = 3
    title = "fieldlinetraces"

    fig = plt.figure(figsize=(7, 7))
    fig.suptitle(title, fontsize=16)

    # x-z
    ax = plt.subplot(1, 1, 1)
    ax.axis("square")  # creates square ax
    plt.xlim((lim, -lim))  # set axis limits
    plt.ylim((-lim, lim))

    for i in range(len(alpha)):
        plt.plot(
            x_start[i] / R_M, z_start[i] / R_M, "o", color="orange", label="start point"
        )

        print("calculate fieldline ", i, "/", str(len(alpha) - 1))
        fieldline1 = kth.trace_fieldline_v10(
            x_start[i],
            y_start[i],
            z_start[i],
            r_hel,
            di,
            aberration,
            delta_t_in=delta_t_in,
        )
        fieldline2 = kth.trace_fieldline_v10(
            x_start[i],
            y_start[i],
            z_start[i],
            r_hel,
            di,
            aberration,
            delta_t_in=delta_t_in * -1,
        )

        plt.plot(fieldline1[0] / R_M, fieldline1[2] / R_M, color="blue")
        plt.plot(fieldline2[0] / R_M, fieldline2[2] / R_M, color="blue")

        # save fieldlines
        np.savetxt(
            "fieldline1_alpha_"
            + str(np.round(alpha[i] / 2 / np.pi * 360, 0).astype(int))
            + ".txt",
            fieldline1,
        )
        np.savetxt(
            "fieldline2_alpha"
            + str(np.round(alpha[i] / 2 / np.pi * 360, 0).astype(int))
            + ".txt",
            fieldline2,
        )

    ax.add_artist(m_day1)  # creats grey circle (mercury)
    ax.add_artist(m_night1)  # creats grey circle (mercury)
    plt.xlabel(r"$X_{MSM}$" + " in " + "$R_M$")
    plt.ylabel(r"$Z_{MSM}$" + " in " + "$R_M$")
    plt.grid()

plt.show()
