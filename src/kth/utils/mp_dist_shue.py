import numpy as np


def shue_mp_calc_r_mp(x, y, z, RMP, alpha):
    """Calculated magnetopause distance.

    This calculates the magnetopause distance after the Shue et al. magnetopause model
    for the radial extension of an arbitrary point.

    Args:
        x,y,z : coordinates - arbitrary units in MSM coordinate system
        RMP : subsolar standoff distance - arbitrary units --> result will have the same units
        alpha : mp flaring parameter

    Returns:
        magnetopause distance w.r.t. planetary center
    """
    # distance to x-axis
    rho_x = np.sqrt(y**2 + z**2)
    # angle with x-axis
    epsilon = np.arctan2(rho_x, x)

    # Shue's formula
    mp_distance = RMP * np.power((2.0 / (1.0 + np.cos(epsilon))), alpha)

    return mp_distance
