import numpy as np


def mp_normal_v10(x_msm, y_msm, z_msm, RMP, alpha):
    """This function calculates the normal vector at the magnetopause.

    x,y,z : coordinates - arbitrary units in MSM coordinate system
    RMP : subsolar standoff distance - arbitrary units --> result will have the same units
    alpha : mp flaring parameter

    return : x, y, z components of normal vector (on MP)
    """
    r = np.sqrt(x_msm**2 + y_msm**2 + z_msm**2)
    phi = np.arctan2(y_msm, x_msm)
    theta = np.arccos(z_msm / r)

    r_mp = shue_mp_calc_r_mp(x_msm, y_msm, z_msm, RMP, alpha)
    mp_loc_x = r_mp * np.sin(theta) * np.cos(phi)
    mp_loc_y = r_mp * np.sin(theta) * np.sin(phi)
    mp_loc_z = r_mp * np.cos(theta)

    # first tangential vector: along rotation of gamma (rotation axis: x-axis)
    gamma = np.arctan2(mp_loc_y, mp_loc_z)
    e_gamma_x = 0.0
    e_gamma_y = np.cos(gamma)
    e_gamma_z = -np.sin(gamma)

    # second tangential vector: along the change of epsilon. This does NOT change gamma
    epsilon = np.cos(2.0 * ((r_mp / RMP) ** (-1.0 / alpha)) - 1.0)
    d_epsilon = 1e-3
    new_epsilon = epsilon + d_epsilon

    # with the new epsilon angle, calculate the corresponding magnetpause position
    new_r_mp = RMP * (2.0 / (1.0 + np.cos(new_epsilon))) ** alpha
    new_mp_loc_x = new_r_mp * np.cos(new_epsilon)
    new_mp_loc_y = new_r_mp * np.sin(new_epsilon) * np.sin(gamma)
    new_mp_loc_z = new_r_mp * np.sin(new_epsilon) * np.cos(gamma)

    # the difference vector is the connection vector
    connect_x = new_mp_loc_x - mp_loc_x
    connect_y = new_mp_loc_y - mp_loc_y
    connect_z = new_mp_loc_z - mp_loc_z

    # normalize and take the opposite direction
    magnitude = np.sqrt(connect_x**2 + connect_y**2 + connect_z**2)

    connect_x = -connect_x / magnitude
    connect_y = -connect_y / magnitude
    connect_z = -connect_z / magnitude

    # get normal direction by cross-product of tangentials
    # since both vectors are nomalized to 1 the cross-product has also the length 1
    mp_normal_x = connect_y * e_gamma_z - e_gamma_y * connect_z
    mp_normal_y = connect_z * e_gamma_x - e_gamma_z * connect_x
    mp_normal_z = connect_x * e_gamma_y - e_gamma_x * connect_y

    return mp_normal_x, mp_normal_y, mp_normal_z
