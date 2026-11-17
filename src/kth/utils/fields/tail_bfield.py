import numpy as np
from numba import njit, prange

from ..currents.nsc_density import fx


def tail_field_ns_bs_v10(x_target, y_target, z_target, z_offset):
    mu = 4e-7 * np.pi
    RPL = 2440e3

    # bounds for the integration over j_y
    x_bounds = [-5.0 * RPL, -1.0 * RPL]
    y_bounds = [-1.0 * RPL, 1.0 * RPL]
    z_bounds = [-1.0 * RPL + z_offset * RPL, 1.0 * RPL + z_offset * RPL]

    # number of grid points in each dimension
    x_steps = 80
    y_steps = 40
    z_steps = 40

    # differential volumen
    dV = (
        (x_bounds[1] - x_bounds[0])
        / float(x_steps - 1.0)
        * (y_bounds[1] - y_bounds[0])
        / float(y_steps - 1.0)
        * (z_bounds[1] - z_bounds[0])
        / float(z_steps - 1.0)
    )

    # create integration mesh with meshgrid
    x_coords_1d = (
        np.arange(x_steps) / float(x_steps - 1) * (x_bounds[1] - x_bounds[0])
        + x_bounds[0]
    )
    y_coords_1d = (
        np.arange(y_steps) / float(y_steps - 1) * (y_bounds[1] - y_bounds[0])
        + y_bounds[0]
    )
    z_coords_1d = (
        np.arange(z_steps) / float(z_steps - 1) * (z_bounds[1] - z_bounds[0])
        + z_bounds[0]
    )

    # prepare coordinates in multidimensional arrays - this information goes into the f_xyz functions that define the current density
    x_coords, y_coords, z_coords = np.meshgrid(x_coords_1d, y_coords_1d, z_coords_1d)

    x_coords = np.reshape(x_coords, x_coords.size)
    y_coords = np.reshape(y_coords, y_coords.size)
    z_coords = np.reshape(z_coords, z_coords.size)

    n_vec = len(x_target)

    fx_result = fx(x_coords / RPL)

    # @njit(fastmath=True, parallel=True)
    # def tail_field_fast_sum_multi():
    #     Bx_arr = np.zeros(n_vec)
    #     By_arr = np.zeros(n_vec)
    #     Bz_arr = np.zeros(n_vec)

    #     #differential volume

    #     for i_vec in prange(n_vec):

    #         x_rel = x_target[i_vec]*RPL - x_coords
    #         y_rel = y_target[i_vec]*RPL - y_coords
    #         z_rel = z_target[i_vec]*RPL - z_coords

    #         r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

    #         #check for singularity
    #         for i in range(len(r)) :
    #             if r[i] < (0.1*RPL):
    #                 r[i] = 0.1*RPL
    #         r_inv = r**(-3.)

    #         Bx = mu / (4. * np.pi) * np.sum( fx_result * np.exp(-0.5*(y_coords/RPL)**2) * np.exp(-0.1 * ((z_coords/RPL) / 0.12)**2) * z_rel * r_inv) * dV
    #         Bz = mu / (4. * np.pi) * np.sum(-fx_result * np.exp(-0.5*(y_coords/RPL)**2) * np.exp(-0.1 * ((z_coords/RPL) / 0.12)**2) * x_rel * r_inv) * dV
    #         Bx_arr[i_vec] = Bx
    #         Bz_arr[i_vec] = Bz

    #     return Bx_arr, By_arr, Bz_arr

    @njit(fastmath=True, parallel=True)
    def tail_field_fast_sum_multi():
        # Pre-allocate arrays
        B_arr = np.zeros((3, n_vec))

        # differential volume

        for i_vec in prange(n_vec):
            x_rel = x_target[i_vec] * RPL - x_coords
            y_rel = y_target[i_vec] * RPL - y_coords
            z_rel = z_target[i_vec] * RPL - z_coords

            r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

            # Check for singularity
            for i in range(len(r)):
                if r[i] < (0.1 * RPL):
                    r[i] = 0.1 * RPL

            r_inv = r ** (-3.0)

            # Calculate Bx and Bz (By remains zero)
            common_factor = np.exp(-0.5 * (y_coords / RPL) ** 2) * np.exp(
                -0.1 * ((z_coords / RPL) / 0.12) ** 2
            )
            Bx = (
                mu
                / (4.0 * np.pi)
                * np.sum(fx_result * common_factor * z_rel * r_inv)
                * dV
            )
            Bz = (
                mu
                / (4.0 * np.pi)
                * np.sum(-fx_result * common_factor * x_rel * r_inv)
                * dV
            )

            B_arr[0, i_vec] = Bx  # Bx
            B_arr[2, i_vec] = Bz  # Bz

        return B_arr

    def tail_field_fast_sum_single():
        Bx_arr = np.zeros(n_vec)
        By_arr = np.zeros(n_vec)
        Bz_arr = np.zeros(n_vec)

        # differential volume

        for i_vec in range(n_vec):
            x_rel = x_target[i_vec] * RPL - x_coords
            y_rel = y_target[i_vec] * RPL - y_coords
            z_rel = z_target[i_vec] * RPL - z_coords

            r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

            # check for singularity
            for i in range(len(r)):
                if r[i] < (0.1 * RPL):
                    r[i] = 0.1 * RPL

            r_inv = r ** (-3.0)

            Bx = (
                mu
                / (4.0 * np.pi)
                * np.sum(
                    fx_result
                    * np.exp(-0.5 * (y_coords / RPL) ** 2)
                    * np.exp(-0.1 * ((z_coords / RPL) / 0.12) ** 2)
                    * z_rel
                    * r_inv
                )
                * dV
            )
            Bz = (
                mu
                / (4.0 * np.pi)
                * np.sum(
                    -fx_result
                    * np.exp(-0.5 * (y_coords / RPL) ** 2)
                    * np.exp(-0.1 * ((z_coords / RPL) / 0.12) ** 2)
                    * x_rel
                    * r_inv
                )
                * dV
            )
            Bx_arr[i_vec] = Bx
            Bz_arr[i_vec] = Bz

        return Bx_arr, By_arr, Bz_arr

    # decision wether single or multi processing is used to make the model faster.
    # On test computer ( Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz ), multi processing is faster
    # when more than 70 data points are calculated. This value might differ on other
    # computers.

    single_multi_limit = 70

    if x_target.size > single_multi_limit:
        # print('multi')
        output_tail_field = tail_field_fast_sum_multi()

    else:
        # print('single')
        output_tail_field = tail_field_fast_sum_single()

    # Convert B_nsc to a numpy array if it's a list or tuple
    if isinstance(output_tail_field, (list, tuple)):
        output_tail_field = np.array(output_tail_field)
    return output_tail_field
