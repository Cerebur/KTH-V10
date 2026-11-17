import numpy as np


def cf_field_v10(
    x_msm: np.ndarray,
    y_msm: np.ndarray,
    z_msm: np.ndarray,
    lin_coeff: list,
    p_i,
    q_i,
    x_shift,
):
    # this function calculates the chapman-ferraro-field (schielding field) for the KTH Model

    n_vec = x_msm.size

    N = len(p_i)

    b_x_cf = np.zeros(n_vec)
    b_y_cf = np.zeros(n_vec)
    b_z_cf = np.zeros(n_vec)

    for i_vec in range(n_vec):
        for i in range(N):
            for k in range(N):
                pq = np.sqrt(p_i[i] * p_i[i] + q_i[k] * q_i[k])

                lin_index = i * N + k

                b_x_cf[i_vec] = b_x_cf[i_vec] - pq * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])
                ) * np.cos(p_i[i] * y_msm[i_vec]) * np.sin(q_i[k] * z_msm[i_vec])

                b_y_cf[i_vec] = b_y_cf[i_vec] + p_i[i] * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])
                ) * np.sin(p_i[i] * y_msm[i_vec]) * np.sin(q_i[k] * z_msm[i_vec])

                b_z_cf[i_vec] = b_z_cf[i_vec] - q_i[k] * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])
                ) * np.cos(p_i[i] * y_msm[i_vec]) * np.cos(q_i[k] * z_msm[i_vec])

    return np.array([b_x_cf, b_y_cf, b_z_cf])
