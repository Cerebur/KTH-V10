def calc_R_SS_km(r_hel, di, control_param_path):
    """Parameters
    ----------
    r_hel : helicocentric distance in AU
    di : disturbance index. Value between 0 and 100
    control_param_path: see KTH Model


    Returns:
    -------
    R_SS : subsolar standoff distance in km
    """
    with open(control_param_path, "r") as file:
        control_params = json.load(file)

    # f is a factor for the scaling for R_SS (subsolar standoff distance)
    # f_a and f_b are determined by fitting agains MESSENGER data

    f = control_params["f_a"] + (control_params["f_b"] * di)

    R_SS = f * (r_hel ** (1 / 3)) * control_params["RPL"]
    return R_SS
