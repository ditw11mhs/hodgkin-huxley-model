import numpy as np
from tqdm import tqdm


def custom_runge_kutta(
    ode_f,
    ode_f_vars: dict,
    y_runge: float,
    dt: float,
):
    """
    There is no t used on the ode function,
    so to reduce memory usage, no x is used

    ode_f (function): ode function
    ode_f_vars (dict): dictionary of unique constant for ode function
    y_runge (float): unique input for ode
    dt (float): time step
    """
    k1 = dt * ode_f(ode_f_vars, y_runge)
    k2 = dt * ode_f(ode_f_vars, y_runge + 0.5 * k1)
    k3 = dt * ode_f(ode_f_vars, y_runge + 0.5 * k2)
    k4 = dt * ode_f(ode_f_vars, y_runge + k3)
    out = y_runge + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return out


def base_ode_function(y, a, b):
    out = -y * (a + b) + a
    return out


def ode_function_n(vars: dict, n: float):
    out = base_ode_function(n, vars["an"], vars["bn"])
    return out


def ode_function_h(vars: dict, h: float):
    out = base_ode_function(h, vars["ah"], vars["bh"])
    return out


def ode_function_m(vars: dict, m: float):
    out = base_ode_function(m, vars["am"], vars["bm"])
    return out


def ode_function_v(vars: dict, *args, **kwargs):
    out = (vars["jin"] - vars["jna"] - vars["jk"] - vars["jl"]) / vars["cm"]
    return out


def ap_generator(j_in, v_gen, n_gen, m_gen, h_gen, c_var):
    vars = {
        "an": 0.01 * (v_gen + 10) / (np.exp(0.1 * v_gen + 1) - 1),
        "bn": 0.125 * np.exp(v_gen / 80),
        "am": 0.1 * (v_gen + 25) / (np.exp(0.1 * v_gen + 2.5) - 1),
        "bm": 4 * np.exp(v_gen / 18),
        "ah": 0.07 * np.exp(v_gen / 20),
        "bh": 1 / (np.exp(0.1 * v_gen + 3) + 1),
        "jin": j_in,
        "cm": c_var["cm"],
    }
    n_out = custom_runge_kutta(ode_function_n, vars, n_gen, c_var["dt"])
    m_out = custom_runge_kutta(ode_function_m, vars, m_gen, c_var["dt"])
    h_out = custom_runge_kutta(ode_function_h, vars, h_gen, c_var["dt"])

    vars["jk"] = c_var["gko"] * np.power(n_out, 4) * (v_gen - c_var["vk"])
    vars["jna"] = c_var["gnao"] * \
        np.power(m_out, 3) * h_out * (v_gen - c_var["vna"])
    vars["jl"] = c_var["glo"] * (v_gen - c_var["vl"])

    v_out = custom_runge_kutta(ode_function_v, vars, v_gen, c_var["dt"])

    return v_out, n_out, m_out, h_out


def ap_propagation(vars, v_p)


def main(t_max, dt):
    t_array = np.arange(0, t_max, dt)
    constant_vars = {
        "dt": dt,
        "cm": 1,
        "gko": 36,
        "gnao": 120,
        "glo": 0.3,
        "vk": 12,
        "vna": -115,
        "vl": -10.613,
        "gm": 3.45e-06,
        "ro": 1.5e06,
        "ri": 1.5e06,
    }
    v_gen = np.zeros(len(t_array))
    n_gen = 0
    h_gen = 0
    m_gen = 0
    for index, t in tqdm(enumerate(t_array)):
        if t >= 0 and t < 25:
            j_in = -5
        elif t >= 25 and t < 38:
            j_in = -10
        elif t >= 38 and t < 50:
            j_in = -20
        else:
            j_in = -60
        if index + 1 == len(t_array):
            break

        # AP Generation
        v_gen[index + 1], n_gen, m_gen, h_gen = ap_generator(
            j_in, v_gen[index], n_gen, m_gen, h_gen, constant_vars
        )

        # AP Propagation

    return v_gen, t_array
