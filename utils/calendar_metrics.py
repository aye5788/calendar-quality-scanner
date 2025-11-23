import numpy as np

def iv_slope(front_iv, back_iv):
    return front_iv - back_iv

def vega_theta_ratio(vega_f, vega_b, theta_f, theta_b):
    net_vega = vega_b - vega_f
    net_theta = abs(theta_f) - abs(theta_b)
    if net_theta <= 0:
        return np.nan
    return net_vega / net_theta

def theta_advantage(theta_f, theta_b):
    return abs(theta_f) - abs(theta_b)

def iv_ratio(front_iv, back_iv):
    return front_iv / back_iv if back_iv != 0 else np.nan

def hover_metric(iv20, hv20):
    return iv20 - hv20

def payoff_ratio(bwidth, debit):
    if debit <= 0:
        return np.nan
    return bwidth / debit
