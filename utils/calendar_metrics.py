import numpy as np

def iv_slope(front_iv, back_iv):
    return front_iv - back_iv

def vega_theta_ratio(vf, vb, tf, tb):
    net_vega = vb - vf
    net_theta = abs(tf) - abs(tb)
    if net_theta <= 0:
        return np.nan
    return net_vega / net_theta

def theta_advantage(tf, tb):
    return abs(tf) - abs(tb)

def iv_ratio(front_iv, back_iv):
    return front_iv / back_iv if back_iv != 0 else np.nan

def hover_metric(iv20, hv20):
    return iv20 - hv20

def payoff_ratio(bwidth, debit):
    if debit <= 0:
        return np.nan
    return bwidth / debit
