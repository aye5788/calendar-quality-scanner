import numpy as np

def breakevens(prices, long_vals, short_vals):
    # total PL = long - short - debit
    pl = long_vals - short_vals

    # sign changes = breakevens
    signs = np.sign(pl)
    idx = np.where(np.diff(signs))[0]

    if len(idx) < 2:
        return None, None

    return prices[idx[0]], prices[idx[1]]
