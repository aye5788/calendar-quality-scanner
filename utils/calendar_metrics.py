import numpy as np

# ------------------------------
# BASIC METRICS
# ------------------------------

def iv_slope(front_iv, back_iv):
    return (back_iv - front_iv) / max(front_iv, 1e-6)

def iv_ratio(front_iv, back_iv):
    return back_iv / max(front_iv, 1e-6)

def vega_theta_ratio(f_vega, b_vega, f_theta, b_theta):
    try:
        vt = (b_vega - f_vega) / abs(f_theta - b_theta + 1e-9)
        return vt if vt != np.inf else np.nan
    except:
        return np.nan

def theta_advantage(f_theta, b_theta):
    return -f_theta - (-b_theta)

def hover_metric(iv20, hv20):
    return iv20 - hv20

def payoff_ratio(bwidth, debit):
    if debit <= 0:
        return np.nan
    return bwidth / debit


# ==========================================================
#  🔥 UNIVERSAL, RESEARCH-BASED CALENDAR SCORING MODEL
# ==========================================================

def calendar_quality_score(row, all_debits, hover):
    score = 0

    # ------------------------------
    # 1. IV Slope
    # ------------------------------
    slope = row["IV Slope"]
    if slope > 0:
        score += 20
    elif slope > -0.03:
        score += 12
    else:
        score += 0

    # ------------------------------
    # 2. IV Ratio
    # ------------------------------
    ivr = row["IV Ratio"]
    if ivr >= 1.02:
        score += 20
    elif ivr >= 0.97:
        score += 12
    else:
        score += 0

    # ------------------------------
    # 3. Debit (normalized)
    # ------------------------------
    debit = row["Debit"]
    if debit == min(all_debits):
        score += 20
    elif debit <= np.median(all_debits):
        score += 12
    else:
        score += 6

    # ------------------------------
    # 4. Theta Advantage
    # ------------------------------
    tadv = row["Theta Adv"]
    if tadv > 0:
        score += 15
    elif tadv > -0.02:
        score += 8
    else:
        score += 0

    # ------------------------------
    # 5. Vega/Theta Ratio
    # ------------------------------
    vtr = row["Vega/Theta"]
    if np.isnan(vtr):
        score += 4
    elif vtr >= 1.3:
        score += 10
    elif vtr >= 0.7:
        score += 6
    else:
        score += 2

    # ------------------------------
    # 6. BE vs Move
    # ------------------------------
    be = row["BE/Move"]
    if be is None or np.isnan(be):
        score += 0
    elif be >= 0.9:
        score += 15
    elif be >= 0.5:
        score += 8
    else:
        score += 0

    # ------------------------------
    # 7. Hover (Ticker regime)
    # ------------------------------
    if hover > 0:
        score += 10

    return score
