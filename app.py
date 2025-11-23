import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.orats_api import ORATS
from utils.calendar_metrics import (
    iv_slope,
    vega_theta_ratio,
    theta_advantage,
    iv_ratio,
    hover_metric,
    payoff_ratio
)
from utils.breakeven_solver import breakevens

# ------------------------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Calendar Quality Scanner",
    layout="wide",
)

st.title("📈 Calendar Spread Quality Scanner (ORATS)")

orats = ORATS()

# ------------------------------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------------------------------
ticker = st.text_input("Ticker", "SLV").upper()

if ticker:
    # pull expiration list
    chains = orats.get_chains(ticker)
    exps = chains["expirations"]

    front_exp = st.selectbox("Select Front Expiry", exps)
    run = st.button("SCAN CALENDARS")

    if run:
        st.subheader("Fetching ORATS Data…")

        cores = orats.get_cores(ticker).iloc[0]
        summaries = orats.get_summaries(ticker).iloc[0]

        stock = summaries["stockPrice"]
        imp_move_abs = stock * summaries["impliedMove"]

        # grab front-month strikes dataframe
        df_front = orats.get_strikes(ticker, front_exp)

        # find ATM strike using delta
        df_front["absdelta"] = df_front["delta"].abs()
        atmK = df_front.sort_values("absdelta").iloc[0]["strike"]

        st.write(f"Using ATM Strike: **{atmK}**")

        results = []

        # loop back expiries > front
        for back_exp in exps:
            if back_exp <= front_exp:
                continue

            df_back = orats.get_strikes(ticker, back_exp)

            # match strike
            f_leg = df_front[df_front["strike"] == atmK].iloc[0]
            b_leg = df_back[df_back["strike"] == atmK].iloc[0]

            # IV slope
            slope = iv_slope(f_leg["smvVol"], b_leg["smvVol"])

            # vega/theta
            vtr = vega_theta_ratio(
                f_leg["vega"], b_leg["vega"],
                f_leg["theta"], b_leg["theta"]
            )

            # theta advantage
            tadv = theta_advantage(f_leg["theta"], b_leg["theta"])

            # cheap long leg
            ivr = iv_ratio(f_leg["smvVol"], b_leg["smvVol"])

            # hover metric
            hover = hover_metric(cores["iv20d"], cores["clsHv20d"])

            # debit (mid)
            debit = (b_leg["callBidPrice"] + b_leg["callAskPrice"]) / 2 \
                    - (f_leg["callBidPrice"] + f_leg["callAskPrice"]) / 2

            # breakeven numeric sim
            prices = np.linspace(stock * 0.8, stock * 1.2, 200)
            long_vals = np.interp(prices, prices, prices*0 + b_leg["callValue"])
            short_vals = np.interp(prices, prices, prices*0 + f_leg["callValue"])

            beL, beU = breakevens(prices, long_vals, short_vals)
            if beL and beU:
                bwidth = beU - beL
                payoff = payoff_ratio(bwidth, debit)
                be_vs_move = bwidth / imp_move_abs
            else:
                payoff = np.nan
                be_vs_move = np.nan

            # score (rough simple total)
            score = 0
            score += (slope > 0) * 5
            score += (vtr > 1.5) * 5
            score += (tadv > 0) * 5
            score += (ivr > 1.05) * 5
            score += (hover > 0) * 5
            score += (payoff is not np.nan and payoff > 3) * 5

            results.append({
                "Back Expiry": back_exp,
                "Debit": round(debit, 3),
                "IV Slope": round(slope, 3),
                "Vega/Theta": round(vtr, 3),
                "Theta Adv": round(tadv, 3),
                "IV Ratio": round(ivr, 3),
                "Hover": round(hover, 3),
                "BE/Move": round(be_vs_move, 3) if not np.isnan(be_vs_move) else None,
                "Payoff Ratio": round(payoff, 3) if not np.isnan(payoff) else None,
                "Score": score
            })

        results_df = pd.DataFrame(results).sort_values("Score", ascending=False)

        st.subheader("📊 Calendar Quality Results")
        st.dataframe(results_df, use_container_width=True)

        # Term structure chart
        st.subheader("📉 ATM Term Structure (ORATS)")
        fig = go.Figure()
        months = ["atmIvM1","atmIvM2","atmIvM3","atmIvM4"]
        fig.add_trace(go.Scatter(
            x=[1,2,3,4],
            y=[cores[m] for m in months],
            mode="lines+markers"
        ))
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="ATM IV",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
