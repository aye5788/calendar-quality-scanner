import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

from utils.orats_api import ORATS
from utils.calendar_metrics import (
    iv_slope, iv_ratio, theta_advantage,
    vega_theta_ratio, hover_metric, payoff_ratio,
    calendar_quality_score
)
from utils.breakeven_solver import breakevens
from openai import OpenAI

st.set_page_config(page_title="Calendar Quality Scanner", layout="wide")
st.title("📈 Calendar Spread Quality Scanner (ORATS)")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
orats = ORATS()

ticker = st.text_input("Ticker", "SLV").upper()

if ticker:

    chains = orats.get_chains(ticker)
    exps = chains["expirations"]

    front_exp = st.selectbox("Select Front Expiry", exps)
    run = st.button("SCAN CALENDARS")

    if run:

        summaries = orats.get_summaries(ticker).iloc[0]
        cores = orats.get_cores(ticker).iloc[0]

        stock = summaries["stockPrice"]
        imp_move_abs = stock * summaries["impliedMove"]

        df_all = orats.get_strikes(ticker)

        df_front = df_all[df_all["expirDate"] == front_exp]
        df_front["absdelta"] = df_front["delta"].abs()
        atmK = df_front.sort_values("absdelta").iloc[0]["strike"]

        f_leg = df_front[df_front["strike"] == atmK].iloc[0]

        results = []

        for back_exp in exps:
            if back_exp <= front_exp:
                continue

            df_back = df_all[df_all["expirDate"] == back_exp]
            if df_back.empty or atmK not in df_back["strike"].values:
                continue

            b_leg = df_back[df_back["strike"] == atmK].iloc[0]

            slope = iv_slope(f_leg["smvVol"], b_leg["smvVol"])
            ivr = iv_ratio(f_leg["smvVol"], b_leg["smvVol"])
            tadv = theta_advantage(f_leg["theta"], b_leg["theta"])
            vtr = vega_theta_ratio(
                f_leg["vega"], b_leg["vega"],
                f_leg["theta"], b_leg["theta"]
            )

            debit = (
                (b_leg["callBidPrice"] + b_leg["callAskPrice"]) / 2
                - (f_leg["callBidPrice"] + f_leg["callAskPrice"]) / 2
            )

            prices = np.linspace(stock * 0.8, stock * 1.2, 200)
            long_vals = np.full_like(prices, b_leg["callValue"])
            short_vals = np.full_like(prices, f_leg["callValue"])

            beL, beU = breakevens(prices, long_vals, short_vals, debit)
            if beL is not None:
                bwidth = beU - beL
                be_vs_move = bwidth / imp_move_abs
            else:
                be_vs_move = None

            results.append({
                "Back Expiry": back_exp,
                "Debit": round(debit, 3),
                "IV Slope": round(slope, 3),
                "Vega/Theta": None if np.isnan(vtr) else round(vtr, 3),
                "Theta Adv": round(tadv, 3),
                "IV Ratio": round(ivr, 3),
                "Hover": round(hover_metric(cores["iv20d"], cores["clsHv20d"]), 3),
                "BE/Move": round(be_vs_move, 3) if be_vs_move else None,
            })

        df = pd.DataFrame(results)

        # NEW SCORING
        all_debits = df["Debit"].tolist()
        hover = hover_metric(cores["iv20d"], cores["clsHv20d"])

        df["Score"] = df.apply(lambda r: calendar_quality_score(r, all_debits, hover), axis=1)

        df = df.sort_values("Score", ascending=False)

        st.subheader("📊 Calendar Quality Results (Research-Based Scoring)")
        st.dataframe(df, use_container_width=True)

        st.subheader("📉 ATM Term Structure (ORATS)")
        fig = go.Figure()
        months = ["atmIvM1","atmIvM2","atmIvM3","atmIvM4"]
        fig.add_trace(go.Scatter(
            x=[1,2,3,4],
            y=[cores[m] for m in months],
            mode="lines+markers"
        ))
        st.plotly_chart(fig, use_container_width=True)

        # =============================
        # AI INTERPRETATION
        # =============================

        enable_ai = st.checkbox("Enable AI Interpretation", value=True)

        if enable_ai:
            st.subheader("🤖 AI Interpretation")

            # compressed for tokens:
            text_table = df.to_string()

            prompt = f"""
Analyze calendar spread quality for {ticker}.

Metrics (top rows):
{text_table}

ATM term structure:
{json.dumps({m: cores[m] for m in months})}

Provide:
1. Term structure interpretation.
2. Strongest expirations & why.
3. Red flags.
4. Bottom-line summary (≤150 words).

MANDATORY TRADE RECOMMENDATION:
- Recommend ONE calendar (front={front_exp}, best back).
- Use ATM strike {atmK}.
- Provide fair debit estimate.
- Include risk notes & exit conditions.
"""

            ai_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250
            )

            st.write(ai_resp.choices[0].message.content)
