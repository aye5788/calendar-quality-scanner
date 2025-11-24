import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from openai import OpenAI

from utils.orats_api import ORATS
from utils.calendar_metrics import (
    iv_slope,
    iv_ratio,
    vega_theta_ratio,
    theta_advantage,
    hover_metric,
    payoff_ratio,
    calendar_quality_score
)
from utils.breakeven_solver import breakevens

# ---------------------------------------------------------
# STREAMLIT SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Calendar Quality Scanner", layout="wide")
st.title("📈 Calendar Spread Quality Scanner (ORATS)")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
orats = ORATS()

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
ticker = st.text_input("Ticker", "SLV").upper()

if ticker:

    # Load chain expirations
    chains = orats.get_chains(ticker)
    exps = chains["expirations"]

    front_exp = st.selectbox("Select Front Expiry", exps)
    run = st.button("SCAN CALENDARS")

    if run:

        st.subheader("Fetching ORATS Data…")

        summaries = orats.get_summaries(ticker).iloc[0]
        cores = orats.get_cores(ticker).iloc[0]

        stock = summaries["stockPrice"]
        imp_move_abs = stock * summaries["impliedMove"]

        df_all = orats.get_strikes(ticker)

        # Identify ATM strike
        df_front = df_all[df_all["expirDate"] == front_exp]
        df_front["absdelta"] = df_front["delta"].abs()
        atmK = df_front.sort_values("absdelta").iloc[0]["strike"]

        st.write(f"Using ATM Strike: **{atmK}**")

        f_leg = df_front[df_front["strike"] == atmK].iloc[0]

        results = []

        # ---------------------------------------------------------
        # BUILD ALL BACK-EXPIRY CALENDARS
        # ---------------------------------------------------------
        for back_exp in exps:

            if back_exp <= front_exp:
                continue

            df_back = df_all[df_all["expirDate"] == back_exp]
            if df_back.empty:
                continue
            if atmK not in df_back["strike"].values:
                continue

            b_leg = df_back[df_back["strike"] == atmK].iloc[0]

            # ---- Metrics ----
            slope = iv_slope(f_leg["smvVol"], b_leg["smvVol"])
            ivr = iv_ratio(f_leg["smvVol"], b_leg["smvVol"])
            tadv = theta_advantage(f_leg["theta"], b_leg["theta"])
            vtr = vega_theta_ratio(
                f_leg["vega"], b_leg["vega"],
                f_leg["theta"], b_leg["theta"]
            )

            # Debit
            debit = (
                (b_leg["callBidPrice"] + b_leg["callAskPrice"]) / 2
                - (f_leg["callBidPrice"] + f_leg["callAskPrice"]) / 2
            )

            # Breakevens
            prices = np.linspace(stock * 0.8, stock * 1.2, 200)
            long_vals = np.full_like(prices, b_leg["callValue"])
            short_vals = np.full_like(prices, f_leg["callValue"])

            beL, beU = breakevens(prices, long_vals, short_vals, debit)
            if beL is not None:
                bwidth = beU - beL
                be_vs_move = bwidth / imp_move_abs
            else:
                be_vs_move = None

            # Add to list
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

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # ---------------------------------------------------------
        # UNIVERSAL RESEARCH-BASED SCORING
        # ---------------------------------------------------------
        all_debits = df["Debit"].tolist()
        hover = hover_metric(cores["iv20d"], cores["clsHv20d"])

        df["Score"] = df.apply(
            lambda r: calendar_quality_score(r, all_debits, hover),
            axis=1
        )

        df = df.sort_values("Score", ascending=False)

        # ---------------------------------------------------------
        # RESULTS TABLE
        # ---------------------------------------------------------
        st.subheader("📊 Calendar Quality Results (Research-Based Scoring)")
        st.dataframe(df, use_container_width=True)

        # ---------------------------------------------------------
        # TERM STRUCTURE CHART
        # ---------------------------------------------------------
        st.subheader("📉 ATM Term Structure (ORATS)")
        fig = go.Figure()
        months = ["atmIvM1", "atmIvM2", "atmIvM3", "atmIvM4"]
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4],
            y=[cores[m] for m in months],
            mode="lines+markers"
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # AI INTERPRETATION
        # ---------------------------------------------------------
        enable_ai = st.checkbox("Enable AI Interpretation", value=True)

        if enable_ai:
            st.subheader("🤖 AI Interpretation")

            # Compress table for tokens
            text_table = df.to_string(index=False)

            # FULL RULE-BASED PROMPT
            prompt = f"""
You are an options strategist. Analyze the following calendar-spread scan for {ticker}.
Use strict professional calendar-spread rules. DO NOT reverse legs.

------------------------------------
SCAN DATA (front expiry = {front_exp}, ATM strike = {atmK})
------------------------------------
{text_table}

ATM TERM STRUCTURE:
{json.dumps({m: cores[m] for m in months})}

------------------------------------
CALENDAR SPREAD RULES YOU MUST APPLY
------------------------------------
TERM STRUCTURE:
- Upward or flat curve is favorable.
- IV Ratio ≥1.02 = good, 0.97–1.02 = neutral, <0.97 = weak.

THETA STRUCTURE:
- Short leg must decay faster.
- ThetaAdv > 0 = good; > -0.02 acceptable; < -0.02 red flag.

VEGA/THETA:
- VTR ≥1.3 strong; 0.7–1.3 moderate; <0.7 weak.
- Missing VTR is NOT fatal.

DEBIT:
- Lower debit or near median preferred.
- Very high debits reduce quality.

BREAKEVENS:
- BE/Move ≥1.2 excellent; 0.8–1.2 acceptable; <0.8 tight.
- If missing, note ORATS-delayed payoff flattening.

VOL REGIME:
- hover = iv20d - hv20d.
- Slight positive hover = fine.
- Large positive hover = vol-rich (crush risk).

TRADE CONSTRUCTION:
***ALWAYS BUY the BACK expiry and SELL the FRONT expiry.***
NEVER reverse this.

------------------------------------
OUTPUT REQUIREMENTS (≤180 words)
------------------------------------
1. Interpret ATM term structure.
2. Identify top back expirations and WHY they score well.
3. Identify risks / red flags.
4. Give a trade recommendation:
   - “Buy BACK expiry, Sell FRONT expiry”
   - Use ATM strike {atmK}
   - Include fair debit estimate, risk notes, and exit plan.
"""

            ai_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5
            )

            st.write(ai_resp.choices[0].message.content)

