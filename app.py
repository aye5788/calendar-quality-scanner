import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json

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

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------
st.set_page_config(page_title="Calendar Quality Scanner", layout="wide")
st.title("📈 Calendar Spread Quality Scanner (ORATS)")

# -------------------------------------------------------
# OPENAI INIT
# -------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

@st.cache_data(show_spinner=False)
def openai_interpretation_cached(cache_key, df, cores, ticker):
    return interpret_with_openai(df, cores, ticker)

def interpret_with_openai(results_df, cores, ticker):
    """
    Sends pruned calendar metrics + term structure to OpenAI
    with token-limiting safeguards.
    """
    if OPENAI_API_KEY is None:
        return "⚠ No OPENAI_API_KEY found in Streamlit secrets."

    # Only keep essential columns
    cols_to_keep = [
        "Back Expiry", "Debit", "IV Slope",
        "Vega/Theta", "Theta Adv", "IV Ratio",
        "Hover", "BE/Move", "Payoff Ratio", "Score"
    ]
    pruned = results_df[cols_to_keep]

    # Only send top 5 rows for context (major token savings)
    text_table = pruned.head(5).to_string(index=False)

    # Term structure summary
    term_structure = {
        "atmIvM1": float(cores.get("atmIvM1", 0)),
        "atmIvM2": float(cores.get("atmIvM2", 0)),
        "atmIvM3": float(cores.get("atmIvM3", 0)),
        "atmIvM4": float(cores.get("atmIvM4", 0)),
    }

    # Highly efficient prompt
    prompt = f"""
Analyze calendar spread quality for {ticker}.

Metrics (top 5 rows):
{text_table}

ATM term structure:
{json.dumps(term_structure)}

Provide:
1. Term structure interpretation (impact on calendars)
2. Best back expirations and why
3. Any red flags (debit, IV slope, vega/theta)
4. Simple bottom-line recommendation

Keep under 150 words.
"""

    # OpenAI API call — lightweight & capped
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        json={
            "model": "gpt-4o-mini",  # cheapest model for this task
            "messages": [
                {"role": "system", "content": "You are a professional calendar-spread analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.25,
            "max_tokens": 250     # HARD token limit
        }
    )

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI API error: {response.text}"


# -------------------------------------------------------
# ORATS INIT
# -------------------------------------------------------
orats = ORATS()

ticker = st.text_input("Ticker", "SLV").upper()

if ticker:

    st.subheader("Fetching ORATS Strikes…")

    df_all = orats.get_strikes(ticker)

    if df_all.empty:
        st.error("No strikes returned from ORATS for this ticker.")
        st.stop()

    exps = sorted(df_all["expirDate"].unique())

    if not exps:
        st.error("No expirations found in ORATS strikes data.")
        st.stop()

    front_exp = st.selectbox("Select Front Expiry", exps)
    run = st.button("SCAN CALENDARS")

    if run:
        st.subheader("Loading ORATS Core + Summary Data…")

        cores_df = orats.get_cores(ticker)
        summaries_df = orats.get_summaries(ticker)

        if cores_df.empty or summaries_df.empty:
            st.error("Cores or Summaries data missing.")
            st.stop()

        cores = cores_df.iloc[0]
        summaries = summaries_df.iloc[0]

        stock = summaries["stockPrice"]
        imp_move_abs = stock * summaries["impliedMove"]

        # Filter front expiry
        df_front = df_all[df_all["expirDate"] == front_exp].copy()
        df_front["absdelta"] = df_front["delta"].abs()
        atmK = df_front.sort_values("absdelta").iloc[0]["strike"]

        st.write(f"Using ATM Strike: **{atmK}**")

        f_leg = df_front[df_front["strike"] == atmK].iloc[0]

        # -------------------------------------------------------
        # SCAN ALL BACK-MONTH CALENDARS
        # -------------------------------------------------------
        results = []

        for back_exp in exps:
            if back_exp <= front_exp:
                continue

            df_back = df_all[df_all["expirDate"] == back_exp]
            if df_back.empty:
                continue

            if atmK not in df_back["strike"].values:
                continue

            b_leg = df_back[df_back["strike"] == atmK].iloc[0]

            # Metrics
            slope = iv_slope(f_leg["smvVol"], b_leg["smvVol"])
            vtr = vega_theta_ratio(f_leg["vega"], b_leg["vega"], f_leg["theta"], b_leg["theta"])
            tadv = theta_advantage(f_leg["theta"], b_leg["theta"])
            ivr = iv_ratio(f_leg["smvVol"], b_leg["smvVol"])
            hover = hover_metric(cores["iv20d"], cores["clsHv20d"])

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
                payoff = payoff_ratio(bwidth, debit)
            else:
                bwidth = np.nan
                be_vs_move = np.nan
                payoff = np.nan

            score = (
                (slope > 0) * 5 +
                (vtr > 1.5) * 5 +
                (tadv > 0) * 5 +
                (ivr > 1.05) * 5 +
                (hover > 0) * 5 +
                (not np.isnan(payoff) and payoff > 3) * 5
            )

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

        # -------------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------------
        st.subheader("📊 Calendar Quality Results")
        st.dataframe(results_df, use_container_width=True)

        # -------------------------------------------------------
        # TERM STRUCTURE CHART
        # -------------------------------------------------------
        st.subheader("📉 ATM Term Structure (ORATS)")
        fig = go.Figure()
        months = ["atmIvM1","atmIvM2","atmIvM3","atmIvM4"]
        fig.add_trace(go.Scatter(
            x=[1,2,3,4],
            y=[cores[m] for m in months],
            mode="lines+markers"
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------
        # AI INTERPRETATION (OPTIONAL TOGGLE)
        # -------------------------------------------------------
        use_ai = st.checkbox("Enable AI Interpretation", value=True)

        if use_ai:
            st.subheader("🤖 AI Interpretation")
            cache_key = results_df.head(5).to_json()
            ai_text = openai_interpretation_cached(cache_key, results_df, cores, ticker)
            st.write(ai_text)

