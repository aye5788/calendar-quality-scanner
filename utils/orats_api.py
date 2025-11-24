import requests
import pandas as pd
import streamlit as st

BASE = "https://api.orats.io/datav2"


class ORATS:
    def __init__(self):
        # Read your key from Streamlit secrets
        self.key = st.secrets["ORATS_API_KEY"]

    def _get(self, path: str, params: dict | None = None):
        """Internal helper to call ORATS with token as query param."""
        if params is None:
            params = {}
        # ORATS delayed API expects ?token=... not Authorization header
        params["token"] = self.key
        url = f"{BASE}{path}"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_chains(self, ticker: str):
        """
        /datav2/chains delayed
        params: token, ticker
        """
        data = self._get("/chains", {"ticker": ticker})
        return data  # your app expects a dict with "expirations"

    def get_strikes(self, ticker: str, exp: str | None = None):
        """
        /datav2/strikes delayed
        params: token, ticker, (optional) fields, dte, delta
        We pull ALL expirations and filter by expirDate in app.py.
        """
        data = self._get("/strikes", {"ticker": ticker})
        # API returns {"data":[...]} per docs; fall back if it's a bare list
        rows = data.get("data", data)
        df = pd.DataFrame(rows)

        if "expirDate" not in df.columns:
            raise ValueError("ORATS strikes response missing 'expirDate' column.")

        return df

    def get_cores(self, ticker: str) -> pd.DataFrame:
        """
        /datav2/cores delayed
        params: token, ticker
        """
        data = self._get("/cores", {"ticker": ticker})
        rows = data.get("data", data)
        return pd.DataFrame(rows)

    def get_summaries(self, ticker: str) -> pd.DataFrame:
        """
        /datav2/summaries delayed
        params: token, ticker
        """
        data = self._get("/summaries", {"ticker": ticker})
        rows = data.get("data", data)
        return pd.DataFrame(rows)

