import requests
import pandas as pd
import streamlit as st

BASE = "https://api.orats.io/datav2"


class ORATS:
    def __init__(self):
        # Read API key from Streamlit secrets
        self.key = st.secrets["ORATS_API_KEY"]

    def _get(self, path: str, params: dict | None = None):
        """Internal helper to call ORATS with token as query param."""
        if params is None:
            params = {}
        params["token"] = self.key  # IMPORTANT: ORATS delayed requires token here
        url = f"{BASE}{path}"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_chains(self, ticker: str):
        """Correct delayed /chains endpoint"""
        data = self._get("/chains", {"ticker": ticker})
        return data

    def get_strikes(self, ticker: str):
        """Correct delayed /strikes endpoint"""
        data = self._get("/strikes", {"ticker": ticker})
        rows = data.get("data", data)
        return pd.DataFrame(rows)

    def get_cores(self, ticker: str) -> pd.DataFrame:
        """Correct delayed /cores endpoint"""
        data = self._get("/cores", {"ticker": ticker})
        rows = data.get("data", data)
        return pd.DataFrame(rows)

    def get_summaries(self, ticker: str) -> pd.DataFrame:
        """Correct delayed /summaries endpoint"""
        data = self._get("/summaries", {"ticker": ticker})
        rows = data.get("data", data)
        return pd.DataFrame(rows)

