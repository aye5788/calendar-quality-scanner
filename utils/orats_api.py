import requests
import pandas as pd
import streamlit as st

BASE = "https://api.orats.io/datav2"

class ORATS:
    def __init__(self):
        self.key = st.secrets["ORATS_API_KEY"]
        self.headers = {"Authorization": f"Bearer {self.key}"}

    def get_chains(self, ticker):
        url = f"{BASE}/chains?ticker={ticker}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()  # unchanged for compatibility

    def get_strikes(self, ticker, exp=None):
        """
        exp is ignored (kept only for compatibility with your existing code)
        ORATS delayed endpoint returns ALL expirations.
        """
        url = f"{BASE}/strikes?ticker={ticker}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()

        js = r.json()
        data = js.get("data", js)  # supports both {data:[]} and [] formats
        df = pd.DataFrame(data)

        if "expirDate" not in df.columns:
            raise ValueError("ORATS strikes missing 'expirDate' field.")

        # Simply return df. Filtering happens in app.py.
        return df

    def get_cores(self, ticker):
        url = f"{BASE}/cores?ticker={ticker}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return pd.DataFrame(r.json())

    def get_summaries(self, ticker):
        url = f"{BASE}/summaries?ticker={ticker}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return pd.DataFrame(r.json())
