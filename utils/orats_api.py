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
        return r.json()

    def get_strikes(self, ticker, exp):
        url = f"{BASE}/strikes?ticker={ticker}&expiration={exp}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return pd.DataFrame(r.json())

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
