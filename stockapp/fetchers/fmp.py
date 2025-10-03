# stockapp/fetchers/fmp.py
from __future__ import annotations
import streamlit as st

_SHOWN = False

def get_all(ticker: str) -> dict:
    global _SHOWN
    if not _SHOWN:
        st.info("ℹ️ FMP är tillfälligt avstängt (väntar på nyckel/plan).")
        _SHOWN = True
    return {}
