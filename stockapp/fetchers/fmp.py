# stockapp/fetchers/fmp.py
from __future__ import annotations
import streamlit as st

# En enda global flagga så vi inte spammar varningar i UI
_SHOWN = False

def get_all(ticker: str) -> dict:
    """
    FMP är tillfälligt avstängt (väntar på nyckel/plan).
    Denna stub gör att ev. import i manual_collect inte kraschar.
    """
    global _SHOWN
    if not _SHOWN:
        st.info("ℹ️ FMP är tillfälligt avstängt. (Stub används – inga FMP-fält hämtas.)")
        _SHOWN = True
    return {}  # tomt => inget att slå ihop
