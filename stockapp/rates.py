# stockapp/rates.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math

import pandas as pd
import streamlit as st

try:
    import requests
except Exception:
    requests = None  # type: ignore

from .sheets import get_spreadsheet, _with_backoff

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
RATES_WS_TITLE: str = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

# Basvärden (fallback om varken live- eller sparade kurser finns)
DEFAULT_RATES: Dict[str, float] = {
    "USD": 10.50,
    "NOK": 1.00,
    "CAD": 7.50,
    "EUR": 11.50,
    "SEK": 1.00,
}

CANON: List[str] = ["USD", "NOK", "CAD", "EUR", "SEK"]


# ------------------------------------------------------------
# Interna hjälp-funktioner
# ------------------------------------------------------------
def _num(x) -> float:
    """Robust float-parser för strängar från Sheets (hanterar kommatecken, tomt etc)."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            if isinstance(x, float) and math.isnan(x):
                return 0.0
        s = str(x).strip().replace(" ", "").replace(",", ".")
        if not s or s.lower() in {"na", "n/a", "null", "none", "-"}:
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _ensure_ws():
    """Se till att bladet för valutakurser finns och har minst headern."""
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, RATES_WS_TITLE)
        # Finns, men kontrollera headern – om tomt, fyll på
        vals = _with_backoff(ws.get_all_values)
        if not vals:
            _with_backoff(ws.update, [["Valuta", "Kurs"]], value_input_option="USER_ENTERED")
        return ws
    except Exception:
        pass

    # Skapa nytt blad
    ws = _with_backoff(ss.add_worksheet, title=RATES_WS_TITLE, rows=20, cols=5)
    _with_backoff(ws.update, [["Valuta", "Kurs"]], value_input_option="USER_ENTERED")
    # Fyll med default
    defaults = [["USD", f"{DEFAULT_RATES['USD']:.6f}"],
                ["NOK", f"{DEFAULT_RATES['NOK']:.6f}"],
                ["CAD", f"{DEFAULT_RATES['CAD']:.6f}"],
                ["EUR", f"{DEFAULT_RATES['EUR']:.6f}"],
                ["SEK", "1.000000"]]
    _with_backoff(ws.append_rows, defaults, value_input_option="USER_ENTERED")
    return ws


# ------------------------------------------------------------
# Publika funktioner
# ------------------------------------------------------------
def read_rates() -> Dict[str, float]:
    """
    Läs valutakurser → SEK från bladet 'Valutakurser'.
    Returnerar dict med nycklar USD/NOK/CAD/EUR/SEK.
    """
    out = DEFAULT_RATES.copy()
    try:
        ws = _ensure_ws()
        # get_all_records är praktiskt här
        rows = _with_backoff(ws.get_all_records)
        tmp: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).strip().upper()
            val = _num(r.get("Kurs"))
            if cur:
                tmp[cur] = val
        # Mappa tillbaka på våra CANON-nycklar
        for k in CANON:
            if k in tmp and tmp[k] > 0:
                out[k] = float(tmp[k])
        # SEK ska alltid vara 1
        out["SEK"] = 1.0
        return out
    except Exception:
        # Fallback till defaults
        out["SEK"] = 1.0
        return out


def save_rates(rates: Dict[str, float]) -> None:
    """
    Spara (skriv över) valutakurser i bladet. Förväntar SEK=1.0.
    """
    ws = _ensure_ws()
    body = [["Valuta", "Kurs"]]
    for k in CANON:
        v = float(rates.get(k, DEFAULT_RATES.get(k, 1.0)))
        if k.upper() == "SEK":
            v = 1.0
        body.append([k, f"{v:.6f}"])

    # Töm och skriv
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body, value_input_option="USER_ENTERED")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar livekurser via exchangerate.host.
    Returnerar SEK-värden för USD/NOK/CAD/EUR (och SEK=1.0).
    Har robust fallback till sparade/DEFAULT_RATES.
    """
    base = read_rates()  # fallback-bas
    result = base.copy()

    if requests is None:
        result["SEK"] = 1.0
        return result

    try:
        # Hämta alla på en gång med bas = SEK, sen invertera.
        # Ex: rates["USD"] = USD per 1 SEK → USD→SEK = 1 / rates["USD"]
        url = "https://api.exchangerate.host/latest"
        params = {"base": "SEK", "symbols": "USD,NOK,CAD,EUR"}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        rates = (data or {}).get("rates", {}) or {}
        # invertera
        for k in ["USD", "NOK", "CAD", "EUR"]:
            v = float(rates.get(k, 0.0) or 0.0)
            if v > 0:
                result[k] = float(1.0 / v)
        result["SEK"] = 1.0
        return result
    except Exception:
        # Fallback: returnera sparade
        result["SEK"] = 1.0
        return result


def repair_rates_sheet() -> None:
    """
    Säkerställ att bladet finns och innehåller rätt header/rader.
    Överskriver inte befintliga icke-tomma värden i onödan.
    """
    ws = _ensure_ws()

    # Läs nuvarande
    try:
        existing = _with_backoff(ws.get_all_records)
    except Exception:
        existing = []

    by_cur = {str(r.get("Valuta", "")).strip().upper(): r for r in existing if r}

    # Bygg uppdateringslista i kanonisk ordning
    final_rows: List[List[str]] = [["Valuta", "Kurs"]]
    for k in CANON:
        if k in by_cur:
            v = _num(by_cur[k].get("Kurs"))
            if k == "SEK":
                v = 1.0
            final_rows.append([k, f"{v:.6f}"])
        else:
            # saknades → lägg default
            v = DEFAULT_RATES.get(k, 1.0)
            if k == "SEK":
                v = 1.0
            final_rows.append([k, f"{v:.6f}"])

    # Skriv försiktigt (clear + update)
    _with_backoff(ws.clear)
    _with_backoff(ws.update, final_rows, value_input_option="USER_ENTERED")
