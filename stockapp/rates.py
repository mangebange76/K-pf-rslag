# stockapp/rates.py
from __future__ import annotations
import requests
import streamlit as st
import gspread
from typing import Dict, List
from google.oauth2.service_account import Credentials

# Rimliga defaultvärden (används bara om inget annat finns)
DEFAULT_RATES: Dict[str, float] = {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0}
RATES_SHEET_NAME = st.secrets.get("RATES_WORKSHEET_NAME", "Valutakurser")
CURRENCIES = ["USD", "NOK", "CAD", "EUR"]

# ---------- Google Sheets ----------
def _gs_client() -> gspread.Client:
    creds = st.secrets.get("GOOGLE_CREDENTIALS")
    if not creds:
        raise RuntimeError("GOOGLE_CREDENTIALS saknas i secrets.")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    return gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))

def _open_sheet():
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    return _gs_client().open_by_url(url)

def _get_or_create_rates_ws():
    ss = _open_sheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=RATES_SHEET_NAME, rows=20, cols=5)
        ws.update([["Valuta", "Kurs"]])
        return ws

def read_rates() -> Dict[str, float]:
    """Läs VALUTA->SEK från bladet 'Valutakurser'."""
    try:
        ws = _get_or_create_rates_ws()
        rows = ws.get_all_records()
        out: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            raw = str(r.get("Kurs", "")).replace(" ", "").replace(",", ".")
            try:
                val = float(raw)
            except Exception:
                continue
            if _is_plausible(val):
                out[cur] = val
        for k, v in DEFAULT_RATES.items():  # säkerställ alla nycklar
            out.setdefault(k, v)
        out["SEK"] = 1.0
        return out
    except Exception:
        return dict(DEFAULT_RATES)

def save_rates(rates: Dict[str, float]) -> None:
    ws = _get_or_create_rates_ws()
    order = CURRENCIES + ["SEK"]
    body: List[List[str]] = [["Valuta", "Kurs"]]
    for k in order:
        v = float(rates.get(k, DEFAULT_RATES.get(k, 1.0)))
        body.append([k, f"{v:.6f}"])
    ws.clear(); ws.update(body)

# ---------- Livehämtning ----------
def _is_plausible(x: float) -> bool:
    # En rimlig växelkurs VALUTA->SEK bör hamna här
    return 0.05 <= x <= 200.0

def _host_inverted() -> Dict[str, float]:
    """exchangerate.host  (1 SEK = X VALUTA) -> invertera."""
    url = "https://api.exchangerate.host/latest"
    r = requests.get(url, params={"base": "SEK", "symbols": ",".join(CURRENCIES)}, timeout=15)
    r.raise_for_status()
    rates = r.json().get("rates", {}) or {}
    out = {}
    for cur in CURRENCIES:
        v = float(rates.get(cur, 0.0))
        if v > 0:
            inv = 1.0 / v
            if _is_plausible(inv):
                out[cur] = round(inv, 6)
    return out

def _frankfurter_one(base: str) -> float:
    """frankfurter.app  base->SEK (direkt i SEK)."""
    r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": "SEK"}, timeout=15)
    r.raise_for_status()
    return float(r.json().get("rates", {}).get("SEK", 0.0))

def _erapi_one(base: str) -> float:
    """open.er-api.com  base->SEK."""
    r = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=15)
    r.raise_for_status()
    return float(r.json().get("rates", {}).get("SEK", 0.0))

def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar VALUTA->SEK med robust fallback:
      1) exchangerate.host (inverterad)
      2) frankfurter.app (per valuta)
      3) open.er-api.com (per valuta)
    """
    out: Dict[str, float] = {}
    # Försök 1 – en request och invertera
    try:
        out.update(_host_inverted())
    except Exception:
        pass

    # Saknas någon? komplettera med frankfurter och sedan er-api
    for cur in CURRENCIES:
        if cur in out and _is_plausible(out[cur]):
            continue
        ok = 0.0
        # 2) frankfurter
        try:
            ok = float(_frankfurter_one(cur))
        except Exception:
            ok = 0.0
        if _is_plausible(ok):
            out[cur] = round(ok, 6)
            continue
        # 3) er-api
        try:
            ok = float(_erapi_one(cur))
        except Exception:
            ok = 0.0
        if _is_plausible(ok):
            out[cur] = round(ok, 6)

    # Fyll ev. luckor med default
    for k in CURRENCIES:
        out.setdefault(k, DEFAULT_RATES[k])
    out["SEK"] = 1.0
    return out
