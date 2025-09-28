# stockapp/rates.py
# -*- coding: utf-8 -*-
"""
Valutamodul:
- Läser/sparar FX-kurser (USD/EUR/NOK/CAD → SEK) i ett separat blad i ditt Google Sheet
- Har en UI-funktion (sidebar_rates) som tryggt hanterar Streamlits session_state
- Kan hämta kurser automatiskt via FMP → Frankfurter → exchangerate.host (fallback)

Användning i app.py:
    from stockapp.rates import sidebar_rates
    user_rates = sidebar_rates()
"""

from typing import Dict, Tuple, Optional, List
import time
import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =========================
# Konstanter & defaults
# =========================

SHEET_URL: str = st.secrets["SHEET_URL"]
RATES_SHEET_NAME: str = "Valutakurser"

STANDARD_VALUTAKURSER: Dict[str, float] = {
    "USD": 9.75,
    "EUR": 11.18,
    "NOK": 0.95,
    "CAD": 7.05,
    "SEK": 1.0,
}

# FMP setup (om nyckel finns i secrets)
FMP_BASE: str = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY: str = st.secrets.get("FMP_API_KEY", "")

# =========================
# Google Sheets-koppling
# =========================

_scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
_credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=_scope)
_client = gspread.authorize(_credentials)


def _with_backoff(func, *args, **kwargs):
    """Liten backoff-hjälpare för att mildra 429/kvotfel."""
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err


def _get_spreadsheet():
    return _client.open_by_url(SHEET_URL)


def _ensure_rates_sheet():
    ss = _get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta", "Kurs"]])
        return ws


# =========================
# I/O för sparade kurser
# =========================

@st.cache_data(show_spinner=False)
def read_saved_rates_cached(nonce: int) -> Dict[str, float]:
    """Läser sparade kurser från 'Valutakurser'-bladet."""
    ws = _ensure_rates_sheet()
    rows = _with_backoff(ws.get_all_records)  # [{'Valuta': 'USD', 'Kurs': '10.10'}, ...]
    out: Dict[str, float] = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except Exception:
            pass
    return out


def read_saved_rates() -> Dict[str, float]:
    """Wrapper som tillåter invalidation via session_state.rates_reload."""
    return read_saved_rates_cached(st.session_state.get("rates_reload", 0))


def save_rates(rates: Dict[str, float]) -> None:
    """Sparar kurser till 'Valutakurser'-bladet i standardordningen."""
    ws = _ensure_rates_sheet()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "EUR", "NOK", "CAD", "SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)


# =========================
# Automatisk hämtning
# =========================

def _fetch_fmp_pair(pair: str) -> Optional[float]:
    if not FMP_KEY:
        return None
    try:
        url = f"{FMP_BASE}/api/v3/fx/{pair}"
        r = requests.get(url, params={"apikey": FMP_KEY}, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json() or {}
        px = j.get("price")
        return float(px) if px is not None else None
    except Exception:
        return None


def fetch_rates_auto() -> Tuple[Dict[str, float], List[str], str]:
    """
    Försök i ordning:
      1) FMP (om API-nyckel finns)
      2) Frankfurter (ECB)
      3) exchangerate.host
    Returnerar (rates, misses, provider)
    """
    misses: List[str] = []
    rates: Dict[str, float] = {}
    provider = None

    # 1) FMP
    if FMP_KEY:
        provider = "FMP"
        for pair in ("USDSEK", "EURSEK", "NOKSEK", "CADSEK"):
            v = _fetch_fmp_pair(pair)
            if v and v > 0:
                base_ccy = pair[:3]
                rates[base_ccy] = float(v)
            else:
                misses.append(pair)

    # 2) Frankfurter
    if len(rates) < 4:
        provider = "Frankfurter"
        for base_ccy in ("USD", "EUR", "NOK", "CAD"):
            try:
                r2 = requests.get("https://api.frankfurter.app/latest",
                                  params={"from": base_ccy, "to": "SEK"}, timeout=10)
                if r2.status_code == 200:
                    rr = r2.json() or {}
                    v = (rr.get("rates") or {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # 3) exchangerate.host
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ("USD", "EUR", "NOK", "CAD"):
            try:
                r = requests.get("https://api.exchangerate.host/latest",
                                 params={"base": base_ccy, "symbols": "SEK"}, timeout=10)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # Fyll luckor med sparat/standard
    saved = read_saved_rates()
    for base_ccy in ("USD", "EUR", "NOK", "CAD"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    return rates, misses, (provider or "okänd")


# =========================
# Sidebar-UI (säker mot state-problem)
# =========================

def sidebar_rates() -> Dict[str, float]:
    """
    Renderar sidopanelen för valutor och returnerar ett dict:
        {"USD":..., "EUR":..., "NOK":..., "CAD":..., "SEK": 1.0}
    Viktigt: sätter session_state-keys INNAN inputs skapas.
    """
    st.sidebar.header("💱 Valutakurser → SEK")

    # 1) Initiera state en gång från sparat/standard
    saved = read_saved_rates()
    defaults = {
        "USD": float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])),
        "EUR": float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
        "NOK": float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
        "CAD": float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
    }
    keymap = {
        "USD": "rate_usd_input",
        "EUR": "rate_eur_input",
        "NOK": "rate_nok_input",
        "CAD": "rate_cad_input",
    }
    # se till att keys finns före inputs
    for ccy, key in keymap.items():
        if key not in st.session_state:
            st.session_state[key] = defaults[ccy]

    # 2) Actions FÖRE inputs (för att undvika "cannot be modified after widget..."-felet)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        auto_btn = st.button("🌐 Hämta kurser automatiskt", key="btn_auto_rates")
    with c2:
        load_btn = st.button("↻ Läs sparade kurser", key="btn_load_saved_rates")

    if auto_btn:
        auto_rates, misses, provider = fetch_rates_auto()
        for ccy, key in keymap.items():
            if ccy in auto_rates and auto_rates[ccy]:
                try:
                    st.session_state[key] = float(auto_rates[ccy])
                except Exception:
                    pass
        st.sidebar.success(f"Valutakurser uppdaterade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Kunde inte hämta:\n- " + "\n- ".join(misses))

    if load_btn:
        saved2 = read_saved_rates()
        for ccy, key in keymap.items():
            try:
                st.session_state[key] = float(saved2.get(ccy, st.session_state[key]))
            except Exception:
                pass
        st.sidebar.info("Läste in sparade kurser.")

    # 3) Inputs
    usd = st.sidebar.number_input("USD → SEK", key=keymap["USD"], step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key=keymap["EUR"], step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key=keymap["NOK"], step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key=keymap["CAD"], step=0.01, format="%.4f")

    # 4) Spara-knapp (ingen rerun – bara state och info)
    if st.sidebar.button("💾 Spara kurser", key="btn_save_rates"):
        to_save = {
            "USD": float(usd),
            "EUR": float(eur),
            "NOK": float(nok),
            "CAD": float(cad),
            "SEK": 1.0
        }
        save_rates(to_save)
        st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
        st.sidebar.success("Valutakurser sparade.")

    # 5) Returnera
    return {
        "USD": float(usd),
        "EUR": float(eur),
        "NOK": float(nok),
        "CAD": float(cad),
        "SEK": 1.0,
    }
