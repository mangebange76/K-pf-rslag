# stockapp/rates.py
# -*- coding: utf-8 -*-
"""
Valutafunktioner (kvottsnåla & robusta):
- las_sparade_valutakurser()     → använder session-cache (TTL) före Sheets-läsning
- spara_valutakurser(rates)      → skriver + uppdaterar session-cache
- hamta_valutakurser_auto()      → FMP → Frankfurter → exchangerate.host
- hamta_valutakurs(valuta, user_rates)

Skydd:
- st.cache_resource för gspread-klient & Spreadsheet
- Circuit breaker vid 429 (blockerar fler läsningar en kort tid)
- Fallback till STANDARD_VALUTAKURSER utan att krascha
"""

from __future__ import annotations
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

import requests
import streamlit as st

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

from .config import STANDARD_VALUTAKURSER, RATES_SHEET_NAME  # t.ex. "Valutakurser"
from .utils import with_backoff

# ---------- Konstanter ----------
_CACHE_TTL_MIN = 10          # minuter för session-cache av valutakurser
_BLOCK_SECONDS_ON_429 = 90   # sekunder att pausa nya läsningar efter en 429


# ---------- Hjälpare ----------
def _now() -> datetime:
    try:
        import pytz
        return datetime.now(pytz.timezone("Europe/Stockholm"))
    except Exception:
        return datetime.now()

def _normalize_title(s: str) -> str:
    return (s or "").strip().lower()


# ---------- Cachat gspread ----------
@st.cache_resource(show_spinner=False)
def _gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(credentials)

@st.cache_resource(show_spinner=False)
def _get_spreadsheet():
    client = _gspread_client()
    return client.open_by_url(st.secrets["SHEET_URL"])


def _find_ws_loose(ss: gspread.Spreadsheet, title: str):
    """Hitta worksheet via case-insensitive + trim-match."""
    want = _normalize_title(title)
    for ws in with_backoff(ss.worksheets):
        if _normalize_title(ws.title) == want:
            return ws
    return None

def _ensure_header(ws: gspread.Worksheet):
    """Säkerställ rubriker i valutabladet."""
    try:
        rows = with_backoff(ws.get_all_values)
        if not rows or not rows[0] or rows[0][:2] != ["Valuta", "Kurs"]:
            with_backoff(ws.clear)
            with_backoff(ws.update, [["Valuta", "Kurs"]])
    except Exception:
        try:
            with_backoff(ws.update, [["Valuta", "Kurs"]])
        except Exception:
            pass


def _get_rates_ws():
    """
    Försöker öppna (eller skapa) valutabladet.
    Respekterar circuit breaker vid 429.
    Returnerar worksheet-objekt eller None vid fel.
    """
    # Circuit breaker?
    block_until = st.session_state.get("_sheets_rates_block_until")
    if block_until and _now() < block_until:
        return None

    title = RATES_SHEET_NAME or "Valutakurser"
    try:
        ss = _get_spreadsheet()
    except Exception as e:
        st.warning(f"⚠️ Kunde inte öppna Google Sheet (SHEET_URL/behörighet): {e}")
        return None

    # 1) Exakt
    try:
        return with_backoff(ss.worksheet, title)
    except gspread.exceptions.APIError as e:
        _maybe_block_on_429(e)
        # fortsätt försöka nedan
    except Exception:
        pass

    # 2) Lös match
    try:
        ws = _find_ws_loose(ss, title)
        if ws:
            _ensure_header(ws)
            return ws
    except gspread.exceptions.APIError as e:
        _maybe_block_on_429(e)
    except Exception:
        pass

    # 3) Skapa bladet
    try:
        ws = with_backoff(ss.add_worksheet, title=title, rows=50, cols=5)
        _ensure_header(ws)
        return ws
    except gspread.exceptions.APIError as e:
        # Race: redan skapat? Försök öppna igen.
        try:
            ws = with_backoff(ss.worksheet, title)
            _ensure_header(ws)
            return ws
        except Exception:
            _maybe_block_on_429(e)
            st.warning(f"⚠️ Kunde inte öppna/skapa bladet '{title}': {e}")
            return None
    except Exception as e:
        st.warning(f"⚠️ Kunde inte öppna/skapa bladet '{title}': {e}")
        return None


def _maybe_block_on_429(err: Exception):
    """Sätt en kort blockering vid 429 för att undvika att elda på kvottaket."""
    try:
        msg = str(err)
        if "429" in msg or "Quota exceeded" in msg:
            st.session_state["_sheets_rates_block_until"] = _now() + timedelta(seconds=_BLOCK_SECONDS_ON_429)
    except Exception:
        pass


# ---------- Session-cache av kurser ----------
def _get_rates_from_session() -> Dict[str, float] | None:
    data = st.session_state.get("_rates_cache")
    ts = st.session_state.get("_rates_cache_ts")
    if not isinstance(data, dict) or not ts:
        return None
    try:
        if _now() - ts <= timedelta(minutes=_CACHE_TTL_MIN):
            return data
    except Exception:
        return None
    return None

def _set_rates_in_session(rates: Dict[str, float]):
    st.session_state["_rates_cache"] = dict(rates)
    st.session_state["_rates_cache_ts"] = _now()


# ---------- Publika API-funktioner ----------
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läser sparade kurser från session-cache (om färska), annars från Sheets (om möjligt).
    Faller tillbaka till STANDARD_VALUTAKURSER utan att krascha.
    """
    # 1) Session-cache
    cached = _get_rates_from_session()
    if cached:
        return dict(cached)

    # 2) Sheets
    ws = _get_rates_ws()
    if ws is None:
        st.warning("Använder standardkurser (valutablad ej tillgängligt).")
        out = {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}
        _set_rates_in_session(out)
        return out

    try:
        rows = with_backoff(ws.get_all_records)  # [{'Valuta':'USD','Kurs':'10.23'}, ...]
        out: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = str(r.get("Kurs", "")).replace(",", ".").strip()
            try:
                out[cur] = float(val)
            except Exception:
                pass
        # Fyll luckor
        for k, v in STANDARD_VALUTAKURSER.items():
            out.setdefault(k, float(v))

        _set_rates_in_session(out)
        return out
    except gspread.exceptions.APIError as e:
        _maybe_block_on_429(e)
        st.warning("Använder standardkurser (läsa från valutablad misslyckades p.g.a. kvottak).")
        out = {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}
        _set_rates_in_session(out)
        return out
    except Exception as e:
        st.warning(f"Använder standardkurser (fel vid läsning): {e}")
        out = {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}
        _set_rates_in_session(out)
        return out

def spara_valutakurser(rates: Dict[str, float]) -> None:
    """
    Skriver in rad-ordning: USD, NOK, CAD, EUR, SEK och uppdaterar session-cache.
    """
    ws = _get_rates_ws()
    if ws is None:
        st.warning("Kunde inte spara kurser – valutablad ej tillgängligt (kvottak/behörighet?).")
        return

    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])

    try:
        with_backoff(ws.clear)
        with_backoff(ws.update, body)
        _set_rates_in_session({k: float(v if isinstance(v, (int, float)) else STANDARD_VALUTAKURSER.get(k, 1.0))
                               for k, v in rates.items()})
        st.success("Valutakurser sparade.")
    except gspread.exceptions.APIError as e:
        _maybe_block_on_429(e)
        st.warning(f"Kunde inte spara valutakurser (kvottak?): {e}")
    except Exception as e:
        st.warning(f"Kunde inte spara valutakurser: {e}")

def hamta_valutakurser_auto() -> Tuple[Dict[str, float], List[str], str]:
    """
    Försöker hämta USD/EUR/CAD/NOK -> SEK.
    Ordning: 1) FMP (om API-nyckel finns) -> 2) Frankfurter -> 3) exchangerate.host
    Returnerar (rates, misses, provider)
    """
    misses: List[str] = []
    rates: Dict[str, float] = {}
    provider: str = "okänd"

    # 1) FMP
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    if fmp_key:
        try:
            base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")

            def _pair(pair: str):
                url = f"{base}/api/v3/fx/{pair}"
                r = requests.get(url, params={"apikey": fmp_key}, timeout=15)
                if r.status_code != 200:
                    return None, r.status_code
                j = r.json() or {}
                return (float(j.get("price")) if j.get("price") is not None else None, 200)

            provider = "FMP"
            for pair in ("USDSEK", "NOKSEK", "CADSEK", "EURSEK"):
                v, sc = _pair(pair)
                if v and v > 0:
                    base_ccy = pair[:3]
                    rates[base_ccy] = float(v)
                else:
                    misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            pass

    # 2) Frankfurter (ECB)
    if len(rates) < 4:
        provider = "Frankfurter"
        for base_ccy in ("USD", "EUR", "CAD", "NOK"):
            try:
                r2 = requests.get(
                    "https://api.frankfurter.app/latest",
                    params={"from": base_ccy, "to": "SEK"},
                    timeout=12,
                )
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
        for base_ccy in ("USD", "EUR", "CAD", "NOK"):
            try:
                r3 = requests.get(
                    "https://api.exchangerate.host/latest",
                    params={"base": base_ccy, "symbols": "SEK"},
                    timeout=12,
                )
                if r3.status_code == 200:
                    v = (r3.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # Fyll luckor med sparade/standard
    saved = _get_rates_from_session() or las_sparade_valutakurser()
    for base_ccy in ("USD", "EUR", "CAD", "NOK", "SEK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    # Uppdatera session-cache med det vi just hämtade
    merged = dict(saved)
    merged.update(rates)
    _set_rates_in_session(merged)

    return rates, misses, provider

def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    """
    Hämtar kurs från user_rates (annars STANDARD_VALUTAKURSER).
    """
    if not valuta:
        return 1.0
    v = str(valuta).upper()
    return float(user_rates.get(v, STANDARD_VALUTAKURSER.get(v, 1.0)))
