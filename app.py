# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import gspread
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials

# --- Lokalt Stockholm-time helpers ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_dt(): return datetime.now(TZ_STHLM)
    def now_stamp(): return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_dt(): return datetime.now()
    def now_stamp(): return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys & investeringsförslag", layout="wide")

# -------------------------
#    IMPORTERA MODULER
# -------------------------
from stockapp.utils import recompute_derived, add_oldest_ts_col
from stockapp.views import (
    kontrollvy,
    analysvy,
    lagg_till_eller_uppdatera,
    visa_investeringsforslag,
    visa_portfolj,
)

# -------------------------
#   GOOGLE SHEETS KONFIG
# -------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0.0, 0.7, 1.5, 3.0]
    last_err = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def _ws_main():
    return get_spreadsheet().worksheet(SHEET_NAME)

def _ws_rates():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta","Kurs"], ["USD","9.75"], ["EUR","11.18"], ["NOK","0.95"], ["CAD","7.05"], ["SEK","1"]])
        return ws

def hamta_data() -> pd.DataFrame:
    try:
        ws = _ws_main()
        rows = _with_backoff(ws.get_all_records)
        df = pd.DataFrame(rows)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    return df

def spara_data(df: pd.DataFrame, do_snapshot: bool = False):
    if do_snapshot:
        try:
            backup_snapshot_sheet(df)
        except Exception as e:
            st.warning(f"Kunde inte skapa snapshot: {e}")
    ws = _ws_main()
    _with_backoff(ws.clear)
    body = [df.columns.tolist()] + df.astype(str).values.tolist()
    _with_backoff(ws.update, body)

# Snapshot
def backup_snapshot_sheet(df: pd.DataFrame):
    ss = get_spreadsheet()
    snap = f"Snapshot-{now_dt().strftime('%Y%m%d-%H%M%S')}"
    ss.add_worksheet(title=snap, rows=max(1000, len(df)+10), cols=max(50, len(df.columns)+2))
    ws = ss.worksheet(snap)
    _with_backoff(ws.update, [df.columns.tolist()] + df.astype(str).values.tolist())
    st.sidebar.success(f"Snapshot skapad: {snap}")

# -------------------------
#   KOLUMN-SCHEMA / TS
# -------------------------
TS_FIELDS = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

FINAL_COLS = [
    # Grund
    "Ticker","Bolagsnamn","Sektor","Valuta",
    "Utestående aktier","Antal aktier","GAV SEK",
    "Aktuell kurs",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Årlig utdelning",
    # Kassa & kvalitet
    "Kassa (valuta)","FCF TTM (valuta)","Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Runway (mån)",
    # Härledda/övrigt
    "Market Cap (nu)","CAGR 5 år (%)",
    # Meta
    "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    # TS-kolumner
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for c in FINAL_COLS:
        if c not in df.columns:
            if c.startswith("TS_") or c in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor","Valuta","Bolagsnamn","Ticker"):
                df[c] = ""
            else:
                df[c] = 0.0
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier","Antal aktier","GAV SEK","Aktuell kurs",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Årlig utdelning","Kassa (valuta)","FCF TTM (valuta)","Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)",
        "Runway (mån)","Market Cap (nu)","CAGR 5 år (%)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]:
        if c in df.columns: df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"): df[c] = df[c].astype(str)
    return df

# -------------------------
#   VALUTAKURSER
# -------------------------
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def _load_saved_rates(nonce: int) -> Dict[str,float]:
    ws = _ws_rates()
    recs = _with_backoff(ws.get_all_records)
    out = {}
    for r in recs:
        k = str(r.get("Valuta","")).upper().strip()
        v = str(r.get("Kurs","")).replace(",", ".").strip()
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def las_sparade_valutakurser() -> Dict[str,float]:
    return _load_saved_rates(st.session_state.get("rates_reload_key", 0))

def spara_valutakurser(rates: Dict[str,float]):
    ws = _ws_rates()
    body = [["Valuta","Kurs"]]
    for k in ["USD","EUR","NOK","CAD","SEK"]:
        body.append([k, str(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurser_auto() -> Tuple[Dict[str,float], List[str], str]:
    misses, rates, provider = [], {}, None
    # Frankfurter
    provider = "Frankfurter"
    for base in ("USD","EUR","CAD","NOK"):
        try:
            r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to":"SEK"}, timeout=12)
            if r.status_code == 200:
                v = (r.json() or {}).get("rates",{}).get("SEK")
                if v: rates[base] = float(v)
            else:
                misses.append(f"{base}SEK (HTTP {r.status_code})")
        except Exception:
            misses.append(f"{base}SEK (fel)")
    # fallback exchangerate.host
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base in ("USD","EUR","CAD","NOK"):
            if base in rates: continue
            try:
                r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": "SEK"}, timeout=12)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates",{}).get("SEK")
                    if v: rates[base] = float(v)
            except Exception:
                pass
    saved = las_sparade_valutakurser()
    for base in ("USD","EUR","CAD","NOK"):
        if base not in rates:
            rates[base] = float(saved.get(base, STANDARD_VALUTAKURSER.get(base, 1.0)))
    return rates, misses, provider or "okänd"

def _sidebar_rates() -> Dict[str,float]:
    st.sidebar.header("💱 Valutakurser → SEK")

    # init default state keys BEFORE widgets:
    for k, dv in (("rate_usd", STANDARD_VALUTAKURSER["USD"]),
                  ("rate_eur", STANDARD_VALUTAKURSER["EUR"]),
                  ("rate_nok", STANDARD_VALUTAKURSER["NOK"]),
                  ("rate_cad", STANDARD_VALUTAKURSER["CAD"])):
        st.session_state.setdefault(k, float(las_sparade_valutakurser().get(k.split("_")[1].upper(), dv)))

    # Auto-hämta innan widgets skapas (så att vi får skriva state-keys säkert)
    if st.sidebar.button("🌐 Hämta kurser automatiskt", help="Frankfurter → exchangerate.host fallback"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Valutakurser hämtade ({provider}).")
        if misses:
            st.sidebar.warning("Vissa par saknades:\n- " + "\n- ".join(misses))
        st.session_state["rate_usd"] = float(auto_rates.get("USD", st.session_state["rate_usd"]))
        st.session_state["rate_eur"] = float(auto_rates.get("EUR", st.session_state["rate_eur"]))
        st.session_state["rate_nok"] = float(auto_rates.get("NOK", st.session_state["rate_nok"]))
        st.session_state["rate_cad"] = float(auto_rates.get("CAD", st.session_state["rate_cad"]))

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser({"USD": usd, "EUR": eur, "NOK": nok, "CAD": cad, "SEK": 1.0})
            st.session_state["rates_reload_key"] = st.session_state.get("rates_reload_key", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col2:
        if st.button("↻ Läs sparade"):
            sr = las_sparade_valutakurser()
            st.session_state["rate_usd"] = float(sr.get("USD", usd))
            st.session_state["rate_eur"] = float(sr.get("EUR", eur))
            st.session_state["rate_nok"] = float(sr.get("NOK", nok))
            st.session_state["rate_cad"] = float(sr.get("CAD", cad))
            st.sidebar.success("Lästa.")

    st.sidebar.markdown("---")
    return {"USD": float(usd), "EUR": float(eur), "NOK": float(nok), "CAD": float(cad), "SEK": 1.0}

# -------------------------
#   RUNNERS (enkla safe)
# -------------------------
def _apply_changes(df: pd.DataFrame, ticker: str, changes: Dict[str,object], source: str) -> Tuple[pd.DataFrame, List[str]]:
    if "Ticker" not in df.columns: return df, []
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    if not mask.any(): return df, []
    ridx = df.index[mask][0]
    changed_fields = []
    for k, v in changes.items():
        if k not in df.columns:
            continue
        old = df.at[ridx, k]
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[ridx, k] = v
            changed_fields.append(k)
        else:
            # även om samma värde: uppdatera TS om fältet är spårat (enligt din preferens)
            if k in TS_FIELDS:
                changed_fields.append(k)  # markera som ”uppdaterad”
    # TS
    for f in changed_fields:
        if f in TS_FIELDS:
            df.at[ridx, TS_FIELDS[f]] = now_stamp()
    # Auto-metadata
    df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source
    return df, changed_fields

def runner_price_only(df: pd.DataFrame, ticker: str, user_rates: Dict[str,float]):
    """Hämtar endast pris, namn & valuta via yfinance. Låg risk, snabb."""
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        price = info.get("regularMarketPrice", None)
        if price is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        name = info.get("shortName") or info.get("longName") or ""
        ccy = (info.get("currency") or "USD").upper()
        changes = {}
        if price and price > 0: changes["Aktuell kurs"] = float(price)
        if name: changes["Bolagsnamn"] = name
        if ccy:  changes["Valuta"] = ccy
        df2, changed = _apply_changes(df, ticker, changes, source="Yahoo (pris-only)")
        df2 = recompute_derived(df2)
        return df2, changed, "pris-only"
    except Exception as e:
        raise RuntimeError(f"pris-only fel: {e}")

def runner_full_auto(df: pd.DataFrame, ticker: str, user_rates: Dict[str,float]):
    """
    Enkel 'full' fallback:
      - pris, namn, valuta (yfinance)
      - försök P/S via info['priceToSalesTrailing12Months']
      - rör ej 'Omsättning idag'/'Omsättning nästa år' (manuella)
    """
    df2, changed, _ = runner_price_only(df, ticker, user_rates)
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        ps = info.get("priceToSalesTrailing12Months", None)
        ch = {}
        if ps and float(ps) > 0:
            ch["P/S"] = float(ps)
        if ch:
            df2, ch2 = _apply_changes(df2, ticker, ch, source="Yahoo (full-lite)")
            changed += ch2
        df2 = recompute_derived(df2)
        return df2, changed, "full-lite"
    except Exception as e:
        raise RuntimeError(f"full-auto fel: {e}")

# registrera default runners i sessionen
st.session_state.setdefault("_runner_price_only", runner_price_only)
st.session_state.setdefault("_runner", runner_full_auto)

# -------------------------
#   BATCH i SIDOPANELEN
# -------------------------
def _build_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    if sort_mode == "Äldst uppdaterade först (alla TS)":
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
        return work["Ticker"].astype(str).tolist()
    # default: A–Ö
    return df.sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in df.columns])["Ticker"].astype(str).tolist()

def _ensure_batch_state():
    st.session_state.setdefault("_batch_queue", [])    # lista med tickers
    st.session_state.setdefault("_batch_pos", 0)       # nästa index
    st.session_state.setdefault("_batch_log", {"changed":{}, "misses":{}, "errors":{}})

def run_batch_update(df: pd.DataFrame, user_rates: Dict[str,float], tickers: List[str], runner=None) -> Tuple[pd.DataFrame, dict]:
    if runner is None:
        runner = st.session_state.get("_runner") or runner_full_auto
    log = {"changed":{}, "misses":{}, "errors":{}}
    n = len(tickers)
    prog = st.sidebar.progress(0.0, text=f"Startar batch… 0/{n}")
    for i, tkr in enumerate(tickers, start=1):
        try:
            df, changed, status = runner(df, tkr, user_rates)
            if changed:
                log["changed"][tkr] = changed
            else:
                log["misses"][tkr] = ["(inga ändringar)"]
        except Exception as e:
            log["errors"][tkr] = [str(e)]
        prog.progress(i/max(n,1), text=f"Kör {i}/{n} • {tkr}")
    st.session_state["_batch_log"] = log
    return df, log

def _sidebar_batch_panel(df: pd.DataFrame, user_rates: Dict[str,float], save_cb):
    st.sidebar.subheader("🧰 Batch-körning")
    _ensure_batch_state()

    sort_mode = st.sidebar.selectbox("Urval", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla TS)"])
    antal = st.sidebar.number_input("Antal att lägga i kö", min_value=1, max_value=500, value=20, step=1)
    if st.sidebar.button("➕ Lägg i kö"):
        order = _build_order(df, sort_mode)
        # lägg till nästa 'antal' som inte redan finns i kön
        already = set(st.session_state["_batch_queue"])
        new_items = [t for t in order if t not in already][:int(antal)]
        st.session_state["_batch_queue"].extend(new_items)
        st.sidebar.success(f"La till {len(new_items)} tickers i kö.")
    # visa kö-status
    q = st.session_state["_batch_queue"]
    pos = st.session_state["_batch_pos"]
    st.sidebar.caption(f"Kö: {pos}/{len(q)} (nästa index → {pos+1 if pos < len(q) else '–'})")
    if q:
        st.sidebar.write(", ".join(q[max(0,pos-3):pos+7]) + (" …" if len(q) > pos+7 else ""))

    col_b1, col_b2, col_b3 = st.sidebar.columns(3)
    with col_b1:
        if st.button("▶️ Nästa"):
            if pos < len(q):
                df2, log = run_batch_update(df, user_rates, [q[pos]])
                st.session_state["_batch_pos"] += 1
                if save_cb:
                    try: save_cb(df2); st.sidebar.success("Sparat.")
                    except Exception as e: st.sidebar.warning(f"Kunde inte spara: {e}")
                st.session_state["_df_ref"] = df2
            else:
                st.sidebar.info("Kön är slut.")
    with col_b2:
        if st.button("⏩ Kör 5"):
            if pos < len(q):
                end = min(len(q), pos+5)
                df2, log = run_batch_update(df, user_rates, q[pos:end])
                st.session_state["_batch_pos"] = end
                if save_cb:
                    try: save_cb(df2); st.sidebar.success("Sparat.")
                    except Exception as e: st.sidebar.warning(f"Spara-fel: {e}")
                st.session_state["_df_ref"] = df2
            else:
                st.sidebar.info("Kön är slut.")
    with col_b3:
        if st.button("🏁 Kör alla"):
            if pos < len(q):
                df2, log = run_batch_update(df, user_rates, q[pos:])
                st.session_state["_batch_pos"] = len(q)
                if save_cb:
                    try: save_cb(df2); st.sidebar.success("Sparat.")
                    except Exception as e: st.sidebar.warning(f"Spara-fel: {e}")
                st.session_state["_df_ref"] = df2
            else:
                st.sidebar.info("Kön är slut.")

    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        if st.button("🗑️ Töm kö"):
            st.session_state["_batch_queue"] = []
            st.session_state["_batch_pos"] = 0
            st.sidebar.info("Kö rensad.")
    with col_c2:
        if st.button("↩️ Återställ pos"):
            st.session_state["_batch_pos"] = 0
            st.sidebar.info("Position återställd (0).")


# -------------------------
#           MAIN
# -------------------------
def main():
    # Rates först (behöver inga reruns)
    user_rates = _sidebar_rates()

    # Data in
    df = hamta_data()
    st.session_state["_df_ref"] = df  # arbetskopia för batchknappar som skriver tillbaka

    # Batch-panel i sidopanelen
    def _save(df_to_save): spara_data(df_to_save, do_snapshot=False)
    _sidebar_batch_panel(st.session_state["_df_ref"], user_rates, save_cb=_save)

    # Övergripande meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    if meny == "Kontroll":
        kontrollvy(st.session_state["_df_ref"])
    elif meny == "Analys":
        analysvy(st.session_state["_df_ref"], user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates, save_cb=_save)
        st.session_state["_df_ref"] = df2
    elif meny == "Investeringsförslag":
        # Här räknas härledda fält om
        df_calc = recompute_derived(st.session_state["_df_ref"])
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        df_calc = recompute_derived(st.session_state["_df_ref"])
        visa_portfolj(df_calc, user_rates)

if __name__ == "__main__":
    main()
