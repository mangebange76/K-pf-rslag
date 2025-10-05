# app.py (Del 1/5)
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# yfinance för snabb pris/namn etc
try:
    import yfinance as yf
except Exception:
    yf = None

# Egna moduler
from stockapp.sheets import (
    ws_read_df, ws_write_df, list_worksheet_titles, delete_worksheet
)
from stockapp.rates import (
    read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES
)
from stockapp.dividends import build_dividend_calendar

# (Valfritt) fetchers för enskild/full uppdatering
try:
    from stockapp.fetchers.yahoo import get_all as y_overview
except Exception:
    y_overview = None
try:
    from stockapp.fetchers.finviz import get_overview as fz_overview
except Exception:
    fz_overview = None
try:
    from stockapp.fetchers.morningstar import get_overview as ms_overview
except Exception:
    ms_overview = None
try:
    from stockapp.fetchers.sec import get_pb_quarters as sec_pb_quarters
except Exception:
    sec_pb_quarters = None

st.set_page_config(page_title="K-pf-rslag", layout="wide")

# ---------- tids-hjälpare ----------
def _now_sthlm() -> datetime:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz)
    except Exception:
        return datetime.now()

def now_stamp() -> str:
    return _now_sthlm().strftime("%Y-%m-%d")

# ---------- Snapshot-setup ----------
SNAP_PREFIX = "SNAP__"

def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

def _parse_snap_title(title: str) -> Optional[datetime]:
    m = re.search(r"(\d{8}_\d{6})$", str(title))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return None

def snapshot_on_start(df: pd.DataFrame, base_ws_title: str):
    """
    Skapar snapshot-blad 1 gång per appstart och rensar äldre än 5 dagar.
    Om kalkylarket är nära celltaket (10M) hoppar vi över snapshot-skapande.
    """
    if st.session_state.get("_snapshot_done"):
        return
    st.session_state["_snapshot_done"] = True

    # Skydd mot celltak: skapa bara snapshot om df är rimlig storlek
    approx_cells = max(1, len(df)) * max(1, len(df.columns))
    if approx_cells > 200_000:  # skydd – tunable
        st.sidebar.info("Hoppar över snapshot (för stort blad).")
        return

    now = _now_sthlm()
    snap_title = f"{SNAP_PREFIX}{base_ws_title}__{_format_ts(now)}"
    try:
        ws_write_df(snap_title, df)
        st.sidebar.success(f"Snapshot sparat: {snap_title}")
    except Exception as e:
        st.sidebar.warning(f"Kunde inte spara snapshot: {e}")

    # Rensa äldre än 5 dagar
    try:
        titles = list_worksheet_titles() or []
        cutoff = now - timedelta(days=5)
        for t in titles:
            if not str(t).startswith(SNAP_PREFIX):
                continue
            ts = _parse_snap_title(t)
            if ts and ts < cutoff.replace(tzinfo=None):
                try:
                    delete_worksheet(t)
                except Exception:
                    pass
    except Exception as e:
        st.sidebar.warning(f"Kunde inte rensa gamla snapshot-blad: {e}")

# ---------- Kolumnschema ----------
FINAL_COLS: List[str] = [
    # Bas
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "GAV (SEK)", "Aktuell kurs",
    "Utestående aktier",  # miljoner

    # Multiplar (P/S & P/B inkl kvartal och snitt)
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt (Q1..Q4)",
    "P/B", "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",

    # Omsättning (M)
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (bolagets valuta)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Utdelning
    "Årlig utdelning", "Payout (%)",

    # Övrigt
    "CAGR 5 år (%)",

    # Tidsstämplar
    "Senast manuellt uppdaterad",      # M
    "Senast auto uppdaterad",          # A (fetchers/kurs)
    "Auto källa",                      # text
    "Senast beräknad",                 # B (beräkningar)

    # Visning/score (sparas – ej den dynamiska 'Uppsida (%)')
    "DA (%)", "Uppsida idag (%)", "Uppsida 1 år (%)", "Uppsida 2 år (%)", "Uppsida 3 år (%)",
    "Score (Growth)", "Score (Dividend)", "Score (Financials)", "Score (Total)", "Confidence",

    # Sparade score per horisont
    "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",
    "Score Growth (Idag)", "Score Dividend (Idag)", "Score Financials (Idag)",
    "Score Growth (1 år)", "Score Dividend (1 år)", "Score Financials (1 år)",
    "Score Growth (2 år)", "Score Dividend (2 år)", "Score Financials (2 år)",
    "Score Growth (3 år)", "Score Dividend (3 år)", "Score Financials (3 år)",

    # Utdelningsschema
    "Div_Frekvens/år", "Div_Månader", "Div_Vikter",
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FINAL_COLS:
        if c not in out.columns:
            if any(k in c.lower() for k in [
                "kurs","omsättning","p/s","p/b","utdelning","cagr","aktier",
                "riktkurs","payout","score","uppsida","da","confidence",
                "frekvens","gav"
            ]):
                out[c] = 0.0
            else:
                out[c] = ""
    # typer
    float_cols = [
        "Antal aktier", "GAV (SEK)", "Aktuell kurs", "Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
        "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
        "DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
        "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",
        "Score Total (Idag)","Score Total (1 år)","Score Total (2 år)","Score Total (3 år)",
        "Score Growth (Idag)","Score Dividend (Idag)","Score Financials (Idag)",
        "Score Growth (1 år)","Score Dividend (1 år)","Score Financials (1 år)",
        "Score Growth (2 år)","Score Dividend (2 år)","Score Financials (2 år)",
        "Score Growth (3 år)","Score Dividend (3 år)","Score Financials (3 år)",
        "Div_Frekvens/år",
    ]
    for c in float_cols:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce").fillna(0.0)

    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Auto källa",
              "Senast manuellt uppdaterad","Senast auto uppdaterad","Senast beräknad",
              "Div_Månader","Div_Vikter"]:
        out[c] = out.get(c, "").astype(str)
    return out

# app.py (Del 2/5)

# ---------- Småhjälpare ----------
def _to_float(x) -> float:
    """Robust float: accepterar '1 234,56' och None."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace("\u00A0", "")
        # kommatecken → punkt
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return 0.0


def _drop_transient_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Skriv aldrig till Sheets kolumner som inte ingår i FINAL_COLS."""
    keep = [c for c in FINAL_COLS if c in df.columns]
    out = df.copy()
    # lägg till saknade FINAL_COLS (som tomma) så skrivningen inte havererar
    for c in FINAL_COLS:
        if c not in out.columns:
            out[c] = 0.0 if any(k in c.lower() for k in
                                ["kurs","omsättning","p/s","p/b","utdelning","cagr","aktier",
                                 "riktkurs","payout","score","uppsida","da","confidence",
                                 "frekvens","gav"]) else ""
    return out[FINAL_COLS].copy()


# ---------- Sidopanel: valutakurser ----------
def sidebar_rates() -> Dict[str, float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")

    # Initiera sessionstate en gång före widgets
    if "rates_loaded" not in st.session_state:
        try:
            saved = read_rates()
        except Exception:
            saved = {}
        st.session_state["rate_usd"] = float(saved.get("USD", DEFAULT_RATES["USD"]))
        st.session_state["rate_nok"] = float(saved.get("NOK", DEFAULT_RATES["NOK"]))
        st.session_state["rate_cad"] = float(saved.get("CAD", DEFAULT_RATES["CAD"]))
        st.session_state["rate_eur"] = float(saved.get("EUR", DEFAULT_RATES["EUR"]))
        st.session_state["rates_loaded"] = True

    colA, colB = st.sidebar.columns(2)
    if colA.button("🌐 Hämta livekurser"):
        try:
            live = fetch_live_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(live[k])
            st.sidebar.success("Livekurser hämtade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")

    if colB.button("↻ Läs sparade kurser"):
        try:
            saved = read_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
            st.sidebar.success("Sparade kurser inlästa.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte läsa sparade kurser: {e}")

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.000001, format="%.6f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.000001, format="%.6f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.000001, format="%.6f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.000001, format="%.6f")

    colC, colD = st.sidebar.columns(2)
    if colC.button("💾 Spara valutakurser"):
        try:
            save_rates({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.sidebar.success("Valutakurser sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    if colD.button("🛠 Reparera bladet"):
        try:
            repair_rates_sheet()
            st.sidebar.success("Bladet reparerat.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte reparera: {e}")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}


# ---------- IO (med cache-nonce) ----------
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str, _nonce: int) -> pd.DataFrame:
    # _nonce används endast för att bust:a cachen efter sparning
    return ws_read_df(ws_title)

def load_df(ws_title: str) -> pd.DataFrame:
    n = st.session_state.get("_reload_nonce", 0)
    df = load_df_cached(ws_title, n)
    return ensure_columns(df)

def save_df(ws_title: str, df: pd.DataFrame, bust_cache: bool = True):
    df_to_write = _drop_transient_cols(df)
    ws_write_df(ws_title, df_to_write)
    if bust_cache:
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1


# ---------- Snabb Yahoo (pris, namn, valuta, utd, CAGR) ----------
@st.cache_data(show_spinner=False, ttl=600)
def yahoo_fetch_one_quick(ticker: str) -> Dict[str, float | str]:
    out = {"Bolagsnamn":"", "Valuta":"USD", "Aktuell kurs":0.0, "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0}
    if yf is None or not ticker:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        nm = info.get("shortName") or info.get("longName")
        if nm: out["Bolagsnamn"] = str(nm)

        cur = info.get("currency")
        if cur: out["Valuta"] = str(cur).upper()

        px  = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h.columns:
                px = float(h["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)

        dr = info.get("dividendRate")
        if dr is not None:
            out["Årlig utdelning"] = float(dr)

        # grov CAGR på Total Revenue (årsdata)
        try:
            df_is = getattr(t, "income_stmt", None)
            ser = None
            if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
                ser = df_is.loc["Total Revenue"].dropna()
            else:
                df_fin = getattr(t, "financials", None)
                if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                    ser = df_fin.loc["Total Revenue"].dropna()
            if ser is not None and len(ser) >= 2:
                ser = ser.sort_index()
                start, end = float(ser.iloc[0]), float(ser.iloc[-1])
                years = max(1, len(ser)-1)
                if start > 0:
                    out["CAGR 5 år (%)"] = round(((end/start)**(1.0/years) - 1.0) * 100.0, 2)
        except Exception:
            pass
    except Exception:
        pass
    return out


# ---------- Full uppdatering för EN ticker (alla fetchers) ----------
def _update_one_all_fetchers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df

    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        return df

    # 1) Yahoo bred (om tillgänglig)
    if y_overview:
        try:
            y = y_overview(tkr) or {}
            mapping = {
                "name": "Bolagsnamn",
                "currency": "Valuta",
                "price": "Aktuell kurs",
                "dividend_rate": "Årlig utdelning",
                "ps_ttm": "P/S",
                "pb": "P/B",
                "shares_outstanding": "Utestående aktier",  # absolut → M
                "cagr5_pct": "CAGR 5 år (%)",
            }
            for k_src, k_dst in mapping.items():
                v = y.get(k_src)
                if v is None:
                    continue
                if k_dst == "Utestående aktier":
                    df.loc[mask, k_dst] = _to_float(v) / 1e6
                else:
                    if isinstance(v, (int,float,np.floating)):
                        df.loc[mask, k_dst] = float(v)
                    else:
                        df.loc[mask, k_dst] = str(v)
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Yahoo"
        except Exception:
            pass

    # 2) Finviz
    if fz_overview:
        try:
            fz = fz_overview(tkr) or {}
            if _to_float(fz.get("ps_ttm", 0.0)) > 0:
                df.loc[mask, "P/S"] = _to_float(fz["ps_ttm"])
            if _to_float(fz.get("pb", 0.0)) > 0:
                df.loc[mask, "P/B"] = _to_float(fz["pb"])
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Finviz"
        except Exception:
            pass

    # 3) Morningstar
    if ms_overview:
        try:
            ms = ms_overview(tkr) or {}
            if _to_float(ms.get("ps_ttm", 0.0)) > 0:
                df.loc[mask, "P/S"] = _to_float(ms["ps_ttm"])
            if _to_float(ms.get("pb", 0.0)) > 0:
                df.loc[mask, "P/B"] = _to_float(ms["pb"])
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Morningstar"
        except Exception:
            pass

    # 4) SEC → P/B kvartal
    if sec_pb_quarters:
        try:
            sec = sec_pb_quarters(tkr) or {}
            pairs = sec.get("pb_quarters") or []  # list[(date, pb)]
            if pairs:
                pairs = pairs[:4]  # Q1..Q4
                for idx, (_, pbv) in enumerate(pairs, start=1):
                    if idx > 4:
                        break
                    df.loc[mask, f"P/B Q{idx}"] = _to_float(pbv)
            # P/B-snitt uppdateras senare i beräkningarna
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "SEC"
        except Exception:
            pass

    return df


# ---------- Massuppdateringar (sidopanel) ----------
def sidebar_mass_update_controls(df: pd.DataFrame, ws_title: str) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 Uppdatera data")

    # Snabb (Yahoo) – pris/namn/valuta/div/CAGR
    if st.sidebar.button("🔄 Snabbuppdatera alla (Yahoo)"):
        bar = st.sidebar.progress(0)
        status = st.sidebar.empty()
        n = len(df)
        fails = []
        for i, row in df.iterrows():
            tkr = str(row.get("Ticker","")).strip().upper()
            if not tkr:
                bar.progress((i+1)/n)
                continue
            status.write(f"Yahoo snabb: {tkr} ({i+1}/{n})")
            try:
                q = yahoo_fetch_one_quick(tkr)
                if q.get("Bolagsnamn"):   df.at[i, "Bolagsnamn"] = str(q["Bolagsnamn"])
                if q.get("Valuta"):       df.at[i, "Valuta"]     = str(q["Valuta"])
                if _to_float(q.get("Aktuell kurs",0.0))>0: df.at[i, "Aktuell kurs"] = _to_float(q["Aktuell kurs"])
                df.at[i, "Årlig utdelning"]   = _to_float(q.get("Årlig utdelning",0.0))
                df.at[i, "CAGR 5 år (%)"]     = _to_float(q.get("CAGR 5 år (%)",0.0))
                df.at[i, "Senast auto uppdaterad"] = now_stamp()
                df.at[i, "Auto källa"] = "Yahoo (snabb)"
            except Exception as e:
                fails.append(f"{tkr}: {e}")
            time.sleep(0.6)  # artigt
            bar.progress((i+1)/n)

        if fails:
            st.sidebar.warning("Vissa misslyckades – detaljer:")
            st.sidebar.text_area("Fel", "\n".join(fails), height=160)
        # spara efter snabb uppdatering (beräkning görs i main före visning)
        try:
            save_df(ws_title, df, bust_cache=True)
            st.sidebar.success("Snabbuppdatering sparad.")
            st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

    # Full (alla fetchers)
    if st.sidebar.button("🧰 Full uppdatering (alla fetchers)"):
        bar = st.sidebar.progress(0)
        status = st.sidebar.empty()
        n = len(df)
        fails = []
        for i, row in df.iterrows():
            tkr = str(row.get("Ticker","")).strip().upper()
            if not tkr:
                bar.progress((i+1)/n)
                continue
            status.write(f"Full fetchers: {tkr} ({i+1}/{n})")
            try:
                df = _update_one_all_fetchers(df, tkr)
            except Exception as e:
                fails.append(f"{tkr}: {e}")
            time.sleep(0.6)  # artigt
            bar.progress((i+1)/n)

        if fails:
            st.sidebar.warning("Vissa misslyckades – detaljer:")
            st.sidebar.text_area("Fel", "\n".join(fails), height=160)

        try:
            save_df(ws_title, df, bust_cache=True)
            st.sidebar.success("Full uppdatering sparad.")
            st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

    return df

# app.py (Del 3/5)

# ---------- Beräkningar ----------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_ps_pb_snitt(row: pd.Series) -> Tuple[float, float]:
    ps_vals = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
    ps_clean = [_to_float(x) for x in ps_vals if _to_float(x) > 0]
    ps_avg = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

    pb_vals = [row.get("P/B Q1", 0), row.get("P/B Q2", 0), row.get("P/B Q3", 0), row.get("P/B Q4", 0)]
    pb_clean = [_to_float(x) for x in pb_vals if _to_float(x) > 0]
    pb_avg = round(float(np.mean(pb_clean)), 2) if pb_clean else 0.0
    return ps_avg, pb_avg

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Säkerställ nödvändiga kolumner som kan saknas efter inläsning
    for c in ["DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
              "P/S-snitt (Q1..Q4)","P/B-snitt (Q1..Q4)","Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
        if c not in out.columns:
            out[c] = 0.0

    for i, r in out.iterrows():
        # P/S- & P/B-snitt
        ps_avg, pb_avg = compute_ps_pb_snitt(r)
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i, "P/B-snitt (Q1..Q4)"] = pb_avg

        # CAGR clamp 2..50
        cagr = _to_float(r.get("CAGR 5 år (%)", 0.0))
        g = clamp(cagr, 2.0, 50.0) / 100.0

        # Omsättning 2–3 år från ”Omsättning nästa år”
        next_rev = _to_float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev. manuellt ifyllda
            out.at[i, "Omsättning om 2 år"] = _to_float(r.get("Omsättning om 2 år", 0.0))
            out.at[i, "Omsättning om 3 år"] = _to_float(r.get("Omsättning om 3 år", 0.0))

        # Riktkurser via P/S-snitt
        shares_m = _to_float(r.get("Utestående aktier", 0.0))
        if shares_m <= 0 or ps_avg <= 0:
            out.at[i, "Riktkurs idag"] = 0.0
            out.at[i, "Riktkurs om 1 år"] = 0.0
            out.at[i, "Riktkurs om 2 år"] = 0.0
            out.at[i, "Riktkurs om 3 år"] = 0.0
        else:
            out.at[i, "Riktkurs idag"]    = round(_to_float(r.get("Omsättning idag", 0.0))     * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 1 år"] = round(_to_float(r.get("Omsättning nästa år", 0.0)) * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 2 år"] = round(_to_float(out.at[i, "Omsättning om 2 år"])   * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 3 år"] = round(_to_float(out.at[i, "Omsättning om 3 år"])   * ps_avg / shares_m, 2)

        # Uppsidor & DA
        price = _to_float(out.at[i, "Aktuell kurs"])
        for col_src, col_dst in [
            ("Riktkurs idag", "Uppsida idag (%)"),
            ("Riktkurs om 1 år", "Uppsida 1 år (%)"),
            ("Riktkurs om 2 år", "Uppsida 2 år (%)"),
            ("Riktkurs om 3 år", "Uppsida 3 år (%)"),
        ]:
            rk = _to_float(out.at[i, col_src])
            out.at[i, col_dst] = ((rk - price) / price * 100.0) if price > 0 else 0.0

        out.at[i, "DA (%)"] = ((_to_float(r.get("Årlig utdelning", 0.0)) / price) * 100.0) if price > 0 else 0.0

    return out


def _horizon_to_tag(h: str) -> str:
    if "om 1 år" in h: return "1 år"
    if "om 2 år" in h: return "2 år"
    if "om 3 år" in h: return "3 år"
    return "Idag"


def score_rows(df: pd.DataFrame, horizon: str, strategy: str) -> pd.DataFrame:
    out = df.copy()

    # Bered hjälpkolumner
    price = out["Aktuell kurs"].astype(float).replace(0, np.nan)
    rk = out[horizon].astype(float).replace(0, np.nan)

    out["Uppsida (%)"] = ((rk - price) / price * 100.0).fillna(0.0)
    out["DA (%)"] = np.where(out["Aktuell kurs"] > 0,
                             (out["Årlig utdelning"].map(_to_float) / out["Aktuell kurs"].map(_to_float)) * 100.0,
                             0.0)

    # Growth-score
    cur_ps = out["P/S"].astype(float).replace(0, np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].astype(float).replace(0, np.nan)
    cheap_ps = (ps_avg / (cur_ps * 2.0)).clip(upper=1.0).fillna(0.0)

    g_norm = (out["CAGR 5 år (%)"].map(_to_float) / 30.0).clip(0, 1)
    u_norm = (out["Uppsida (%)"] / 50.0).clip(0, 1)
    out["Score (Growth)"] = (0.4 * g_norm + 0.4 * u_norm + 0.2 * cheap_ps) * 100.0

    # Dividend-score
    payout = out["Payout (%)"].map(_to_float)
    payout_health = 1 - (abs(payout - 60.0) / 60.0)
    payout_health = payout_health.clip(0, 1)
    payout_health = np.where(payout <= 0, 0.85, payout_health)
    y_norm = (out["DA (%)"] / 8.0).clip(0, 1)
    grow_ok = np.where(out["CAGR 5 år (%)"].map(_to_float) >= 0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6 * y_norm + 0.3 * payout_health + 0.1 * grow_ok) * 100.0

    # Financials-score
    cur_pb = out["P/B"].astype(float).replace(0, np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].astype(float).replace(0, np.nan)
    cheap_pb = (pb_avg / (cur_pb * 2.0)).clip(upper=1.0).fillna(0.0)
    out["Score (Financials)"] = (0.7 * cheap_pb + 0.3 * u_norm) * 100.0

    # Viktning per strategi/sektor
    def weights_for_row(sektor: str, strategy: str) -> Tuple[float, float, float]:
        if strategy == "Tillväxt":   return (0.70, 0.10, 0.20)
        if strategy == "Utdelning":  return (0.15, 0.70, 0.15)
        if strategy == "Finans":     return (0.20, 0.20, 0.60)
        s = (sektor or "").lower()
        if any(k in s for k in ["bank","finans","insurance","financial"]): return (0.25, 0.25, 0.50)
        if any(k in s for k in ["utility","utilities","consumer staples","telecom"]): return (0.20, 0.60, 0.20)
        if any(k in s for k in ["tech","information technology","semiconductor","software"]): return (0.70, 0.10, 0.20)
        return (0.45, 0.35, 0.20)

    strat = "Auto" if str(strategy).startswith("Auto") else strategy
    Wg, Wd, Wf = [], [], []
    for _, r in out.iterrows():
        wg, wd, wf = weights_for_row(r.get("Sektor",""), strat)
        Wg.append(wg); Wd.append(wd); Wf.append(wf)
    Wg = np.array(Wg); Wd = np.array(Wd); Wf = np.array(Wf)

    out["Score (Total)"] = (Wg*out["Score (Growth)"] + Wd*out["Score (Dividend)"] + Wf*out["Score (Financials)"]).round(2)

    # Confidence: nyckelfält på plats
    need = [
        out["Aktuell kurs"].astype(float) > 0,
        out["P/S-snitt (Q1..Q4)"].astype(float) > 0,
        out["Omsättning idag"].astype(float) >= 0,
        out["Omsättning nästa år"].astype(float) >= 0,
    ]
    present = np.stack(need, axis=0).astype(float)
    out["Confidence"] = (present.mean(axis=0) * 100.0).round(0)
    return out


def compute_scores_all_horizons(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    mapping = [("Riktkurs idag","Idag"), ("Riktkurs om 1 år","1 år"),
               ("Riktkurs om 2 år","2 år"), ("Riktkurs om 3 år","3 år")]
    strat = "Auto" if str(strategy).startswith("Auto") else strategy
    for horizon, tag in mapping:
        tmp = score_rows(out, horizon=horizon, strategy=strat)
        out[f"Score Growth ({tag})"]     = tmp["Score (Growth)"].round(2)
        out[f"Score Dividend ({tag})"]   = tmp["Score (Dividend)"].round(2)
        out[f"Score Financials ({tag})"] = tmp["Score (Financials)"].round(2)
        out[f"Score Total ({tag})"]      = tmp["Score (Total)"].round(2)
    return out


def enrich_for_save(df: pd.DataFrame, horizon_for_score: str = "Riktkurs idag", strategy: str = "Auto") -> pd.DataFrame:
    df2 = update_calculations(df)
    df2 = score_rows(df2, horizon=horizon_for_score, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2 = compute_scores_all_horizons(df2, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2["Senast beräknad"] = now_stamp()
    return df2


# ---------- Etiketttext bredvid fält ----------
def _m_tag(df_row: pd.Series) -> str:
    d = str(df_row.get("Senast manuellt uppdaterad","")).strip()
    return f"〔M: {d or '—'}〕"

def _a_tag(df_row: pd.Series) -> str:
    d = str(df_row.get("Senast auto uppdaterad","")).strip()
    src = str(df_row.get("Auto källa","")).strip()
    return f"〔A: {d or '—'}{(' · '+src) if src else ''}〕"

def _b_tag(df_row: pd.Series) -> str:
    d = str(df_row.get("Senast beräknad","")).strip()
    return f"〔B: {d or '—'}〕"


# ---------- Äldst-tabeller (topp10) ----------
def _oldest_tables(df: pd.DataFrame):
    st.markdown("### ⏱️ Äldst uppdaterade")
    def _to_date(s: str) -> Optional[pd.Timestamp]:
        s = (s or "").strip()
        if not s: return None
        try: return pd.to_datetime(s)
        except Exception: return None

    tmp = df.copy()
    tmp["d_man"]  = tmp["Senast manuellt uppdaterad"].apply(_to_date)
    tmp["d_auto"] = tmp["Senast auto uppdaterad"].apply(_to_date)
    tmp["d_any"]  = tmp[["d_man","d_auto"]].min(axis=1)

    any_sorted = tmp.dropna(subset=["d_any"]).sort_values("d_any", ascending=True)
    if any_sorted.empty:
        st.info("Inga tidsstämplar ännu.")
    else:
        st.dataframe(any_sorted.head(10)[["Ticker","Bolagsnamn","d_any"]]
                     .rename(columns={"d_any":"Äldst (valfri)"}), use_container_width=True)

    man_sorted = tmp.dropna(subset=["d_man"]).sort_values("d_man", ascending=True)
    if not man_sorted.empty:
        st.dataframe(man_sorted.head(10)[["Ticker","Bolagsnamn","d_man"]]
                     .rename(columns={"d_man":"Äldst (manuell)"}), use_container_width=True)

    auto_sorted = tmp.dropna(subset=["d_auto"]).sort_values("d_auto", ascending=True)
    if not auto_sorted.empty:
        st.dataframe(auto_sorted.head(10)[["Ticker","Bolagsnamn","d_auto","Auto källa"]]
                     .rename(columns={"d_auto":"Äldst (auto)"}), use_container_width=True)

    st.markdown("---")


# ---------- Manuell insamling ----------
def view_manual(df: pd.DataFrame, ws_title: str):
    st.subheader("🧩 Manuell insamling")

    # Överst: äldst-tabeller
    _oldest_tables(df)

    # Navigering & val
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn","")).strip() else str(r["Ticker"]) for _, r in vis.iterrows()]
    labels = ["➕ Lägg till nytt bolag..."] + labels

    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0

    sel = st.selectbox("Välj bolag att redigera", list(range(len(labels))),
                       format_func=lambda i: labels[i], index=st.session_state["manual_idx"])
    st.session_state["manual_idx"] = sel

    c_prev, c_next = st.columns([1,1])
    with c_prev:
        if st.button("⬅️ Föregående", use_container_width=True, disabled=(sel<=0)):
            st.session_state["manual_idx"] = max(0, sel-1); st.rerun()
    with c_next:
        if st.button("➡️ Nästa", use_container_width=True, disabled=(sel>=len(labels)-1)):
            st.session_state["manual_idx"] = min(len(labels)-1, sel+1); st.rerun()

    is_new = (sel == 0)
    if not is_new:
        row = vis.iloc[sel-1]
    else:
        row = pd.Series({c: (0.0 if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else "") for c in df.columns})

    # Enskild full uppdatering
    col_up1, col_up2 = st.columns([1,1])
    with col_up1:
        if not is_new and st.button("🔭 Full uppdatering (alla fetchers) för vald ticker"):
            df = _update_one_all_fetchers(df, row["Ticker"])
            df = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            try:
                save_df(ws_title, df, bust_cache=True)
                st.success("Fetchers körda, beräkningar uppdaterade och sparade.")
                st.rerun()
            except Exception as e:
                st.error(f"Kunde inte spara: {e}")
    with col_up2:
        st.caption("Kör Yahoo/Finviz/Morningstar/SEC (om tillgängliga) för just den här tickern.")

    # Obligatoriska fält
    st.markdown("### Obligatoriska fält")
    mtag = _m_tag(row)

    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input(f"Ticker (Yahoo-format) {mtag}",
                               value=str(row.get("Ticker","")).upper() if not is_new else "",
                               placeholder="t.ex. AAPL")
        antal  = st.number_input(f"Antal aktier (du äger) {mtag}",
                                 value=_to_float(row.get("Antal aktier",0.0) or 0.0), step=1.0, min_value=0.0)
        gav    = st.number_input(f"GAV (SEK) {mtag}",
                                 value=_to_float(row.get("GAV (SEK)",0.0) or 0.0), step=0.01, min_value=0.0, format="%.2f")
    with c2:
        oms_idag = st.number_input(f"Omsättning idag (M) {mtag}",
                                   value=_to_float(row.get("Omsättning idag",0.0) or 0.0), step=1.0, min_value=0.0)
        oms_nxt  = st.number_input(f"Omsättning nästa år (M) {mtag}",
                                   value=_to_float(row.get("Omsättning nästa år",0.0) or 0.0), step=1.0, min_value=0.0)

    # Övriga fält: FETCHERS
    atag = _a_tag(row)
    with st.expander("🌐 Fält som hämtas (auto)"):
        cL, cR = st.columns(2)
        with cL:
            bolagsnamn = st.text_input(f"Bolagsnamn {atag}", value=str(row.get("Bolagsnamn","")))
            sektor     = st.text_input(f"Sektor {atag}", value=str(row.get("Sektor","")))
            valuta     = st.text_input(f"Valuta (t.ex. USD, SEK) {atag}", value=str(row.get("Valuta","") or "USD").upper())
            aktuell_kurs = st.number_input(f"Aktuell kurs {atag}", value=_to_float(row.get("Aktuell kurs",0.0) or 0.0), step=0.01, min_value=0.0)
            utd_arlig    = st.number_input(f"Årlig utdelning {atag}", value=_to_float(row.get("Årlig utdelning",0.0) or 0.0), step=0.01, min_value=0.0)
            payout_pct   = st.number_input(f"Payout (%) {atag}", value=_to_float(row.get("Payout (%)",0.0) or 0.0), step=1.0, min_value=0.0)
        with cR:
            utest_m = st.number_input(f"Utestående aktier (miljoner) {atag}", value=_to_float(row.get("Utestående aktier",0.0) or 0.0), step=1.0, min_value=0.0)
            ps  = st.number_input(f"P/S {atag}",   value=_to_float(row.get("P/S",0.0) or 0.0), step=0.01, min_value=0.0)
            ps1 = st.number_input(f"P/S Q1 {atag}", value=_to_float(row.get("P/S Q1",0.0) or 0.0), step=0.01, min_value=0.0)
            ps2 = st.number_input(f"P/S Q2 {atag}", value=_to_float(row.get("P/S Q2",0.0) or 0.0), step=0.01, min_value=0.0)
            ps3 = st.number_input(f"P/S Q3 {atag}", value=_to_float(row.get("P/S Q3",0.0) or 0.0), step=0.01, min_value=0.0)
            ps4 = st.number_input(f"P/S Q4 {atag}", value=_to_float(row.get("P/S Q4",0.0) or 0.0), step=0.01, min_value=0.0)
            pb  = st.number_input(f"P/B {atag}",   value=_to_float(row.get("P/B",0.0) or 0.0), step=0.01, min_value=0.0)
            pb1 = st.number_input(f"P/B Q1 {atag}", value=_to_float(row.get("P/B Q1",0.0) or 0.0), step=0.01, min_value=0.0)
            pb2 = st.number_input(f"P/B Q2 {atag}", value=_to_float(row.get("P/B Q2",0.0) or 0.0), step=0.01, min_value=0.0)
            pb3 = st.number_input(f"P/B Q3 {atag}", value=_to_float(row.get("P/B Q3",0.0) or 0.0), step=0.01, min_value=0.0)
            pb4 = st.number_input(f"P/B Q4 {atag}", value=_to_float(row.get("P/B Q4",0.0) or 0.0), step=0.01, min_value=0.0)

    # Övriga fält: BERÄKNADE (read-only)
    btag = _b_tag(row)
    with st.expander("🧮 Beräknade fält (auto)"):
        cA, cB = st.columns(2)
        with cA:
            st.number_input(f"P/S-snitt (Q1..Q4) {btag}", value=_to_float(row.get("P/S-snitt (Q1..Q4)",0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Omsättning om 2 år (M) {btag}", value=_to_float(row.get("Omsättning om 2 år",0.0) or 0.0), step=1.0, disabled=True)
            st.number_input(f"Riktkurs idag {btag}", value=_to_float(row.get("Riktkurs idag",0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Riktkurs om 2 år {btag}", value=_to_float(row.get("Riktkurs om 2 år",0.0) or 0.0), step=0.01, disabled=True)
        with cB:
            st.number_input(f"P/B-snitt (Q1..Q4) {btag}", value=_to_float(row.get("P/B-snitt (Q1..Q4)",0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Omsättning om 3 år (M) {btag}", value=_to_float(row.get("Omsättning om 3 år",0.0) or 0.0), step=1.0, disabled=True)
            st.number_input(f"Riktkurs om 1 år {btag}", value=_to_float(row.get("Riktkurs om 1 år",0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Riktkurs om 3 år {btag}", value=_to_float(row.get("Riktkurs om 3 år",0.0) or 0.0), step=0.01, disabled=True)

    # Spara
    def _any_core_change(before: pd.Series, after: dict) -> bool:
        core = ["Antal aktier","GAV (SEK)","Omsättning idag","Omsättning nästa år"]
        for k in core:
            b = _to_float(before.get(k,0.0) or 0.0); a = _to_float(after.get(k,0.0) or 0.0)
            if abs(a-b) > 1e-12: return True
        return False

    if st.button("💾 Spara"):
        errors = []
        if not ticker.strip(): errors.append("Ticker saknas.")
        if antal < 0: errors.append("Antal aktier kan inte vara negativt.")
        if gav < 0: errors.append("GAV (SEK) kan inte vara negativt.")
        if oms_idag < 0 or oms_nxt < 0: errors.append("Omsättning idag/nästa år kan inte vara negativt.")
        if errors:
            st.error(" | ".join(errors)); return

        exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())

        update = {
            # manuella (kärn)
            "Ticker": ticker.upper(),
            "Antal aktier": _to_float(antal),
            "GAV (SEK)": _to_float(gav),
            "Omsättning idag": _to_float(oms_idag),
            "Omsättning nästa år": _to_float(oms_nxt),
            # auto-fält (tillåt korr)
            "Bolagsnamn": str(bolagsnamn or "").strip(),
            "Sektor": str(sektor or "").strip(),
            "Valuta": str(valuta or "").strip().upper(),
            "Aktuell kurs": _to_float(aktuell_kurs),
            "Årlig utdelning": _to_float(utd_arlig),
            "Payout (%)": _to_float(payout_pct),
            "Utestående aktier": _to_float(utest_m),
            "P/S": _to_float(ps), "P/S Q1": _to_float(ps1), "P/S Q2": _to_float(ps2), "P/S Q3": _to_float(ps3), "P/S Q4": _to_float(ps4),
            "P/B": _to_float(pb), "P/B Q1": _to_float(pb1), "P/B Q2": _to_float(pb2), "P/B Q3": _to_float(pb3), "P/B Q4": _to_float(pb4),
        }

        def _apply_update(df_in: pd.DataFrame, mask, data: dict) -> pd.DataFrame:
            out = df_in.copy()
            for k,v in data.items():
                if k not in out.columns: 
                    continue
                if isinstance(v, str):
                    out.loc[mask, k] = v
                else:
                    out.loc[mask, k] = v
            return out

        if exists_mask.any():
            before_row = df.loc[exists_mask].iloc[0].copy()
            df = _apply_update(df, exists_mask, update)
            if _any_core_change(before_row, update):
                df.loc[exists_mask, "Senast manuellt uppdaterad"] = now_stamp()
        else:
            base = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Sektor","Valuta",
                                         "Senast manuellt uppdaterad","Senast auto uppdaterad","Auto källa",
                                         "Senast beräknad","Div_Månader","Div_Vikter"] else "")
                    for c in FINAL_COLS}
            base.update(update)
            base["Senast manuellt uppdaterad"] = now_stamp()
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
            exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())

        # Direkt efter spara → snabb Yahoo + auto-stämpel
        try:
            quick = yahoo_fetch_one_quick(ticker.upper())
            if quick.get("Bolagsnamn"): df.loc[exists_mask, "Bolagsnamn"] = str(quick["Bolagsnamn"])
            if quick.get("Valuta"):     df.loc[exists_mask, "Valuta"]     = str(quick["Valuta"])
            if _to_float(quick.get("Aktuell kurs",0.0))>0: df.loc[exists_mask, "Aktuell kurs"] = _to_float(quick["Aktuell kurs"])
            df.loc[exists_mask, "Årlig utdelning"] = _to_float(quick.get("Årlig utdelning",0.0))
            df.loc[exists_mask, "CAGR 5 år (%)"]   = _to_float(quick.get("CAGR 5 år (%)",0.0))
            df.loc[exists_mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[exists_mask, "Auto källa"] = "Yahoo (snabb)"
        except Exception:
            pass

        # Beräkna & spara
        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            st.success("Sparat, snabbdata hämtad, beräkningar uppdaterade.")
            st.rerun()
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

# app.py (Del 4/5)

# ---------- Datavy + massuppdatering ----------

def _iter_tickers(df: pd.DataFrame) -> List[str]:
    return [str(t).strip().upper() for t in df["Ticker"].astype(str).tolist() if str(t).strip()]

def _quick_update_one(df: pd.DataFrame, idx: int, ticker: str) -> None:
    """Uppdatera snabbdata (Yahoo) för en rad - in-place."""
    try:
        q = yahoo_fetch_one_quick(ticker)
        if q.get("Bolagsnamn"):
            df.at[idx, "Bolagsnamn"] = str(q["Bolagsnamn"])
        if q.get("Valuta"):
            df.at[idx, "Valuta"] = str(q["Valuta"]).upper()
        px = _to_float(q.get("Aktuell kurs", 0.0))
        if px > 0:
            df.at[idx, "Aktuell kurs"] = px
        df.at[idx, "Årlig utdelning"] = _to_float(q.get("Årlig utdelning", 0.0))
        df.at[idx, "CAGR 5 år (%)"] = _to_float(q.get("CAGR 5 år (%)", 0.0))
        df.at[idx, "Senast auto uppdaterad"] = now_stamp()
        if not str(df.at[idx, "Auto källa"]).strip():
            df.at[idx, "Auto källa"] = "Yahoo (snabb)"
    except Exception:
        pass

def _full_update_one(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Kör alla fetchers för en ticker (returnerar nytt df)."""
    return _update_one_all_fetchers(df, ticker)

def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data (hela bladet)")

    # Massuppdatering i sidopanel
    st.sidebar.markdown("### 🔁 Massuppdatering (hela bladet)")
    delay_quick = st.sidebar.slider("Delay mellan snabbhämtningar (sek)", min_value=0.3, max_value=2.0, value=0.6, step=0.1)
    delay_full  = st.sidebar.slider("Delay mellan fulla hämtningar (sek)", min_value=0.5, max_value=3.0, value=0.9, step=0.1)

    if st.sidebar.button("⚡ Snabbuppdatera kurser (Yahoo)"):
        try:
            tks = _iter_tickers(df)
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            n = max(1, len(tks))
            for i, tkr in enumerate(tks):
                status.write(f"Snabbuppdaterar {i+1}/{n}: {tkr}")
                # hitta index för raden
                idxs = df.index[df["Ticker"].astype(str).str.upper() == tkr].tolist()
                if idxs:
                    _quick_update_one(df, idxs[0], tkr)
                time.sleep(float(delay_quick))
                prog.progress((i+1)/n)
            # efter snabb uppd – beräkna & spara
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Snabbuppdatering klar & sparad.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Kunde inte snabbuppdatera: {e}")

    if st.sidebar.button("🔭 Full uppdatering (alla fetchers)"):
        try:
            tks = _iter_tickers(df)
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            n = max(1, len(tks))
            for i, tkr in enumerate(tks):
                status.write(f"Full uppdatering {i+1}/{n}: {tkr}")
                df = _full_update_one(df, tkr)
                time.sleep(float(delay_full))
                prog.progress((i+1)/n)
            # efter full uppd – beräkna & spara
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Full uppdatering klar & sparad.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Kunde inte full-uppdatera: {e}")

    # Visa datatabell
    st.dataframe(df, use_container_width=True)

    st.markdown("**Spara alla beräkningar till Google Sheets**")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox(
        "Score-horisont vid sparning",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0
    )
    strategy = c2.selectbox(
        "Strategi för score vid sparning",
        ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0
    )

    if st.button("💾 Spara beräkningar → Google Sheets"):
        try:
            df2 = enrich_for_save(df, horizon_for_score=horizon, strategy=("Auto" if strategy.startswith("Auto") else strategy))
            save_df(ws_title, df2, bust_cache=True)
            st.success("Beräkningar sparade till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")


# ---------- Portfölj ----------

def _fx_for(valuta: str, rates: Dict[str, float]) -> float:
    if not valuta:
        return 1.0
    v = str(valuta).upper().strip()
    return float(rates.get(v, 1.0))

def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")

    port = df.copy()
    # Antal aktier kan komma in som str med , eller .
    port["Antal aktier"] = port["Antal aktier"].map(_to_float)
    port = port[port["Antal aktier"] > 0]

    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Säkerställ numerik
    port["Aktuell kurs"] = port["Aktuell kurs"].map(_to_float)
    port["Årlig utdelning"] = port["Årlig utdelning"].map(_to_float)
    port["GAV (SEK)"] = port["GAV (SEK)"].map(_to_float)

    # FX & SEK-värden
    port["FX→SEK"] = port["Valuta"].apply(lambda v: _fx_for(v, rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["FX→SEK"]
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    port["Vinst (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst (%)"] = np.where(port["Anskaffningsvärde (SEK)"] > 0,
                                 (port["Vinst (SEK)"] / port["Anskaffningsvärde (SEK)"]) * 100.0, 0.0)
    port["DA (%)"] = np.where(port["Aktuell kurs"] > 0,
                              (port["Årlig utdelning"] / port["Aktuell kurs"]) * 100.0, 0.0)
    # YOC = utdelning i SEK per aktie / GAV(SEK)
    port["YOC (%)"] = np.where(port["GAV (SEK)"] > 0,
                               (port["Årlig utdelning"] * port["FX→SEK"] / port["GAV (SEK)"]) * 100.0, 0.0)

    total_value = float(port["Värde (SEK)"].sum())
    total_cost  = float(port["Anskaffningsvärde (SEK)"].sum())
    total_gain  = total_value - total_cost
    total_gain_pct = (total_gain / total_cost * 100.0) if total_cost > 0 else 0.0

    port["Andel (%)"] = np.where(total_value > 0, (port["Värde (SEK)"]/total_value)*100.0, 0.0)

    # Headline
    st.markdown(
        f"""
**Totalt portföljvärde:** {round(total_value,2)} SEK  
**Anskaffningsvärde:** {round(total_cost,2)} SEK  
**Vinst / Förlust:** {round(total_gain,2)} SEK ({round(total_gain_pct,2)} %)  
"""
    )

    cols = [
        "Ticker","Bolagsnamn","Sektor","Antal aktier","Aktuell kurs","Valuta","FX→SEK",
        "GAV (SEK)","Värde (SEK)","Anskaffningsvärde (SEK)","Vinst (SEK)","Vinst (%)",
        "Årlig utdelning","DA (%)","YOC (%)"
    ]
    st.dataframe(
        port[cols].sort_values("Andel (%)", ascending=False),
        use_container_width=True
    )


# ---------- Köpförslag ----------

def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")

    if df.empty:
        st.info("Inga rader."); 
        return

    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    strategy = st.selectbox("Strategi", ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0)

    subset = st.radio("Visa", ["Alla bolag","Endast portfölj"], horizontal=True)
    base = df.copy()
    if subset == "Endast portfölj":
        base["Antal aktier"] = base["Antal aktier"].map(_to_float)
        base = base[base["Antal aktier"] > 0].copy()

    base = update_calculations(base)
    base = base[(base[horizon].map(_to_float) > 0) & (base["Aktuell kurs"].map(_to_float) > 0)].copy()
    if base.empty:
        st.info("Inget att visa.")
        return

    base = score_rows(base, horizon=horizon, strategy=("Auto" if strategy.startswith("Auto") else strategy))

    show_components = st.checkbox("Visa komponentpoäng (Growth/Dividend/Financials) för vald horisont", True)
    show_saved = st.checkbox("Visa sparade horisontpoäng från Google Sheets", True)

    tag = _horizon_to_tag(horizon)
    saved_cols_all = [
        "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",
        "Score Growth (Idag)", "Score Dividend (Idag)", "Score Financials (Idag)",
        "Score Growth (1 år)", "Score Dividend (1 år)", "Score Financials (1 år)",
        "Score Growth (2 år)", "Score Dividend (2 år)", "Score Financials (2 år)",
        "Score Growth (3 år)", "Score Dividend (3 år)", "Score Financials (3 år)",
    ]
    available_saved = [c for c in saved_cols_all if c in base.columns]

    default_saved = [f"Score Total ({tag})"]
    for group in ["Growth","Dividend","Financials"]:
        colname = f"Score {group} ({tag})"
        if colname in available_saved:
            default_saved.append(colname)

    selected_saved_cols = []
    if show_saved and available_saved:
        selected_saved_cols = st.multiselect(
            "Välj sparade score-kolumner att visa",
            options=available_saved,
            default=default_saved
        )

    sort_options = ["Score (Total)", "Uppsida (%)", "DA (%)"]
    for c in ["Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)"]:
        if c in base.columns and c not in sort_options:
            sort_options.append(c)
    sort_on = st.selectbox("Sortera på", sort_options, index=0)

    base["Uppsida (%)"] = ((base[horizon].map(_to_float) - base["Aktuell kurs"].map(_to_float)) /
                           base["Aktuell kurs"].map(_to_float) * 100.0).round(2)
    base["DA (%)"] = np.where(base["Aktuell kurs"].map(_to_float) > 0,
                              (base["Årlig utdelning"].map(_to_float) / base["Aktuell kurs"].map(_to_float)) * 100.0, 0.0).round(2)

    ascending = False
    if sort_on == "Uppsida (%)":
        trim_mode = st.checkbox("Visa trim/sälj-läge (minst uppsida först)", value=False)
        if trim_mode:
            ascending = True
    reverse_global = st.checkbox("Omvänd sortering (gäller valt fält)", value=False)
    if reverse_global:
        ascending = not ascending

    cols = ["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)","DA (%)"]
    if show_components:
        cols += ["Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence"]
    else:
        cols += ["Score (Total)","Confidence"]
    if show_saved and selected_saved_cols:
        cols += selected_saved_cols

    base = base.sort_values(by=[sort_on], ascending=ascending).reset_index(drop=True)
    st.dataframe(base[cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### Kortvisning (bläddra)")
    if "idea_idx" not in st.session_state:
        st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input(
        "Visa rad #", min_value=0, max_value=max(0, len(base)-1),
        value=st.session_state["idea_idx"], step=1
    )
    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- **Sektor:** {r.get('Sektor','—')}")
        st.write(f"- **Aktuell kurs:** {round(_to_float(r['Aktuell kurs']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs idag:** {round(_to_float(r['Riktkurs idag']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 1 år:** {round(_to_float(r['Riktkurs om 1 år']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 2 år:** {round(_to_float(r['Riktkurs om 2 år']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 3 år:** {round(_to_float(r['Riktkurs om 3 år']),2)} {r['Valuta']}")
        st.write(f"- **Uppsida ({horizon}):** {round(_to_float(r['Uppsida (%)']),2)} %")
    with c2:
        st.write(f"- **P/S-snitt (Q1..Q4):** {round(_to_float(r['P/S-snitt (Q1..Q4)']),2)}")
        st.write(f"- **P/B-snitt (Q1..Q4):** {round(_to_float(r['P/B-snitt (Q1..Q4)']),2)}")
        st.write(f"- **Omsättning idag (M):** {round(_to_float(r['Omsättning idag']),2)}")
        st.write(f"- **Omsättning nästa år (M):** {round(_to_float(r['Omsättning nästa år']),2)}")
        st.write(f"- **Årlig utdelning:** {round(_to_float(r['Årlig utdelning']),2)}")
        st.write(f"- **Payout:** {round(_to_float(r['Payout (%)']),2)} %")
        st.write(f"- **DA (egen):** {round(_to_float(r['DA (%)']),2)} %")
        st.write(f"- **CAGR 5 år:** {round(_to_float(r['CAGR 5 år (%)']),2)} %")
        st.write(f"- **Score – Growth / Dividend / Financials / Total:** "
                 f"{round(_to_float(r['Score (Growth)']),1)} / "
                 f"{round(_to_float(r['Score (Dividend)']),1)} / "
                 f"{round(_to_float(r['Score (Financials)']),1)} / "
                 f"**{round(_to_float(r['Score (Total)']),1)}** "
                 f"(Conf {int(_to_float(r['Confidence']))}%)")


# ---------- Utdelningskalender ----------

def view_dividend_calendar(df: pd.DataFrame, ws_title: str, rates: Dict[str, float]):
    st.subheader("📅 Utdelningskalender (12 månader framåt)")
    months_forward = st.number_input("Antal månader framåt", min_value=3, max_value=24, value=12, step=1)
    write_back = st.checkbox("Skriv tillbaka schema till databasen (Div_Frekvens/år, Div_Månader, Div_Vikter)", value=True)

    if st.button("Bygg kalender"):
        summ, det, df_out = build_dividend_calendar(df, rates, months_forward=int(months_forward), write_back_schedule=bool(write_back))
        st.session_state["div_summ"] = summ
        st.session_state["div_det"] = det
        st.session_state["div_df_out"] = df_out
        st.success("Kalender skapad.")

    if "div_summ" in st.session_state:
        st.markdown("### Summering per månad (SEK)")
        st.dataframe(st.session_state["div_summ"], use_container_width=True)
    if "div_det" in st.session_state:
        st.markdown("### Detalj per bolag/månad (SEK)")
        st.dataframe(st.session_state["div_det"], use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("💾 Spara schema + kalender till Google Sheets"):
        try:
            df_to_save = st.session_state.get("div_df_out", df)
            df2 = enrich_for_save(df_to_save, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            summ = st.session_state.get("div_summ", pd.DataFrame())
            det  = st.session_state.get("div_det", pd.DataFrame())
            ws_write_df("Utdelningskalender – Summering", summ if not summ.empty else pd.DataFrame(columns=["År","Månad","Månad (sv)","Summa (SEK)"]))
            ws_write_df("Utdelningskalender – Detalj", det if not det.empty else pd.DataFrame(columns=[
                "År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta","Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"]))
            st.success("Schema + kalender sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    if c2.button("↻ Rensa kalender-cache"):
        for k in ["div_summ","div_det","div_df_out"]:
            if k in st.session_state:
                del st.session_state[k]
        st.info("Kalender-cache rensad.")

# app.py (Del 5/5)

def main():
    st.title("K-pf-rslag")

    # Välj Google Sheets-blad
    try:
        titles = list_worksheet_titles() or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # Sidopanel: tvinga omläsning (om du ändrat i arket)
    if st.sidebar.button("↻ Läs om data nu"):
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        st.rerun()

    # Valutakurser (med live/spara)
    user_rates = sidebar_rates()

    # Läs data + snapshot + grundberäkningar
    df = load_df(ws_title)
    snapshot_on_start(df, ws_title)     # skapar 5-dagars snapshot & städar äldre
    df = update_calculations(df)

    # Vyer
    tabs = st.tabs([
        "📄 Data",
        "🧩 Manuell insamling",
        "📦 Portfölj",
        "💡 Köpförslag",
        "📅 Utdelningskalender",
    ])
    with tabs[0]:
        view_data(df, ws_title)
    with tabs[1]:
        view_manual(df, ws_title)
    with tabs[2]:
        view_portfolio(df, user_rates)
    with tabs[3]:
        view_ideas(df)
    with tabs[4]:
        view_dividend_calendar(df, ws_title, user_rates)


if __name__ == "__main__":
    main()
