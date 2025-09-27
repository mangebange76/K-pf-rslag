# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st

try:
    import pytz
    TZ = pytz.timezone("Europe/Stockholm")
    def now_dt(): return datetime.now(TZ)
except Exception:
    def now_dt(): return datetime.now()

def now_stamp() -> str:
    return now_dt().strftime("%Y-%m-%d")

def _ts_str(): return now_dt().strftime("%Y%m%d-%H%M%S")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def with_backoff(func, *args, **kwargs):
    delays = [0, 0.4, 0.8, 1.6]
    last = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta: return 1.0
    return float(user_rates.get(valuta.upper(), 1.0))

def auto_rates_fetch():
    """Frankfurter → exchangerate.host fallback."""
    out = {}
    try:
        for base in ("USD","NOK","CAD","EUR"):
            r = requests.get("https://api.frankfurter.app/latest",
                             params={"from": base, "to": "SEK"}, timeout=12)
            if r.status_code == 200:
                v = (r.json() or {}).get("rates", {}).get("SEK")
                if v: out[base] = float(v)
    except Exception:
        pass
    if len(out) < 4:
        try:
            for base in ("USD","NOK","CAD","EUR"):
                r = requests.get("https://api.exchangerate.host/latest",
                                 params={"base": base, "symbols": "SEK"}, timeout=12)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v: out[base] = float(v)
        except Exception:
            pass
    return out

def oldest_any_ts(row: pd.Series, ts_fields_map: dict) -> pd.Timestamp|None:
    dates = []
    for c in ts_fields_map.values():
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d): dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame, ts_fields_map: dict) -> pd.DataFrame:
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(lambda r: oldest_any_ts(r, ts_fields_map), axis=1)
    work["_oldest_any_ts"] = pd.to_datetime(work["_oldest_any_ts"], errors="coerce")
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return work

def make_pretty_money(x: float, unit: str = "") -> str:
    try: v = float(x)
    except: return "-"
    abs_v = abs(v)
    if abs_v >= 1e12: s = f"{v/1e12:.2f} T"
    elif abs_v >= 1e9: s = f"{v/1e9:.2f} B"
    elif abs_v >= 1e6: s = f"{v/1e6:.2f} M"
    else: s = f"{v:.0f}"
    return f"{s} {unit}".strip()

def risklabel_from_mcap_sek(mcap_sek: float) -> str:
    from .config import CAP_BOUNDS_SEK
    try: v = float(mcap_sek)
    except: v = 0.0
    for name, lo, hi in CAP_BOUNDS_SEK:
        if lo <= v < hi: return name
    return "Mega"

def säkerställ_kolumner(df: pd.DataFrame, final_cols: list) -> pd.DataFrame:
    df = df.copy()
    for kol in final_cols:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","ev","cap","equity","margin","cash","flow","burn","runway","mcap","gav"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sector","Industry"):
                df[kol] = ""
            else:
                df[kol] = ""
    # ta bort ev dubletter
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "EV","EBITDA","EV/EBITDA","Market Cap (valuta)","Market Cap (SEK)",
        "Debt/Equity","Gross Margin (%)","Net Margin (%)",
        "Cash & Equivalents","Free Cash Flow","FCF Margin (%)",
        "Monthly Burn","Runway (quarters)","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4",
        "GAV SEK"
    ]
    df = df.copy()
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Sector","Industry","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int, ts_map: dict) -> pd.DataFrame:
    need_cols = ["Omsättning idag","Omsättning nästa år"]
    ts_cols = [ts_map[c] for c in ts_map if c in need_cols]
    out_rows = []
    cutoff = now_dt() - timedelta(days=older_than_days)

    for _, r in df.iterrows():
        missing_val = any((float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
        missing_ts  = any((not str(r.get(ts, "")).strip()) for ts in ts_cols if ts in r)
        oldest = oldest_any_ts(r, ts_map)
        oldest_dt = pd.to_datetime(oldest).to_pydatetime() if pd.notna(oldest) else None
        too_old = (oldest_dt is not None and oldest_dt < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "TS_Omsättning idag": r.get(ts_map["Omsättning idag"], ""),
                "TS_Omsättning nästa år": r.get(ts_map["Omsättning nästa år"], ""),
            })
    return pd.DataFrame(out_rows)
