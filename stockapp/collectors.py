# stockapp/collectors.py
from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Källor
from .fetchers.yahoo import get_all as yf_get_all
from .fetchers.finviz import get_overview as finviz_get_overview
from .fetchers.morningstar import get_overview as ms_get_overview
from .fetchers.sec import get_pb_quarters as sec_get_pb_quarters
from .fetchers.stocktwits import get_symbol_summary as stw_get_symbol_summary


# -------------------------
# Hjälpare
# -------------------------
def _nz(x, y):
    """första icke-noll/icke-tomma."""
    if x is None:
        return y
    if isinstance(x, (int, float)):
        return x if float(x) != 0.0 else y
    if isinstance(x, str):
        return x if x.strip() else y
    return x or y


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _pick(*vals, default=0.0, prefer="first_nonzero"):
    if prefer == "first_nonzero":
        for v in vals:
            try:
                if isinstance(v, str):
                    if v.strip():
                        return v
                else:
                    if v is not None and float(v) != 0.0:
                        return v
            except Exception:
                continue
        return default
    return default


def _maybe_div_from_yield(price: float, yield_pct: float) -> float:
    if price and yield_pct:
        return float(price) * float(yield_pct) / 100.0
    return 0.0


def _pb_quarters_from_sec(ticker: str) -> Dict[str, float]:
    """
    SEC → P/B-kvartal: Q1=senaste kvartalet, Q2=nästa, ...
    Returnerar dict {"P/B Q1":..., "P/B Q2":..., "P/B Q3":..., "P/B Q4":...}
    """
    out = {"P/B Q1": 0.0, "P/B Q2": 0.0, "P/B Q3": 0.0, "P/B Q4": 0.0}
    try:
        secd = sec_get_pb_quarters(ticker) or {}
        pb_pairs: List[Tuple[str, float]] = secd.get("pb_quarters") or []
        if pb_pairs:
            # pb_pairs: [(date, pb)] nyast först (enligt fetchern)
            vals = [float(pb) for _, pb in pb_pairs][:4]
            for i, v in enumerate(vals, start=1):
                out[f"P/B Q{i}"] = round(v, 2)
    except Exception:
        pass
    return out


def _normalize_to_db(y: Dict[str, Any], f: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Slår ihop Yahoo + Finviz + Morningstar till appens kolumner
    med prioritet Yahoo → Finviz → Morningstar.
    """
    y = y or {}
    f = f or {}
    m = m or {}

    # Bas
    name     = _pick(y.get("name"),     None, None, default="")
    currency = _pick(y.get("currency"), None, None, default="USD")
    sector   = _pick(y.get("sector"),   None, None, default="")
    price    = _pick(y.get("price"),    f.get("price"), m.get("price"), default=0.0)

    # Utestående aktier (till miljoner)
    sh_abs = _pick(y.get("shares_outstanding"), f.get("shares_outstanding"), m.get("shares_outstanding"), default=0.0)
    sh_mil = (float(sh_abs) / 1e6) if float(sh_abs) > 0 else 0.0

    # Multiplar
    ps_ttm = _pick(y.get("ps_ttm"), f.get("ps_ttm"), m.get("ps_ttm"), default=0.0)
    pb     = _pick(y.get("pb"),     f.get("pb"),     m.get("pb"),     default=0.0)

    # Utdelning (årstakt): Yahoo dividend_rate → annars räkna från yield (Finviz/Morningstar)
    div_rate = _pick(y.get("dividend_rate"),
                     _maybe_div_from_yield(price, f.get("dividend_yield_pct", 0.0)),
                     _maybe_div_from_yield(price, m.get("dividend_yield_pct", 0.0)),
                     default=0.0)

    # Payout – föredra FCF om tillgänglig, annars EPS-baserad (Yahoo), annars Finviz/Morningstar
    payout_fcf = _to_float(y.get("payout_fcf_pct"))
    payout_eps = _to_float(y.get("payout_ratio_pct"))
    payout_fv  = _to_float(f.get("payout_ratio_pct"))
    payout_ms  = _to_float(m.get("payout_ratio_pct"))
    payout_pct = _pick(payout_fcf, payout_eps, payout_fv, payout_ms, default=0.0)

    # CAGR 5 år (Revenue)
    cagr5 = _to_float(y.get("cagr5_pct"))

    row = {
        "Bolagsnamn": name or "",
        "Valuta": str(currency or "USD").upper(),
        "Sektor": sector or "",
        "Aktuell kurs": float(price or 0.0),
        "Utestående aktier": float(sh_mil),
        "P/S": float(ps_ttm or 0.0),
        "P/B": float(pb or 0.0),
        "Årlig utdelning": float(div_rate or 0.0),
        "Payout (%)": float(payout_pct or 0.0),
        "CAGR 5 år (%)": float(cagr5 or 0.0),
    }
    return row


def collect_for_ticker(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ALLT vi kan från källorna och normaliserar till appens kolumner.
    Lägger även in P/B-kvartal från SEC.
    """
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return {}

    y = {}
    f = {}
    m = {}
    try:
        y = yf_get_all(tkr) or {}
    except Exception:
        y = {}
    try:
        f = finviz_get_overview(tkr) or {}
    except Exception:
        f = {}
    try:
        m = ms_get_overview(tkr) or {}
    except Exception:
        m = {}

    row = _normalize_to_db(y, f, m)

    # SEC P/B kvartal
    row.update(_pb_quarters_from_sec(tkr))

    # (valfritt) Stocktwits-sammanfattning – returneras men skrivs inte i DB av standardflödet
    try:
        stw = stw_get_symbol_summary(tkr) or {}
        row["_stw"] = stw  # om du vill använda i appens visning
    except Exception:
        pass

    return row


def mass_collect_and_apply(
    df: pd.DataFrame,
    ws_title: str | None = None,
    delay_sec: float = 0.5,
    save_fn=None,
) -> pd.DataFrame:
    """
    Loopar igenom df['Ticker'], hämtar allt och uppdaterar kolumnerna:
      - Bolagsnamn, Valuta, Sektor, Aktuell kurs, Utestående aktier (milj),
        P/S, P/B, Årlig utdelning, Payout (%), CAGR 5 år (%),
        P/B Q1..Q4 (från SEC).
    'save_fn' kan vara en funktion som tar (ws_title, df) och sparar till Google Sheets.
    """
    out = df.copy()
    tickers = out["Ticker"].astype(str).tolist()

    status = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    n = max(1, len(tickers))

    for i, tkr in enumerate(tickers):
        status.write(f"Hämtar {i+1}/{len(tickers)} – {tkr}")
        try:
            d = collect_for_ticker(tkr)
            if d:
                mask = (out["Ticker"].astype(str).str.upper() == tkr.upper())
                for k, v in d.items():
                    if k == "_stw":
                        continue  # inte skriv till DB just nu
                    # skriv endast om värde finns (str != "", num != 0) eller för P/B Qx alltid
                    if k.startswith("P/B Q"):
                        out.loc[mask, k] = float(v or 0.0)
                    else:
                        if isinstance(v, str):
                            if v.strip():
                                out.loc[mask, k] = v
                        else:
                            if float(v) != 0.0:
                                out.loc[mask, k] = float(v)
        except Exception:
            pass

        time.sleep(max(0.0, delay_sec))
        bar.progress((i + 1) / n)

    # valfri sparning via injicerad funktion (app.py skickar ws_write_df-wrapper)
    if save_fn and callable(save_fn) and ws_title:
        try:
            save_fn(ws_title, out)
            st.sidebar.success("Uppdaterade data sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

    return out
