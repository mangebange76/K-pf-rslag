# stockapp/fetchers/orchestrator.py
# -*- coding: utf-8 -*-
"""
Orchestrator – samlar SEC + FMP + Yahoo och uppdaterar en rad i DataFrame.

Publikt API:
    run_update_full(df, ticker, user_rates) -> (df_out, logmsg)

- SEC: hämtar kvartalsintäkter (för TTM och P/S Q1..Q4)
- FMP: fallback för nyckeltal (marginaler, EV/EBITDA, shares, utdelning) där Yahoo saknar
- Yahoo: pris, market cap, valuta, sektor/industri, utdelning, marginaler, EV/EBITDA, D/E

OBS:
- Inga Streamlit-anrop här; ren datafunktion.
- Robust kolumnsättning: om kolumn saknas skapas den.
- Stämplar tidsstämplar för uppdaterade fält om "* TS" eller "TS_*" finns.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from datetime import datetime

# Källor
from .yahoo import fetch_ticker as yh_fetch, get_live_price as yh_price
# Dessa moduler ska redan finnas i projektet (vi anropar defensivt)
try:
    from .fmp import fetch_ticker as fmp_fetch  # type: ignore
except Exception:
    fmp_fetch = None  # type: ignore

try:
    # förväntat: fetch_quarterly_revenue(ticker) -> List[{"date": "YYYY-MM-DD", "revenue": float}]
    from .sec import fetch_quarterly_revenue as sec_fetch_quarters  # type: ignore
except Exception:
    sec_fetch_quarters = None  # type: ignore


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        try:
            v = float(str(x).replace(",", "."))
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

def _ensure_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        df[col] = np.nan
    return df

def _set(df: pd.DataFrame, idx, col: str, val: Any):
    if col not in df.columns:
        df[col] = np.nan
    df.loc[idx, col] = val

def _stamp_ts(df: pd.DataFrame, idx, field: str):
    """
    Stämpla tidsstämpel. Stöd både 'FIELD TS' och 'TS_FIELD' om de finns i df.
    """
    ts_val = _now_stamp()
    # suffix-variant
    suf = f"{field} TS"
    if suf in df.columns:
        df.loc[idx, suf] = ts_val
    # prefix-variant
    pre = f"TS_{field}"
    if pre in df.columns:
        df.loc[idx, pre] = ts_val

def _risk_label_from_mcap(mc: Optional[float]) -> str:
    if mc is None:
        return "Unknown"
    # USD gränser (kan mappas i UI till basvaluta)
    if mc >= 2e11:   # >= 200B
        return "Mega"
    if mc >= 1e10:   # 10B–200B
        return "Large"
    if mc >= 2e9:    # 2B–10B
        return "Mid"
    if mc >= 3e8:    # 300M–2B
        return "Small"
    return "Micro"

def _ttm_series_from_quarters(qrows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    Tar en lista med kvartal (nyast först) [{"date": "...", "revenue": float}, ...]
    och beräknar TTM vid varje kvartalsslut: (date, ttm)
    Kräver minst 4 kvartal.
    """
    out: List[Tuple[str, float]] = []
    if not qrows or len(qrows) < 4:
        return out
    # Se till att nyast först:
    rows = list(qrows)
    # säkerställ numerik
    rev = [_to_float(r.get("revenue")) or 0.0 for r in rows]
    for i in range(0, min(4, len(rows))):  # TTM för q0..q3 om möjligt
        if i + 4 <= len(rows):
            ttm = sum(rev[i:i+4])
            out.append((rows[i].get("date") or f"Q{i}", ttm))
    return out  # [(date_newest, ttm_newest), (date-1, ttm-1), ...]

def _ps_from_mcap_ttm(mcap: Optional[float], ttm: Optional[float]) -> Optional[float]:
    if mcap is None or ttm is None:
        return None
    if ttm <= 0:
        return None
    return float(mcap) / float(ttm)


# ------------------------------------------------------------
# Huvud: run_update_full
# ------------------------------------------------------------
def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, str]:
    """
    Uppdaterar en ticker i df med SEC + FMP + Yahoo.
    Returnerar (df_out, logmsg).
    """
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    tkr = str(ticker).strip().upper()

    # Hitta rad eller skapa ny
    mask = df["Ticker"].astype(str).str.upper() == tkr
    if not mask.any():
        # skapa ny rad
        new_row = {c: np.nan for c in df.columns}
        new_row["Ticker"] = tkr
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        mask = df["Ticker"].astype(str).str.upper() == tkr
    ridx = df.index[mask]

    # Säkerställ vanliga kolumner
    for col in [
        "Bolagsnamn", "Valuta", "Kurs", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Market Cap", "Market Cap (valuta)", "Market Cap (SEK)",
        "Årlig utdelning", "Dividend Yield (%)",
        "EV/EBITDA", "Bruttomarginal (%)", "Nettomarginal (%)", "Debt/Equity",
        "Sektor", "Industri",
        "Senast auto-uppdaterad", "Senast uppdaterad källa"
    ]:
        _ensure_col(df, col)

    # -----------------------------
    # Yahoo först (snabb och ofta komplett)
    # -----------------------------
    y = yh_fetch(tkr) or {}
    price = _to_float(y.get("price"))
    currency = y.get("currency")
    mcap = _to_float(y.get("market_cap"))
    shares = _to_float(y.get("shares_outstanding"))
    name = y.get("name")

    annual_div = _to_float(y.get("annual_dividend"))
    div_y = _to_float(y.get("dividend_yield_pct"))
    ev_ebitda = _to_float(y.get("ev_ebitda"))
    gm_pct = _to_float(y.get("gross_margin_pct"))
    nm_pct = _to_float(y.get("net_margin_pct"))
    dte = _to_float(y.get("debt_to_equity"))
    sector = y.get("sector")
    industry = y.get("industry")

    # -----------------------------
    # FMP – fyll luckor
    # -----------------------------
    f = {}
    if fmp_fetch is not None:
        try:
            f = fmp_fetch(tkr) or {}
        except Exception:
            f = {}

    if price is None:
        price = _to_float(f.get("price"))
    if currency is None:
        currency = f.get("currency") or "USD"
    if mcap is None:
        mcap = _to_float(f.get("market_cap"))
    if shares is None:
        shares = _to_float(f.get("shares_outstanding"))
    if name is None:
        name = f.get("name")
    if annual_div is None:
        annual_div = _to_float(f.get("annual_dividend"))
    if div_y is None:
        div_y = _to_float(f.get("dividend_yield_pct"))
    if ev_ebitda is None:
        ev_ebitda = _to_float(f.get("ev_ebitda"))
    if gm_pct is None:
        gm_pct = _to_float(f.get("gross_margin_pct"))
    if nm_pct is None:
        nm_pct = _to_float(f.get("net_margin_pct"))
    if dte is None:
        dte = _to_float(f.get("debt_to_equity"))
    if sector is None:
        sector = f.get("sector")
    if industry is None:
        industry = f.get("industry")

    # Beräkna mcap om saknas men vi har price*shares
    if mcap is None and (price is not None and shares is not None):
        mcap = float(price) * float(shares)

    # -----------------------------
    # SEC – kvartalsintäkter -> TTM & P/S Q1..Q4
    # -----------------------------
    ps_q = [None, None, None, None]  # Q1..Q4
    p_s_current = None
    p_s_avg = None

    quarters: List[Dict[str, Any]] = []
    if sec_fetch_quarters is not None:
        try:
            quarters = sec_fetch_quarters(tkr) or []
        except Exception:
            quarters = []

    # Om SEC saknas kan FMP ibland ha quarterly revenue – vi försöker läsa kompatibelt fält
    if not quarters and f:
        qf = f.get("quarters") or []
        # Försök mappa till [{"date": ..., "revenue": ...}, ...]
        mapped = []
        for it in qf:
            rev = it.get("revenue") or it.get("sales") or it.get("revenueQuarter")
            dt = it.get("date") or it.get("period") or it.get("fiscalDateEnding")
            rv = _to_float(rev)
            if rv is not None and dt:
                mapped.append({"date": str(dt), "revenue": float(rv)})
        if mapped:
            quarters = mapped

    # Nu har vi (förhoppningsvis) quarters (nyast först). Räkna TTM-serie.
    ttm_series = _ttm_series_from_quarters(quarters)  # [(date0, ttm0), (date1, ttm1), ...]
    if mcap is not None and ttm_series:
        # Q1..Q4 = de 4 senaste TTM
        for i in range(min(4, len(ttm_series))):
            ps_q[i] = _ps_from_mcap_ttm(mcap, ttm_series[i][1])
        # nuvarande P/S = TTM senaste
        p_s_current = ps_q[0]
        # medel
        vals = [v for v in ps_q if v is not None]
        if vals:
            p_s_avg = float(np.mean(vals))

    # -----------------------------
    # Skriv in i DataFrame
    # -----------------------------
    # Namn/valuta
    if name:
        _set(df, ridx, "Bolagsnamn", name)
    if currency:
        _set(df, ridx, "Valuta", currency)

    # Pris -> stöd både "Kurs" och "Aktuell kurs"
    if price is not None:
        _set(df, ridx, "Kurs", price)
        _set(df, ridx, "Aktuell kurs", price)
        _stamp_ts(df, ridx, "Kurs")
        _stamp_ts(df, ridx, "Aktuell kurs")

    # Utestående aktier
    if shares is not None:
        _set(df, ridx, "Utestående aktier", float(shares))
        _stamp_ts(df, ridx, "Utestående aktier")

    # Market cap
    if mcap is not None:
        _set(df, ridx, "Market Cap", float(mcap))
        _set(df, ridx, "Market Cap (valuta)", float(mcap))  # i basvalutan (t.ex. USD)
        # SEK-kolumn om finns: kräver växelkurs; sätts i UI-beräkningar normalt
        _stamp_ts(df, ridx, "Market Cap")
        _stamp_ts(df, ridx, "Market Cap (valuta)")

    # P/S nu + Q1..Q4 + snitt
    if p_s_current is not None:
        _set(df, ridx, "P/S", float(p_s_current))
        _stamp_ts(df, ridx, "P/S")
    labels = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
    for i, lab in enumerate(labels):
        if ps_q[i] is not None:
            _set(df, ridx, lab, float(ps_q[i]))
            _stamp_ts(df, ridx, lab)
    if p_s_avg is not None:
        _set(df, ridx, "P/S-snitt", float(p_s_avg))

    # Utdelning
    if annual_div is not None:
        _set(df, ridx, "Årlig utdelning", float(annual_div))
    if div_y is not None:
        _set(df, ridx, "Dividend Yield (%)", float(div_y))

    # Övriga nyckeltal
    if ev_ebitda is not None:
        _set(df, ridx, "EV/EBITDA", float(ev_ebitda))
    if gm_pct is not None:
        _set(df, ridx, "Bruttomarginal (%)", float(gm_pct))
    if nm_pct is not None:
        _set(df, ridx, "Nettomarginal (%)", float(nm_pct))
    if dte is not None:
        _set(df, ridx, "Debt/Equity", float(dte))
    if sector:
        _set(df, ridx, "Sektor", str(sector))
    if industry:
        _set(df, ridx, "Industri", str(industry))

    # Risklabel (enkel mcap-baserad)
    _ensure_col(df, "Risklabel")
    if mcap is not None:
        df.loc[ridx, "Risklabel"] = _risk_label_from_mcap(mcap)

    # Metadata
    _set(df, ridx, "Senast auto-uppdaterad", _now_stamp())
    _set(df, ridx, "Senast uppdaterad källa",
         "SEC+FMP+Yahoo" if quarters else ("FMP+Yahoo" if f else "Yahoo"))

    # Klar
    # loggmeddelande (kort)
    parts = []
    if price is not None:
        parts.append(f"pris={price:.2f}")
    if mcap is not None:
        parts.append(f"mcap≈{mcap:.0f}")
    if p_s_current is not None:
        parts.append(f"P/S={p_s_current:.2f}")
    msg = f"{ticker}: " + ", ".join(parts) if parts else f"{ticker}: uppdaterad"

    return df, msg
