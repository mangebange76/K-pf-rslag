# -*- coding: utf-8 -*-
"""
stockapp.fetchers.orchestrator
------------------------------
Kör full uppdatering av en ticker i ordningen:
  1) Yahoo → 2) FMP → 3) SEC (om modul finns)

Mergar nyckeltal enligt en tydlig prioritet per fält,
skriver endast icke-tomma värden, rör INTE manuella prognosfält,
stämplar TS-kolumner och uppdaterar Risklabel.

Publikt API:
- run_update_full(df, ticker, user_rates) -> (df_out, logstr)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import pandas as pd

# Källor
from .yahoo import get_all as yf_get_all
from .fmp import get_all as fmp_get_all
try:
    from .sec import get_all as sec_get_all  # valfri
except Exception:
    sec_get_all = None  # type: ignore

# App-helpers
from ..config import FINAL_COLS
from ..utils import (
    ensure_schema,
    now_stamp,
    to_float,
    risk_label_from_mcap,
)

# ---- Fältprioritet (källor i ordning: först som finns vinner) ----------
#  - Standard: Yahoo → FMP → SEC
#  - För vissa fält (marginaler/kvoter) är FMP oftast bättre → lyft FMP först.
FIELD_PRIORITY: Dict[str, List[str]] = {
    # Bas
    "Bolagsnamn": ["yahoo", "fmp", "sec"],
    "Valuta": ["yahoo", "fmp", "sec"],
    "Kurs": ["yahoo", "fmp", "sec"],
    "Market Cap": ["yahoo", "fmp", "sec"],
    "Utestående aktier (milj.)": ["fmp", "yahoo", "sec"],

    # Multiplar & marginaler
    "EV/EBITDA (ttm)": ["fmp", "yahoo", "sec"],
    "P/B": ["fmp", "yahoo", "sec"],
    "P/S": ["fmp", "yahoo", "sec"],
    "Gross margin (%)": ["fmp", "yahoo", "sec"],
    "Operating margin (%)": ["fmp", "yahoo", "sec"],
    "Net margin (%)": ["fmp", "yahoo", "sec"],
    "ROE (%)": ["fmp", "yahoo", "sec"],
    "Debt/Equity": ["fmp", "yahoo", "sec"],
    "Net debt / EBITDA": ["fmp", "sec", "yahoo"],
    "FCF Yield (%)": ["fmp", "yahoo", "sec"],
    "Dividend yield (%)": ["fmp", "yahoo", "sec"],
    "Dividend payout (FCF) (%)": ["fmp", "sec", "yahoo"],
    "Kassa (M)": ["fmp", "sec", "yahoo"],

    # P/S-kvartal
    "P/S Q1": ["fmp", "yahoo", "sec"],
    "P/S Q2": ["fmp", "yahoo", "sec"],
    "P/S Q3": ["fmp", "yahoo", "sec"],
    "P/S Q4": ["fmp", "yahoo", "sec"],

    # Klassificering
    "Sektor": ["fmp", "yahoo", "sec"],
    "Industri": ["fmp", "yahoo", "sec"],
}

# Vilka fält vi ALDRIG skriver automatiskt (manuella prognoser)
MANUAL_FIELDS = {
    "Omsättning i år (M)",
    "Omsättning nästa år (M)",
}

# ----------------------------------------------------------------------

def _non_empty(v: Any) -> bool:
    """True om v är ett 'sättbart' värde (tillåter 0.0)."""
    if v is None:
        return False
    if isinstance(v, float):
        if math.isnan(v):
            return False
        return True
    if isinstance(v, str):
        return v.strip() != ""
    return True

def _pick_value(key: str, yv: Dict[str, Any], fv: Dict[str, Any], sv: Dict[str, Any]) -> Any:
    """Välj värde för fält 'key' enligt FIELD_PRIORITY. Fallback: yahoo→fmp→sec."""
    prio = FIELD_PRIORITY.get(key, ["yahoo", "fmp", "sec"])
    for src in prio:
        if src == "yahoo" and _non_empty(yv.get(key)):
            return yv[key]
        if src == "fmp" and _non_empty(fv.get(key)):
            return fv[key]
        if src == "sec" and sv is not None and _non_empty(sv.get(key)):
            return sv[key]
    return None

def _compute_ps_avg(row_like: Dict[str, Any]) -> Optional[float]:
    """Beräkna P/S-snitt (Q1..Q4) om P/S Q1..Q4 finns i row_like."""
    qvals: List[float] = []
    for q in ("P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        v = to_float(row_like.get(q, None))
        if v is not None and not math.isnan(v) and v > 0:
            qvals.append(float(v))
    if not qvals:
        return None
    return round(sum(qvals) / float(len(qvals)), 4)

def _merge_sources(yv: Dict[str, Any], fv: Dict[str, Any], sv: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Slå ihop fält från källor enligt prioritet; rör inte MANUAL_FIELDS."""
    keys = set(yv.keys()) | set(fv.keys()) | (set(sv.keys()) if sv else set())
    out: Dict[str, Any] = {}
    for k in sorted(keys):
        if k in MANUAL_FIELDS:
            continue  # manuella fält hoppar vi
        val = _pick_value(k, yv, fv, (sv or {}))
        if _non_empty(val):
            out[k] = val

    # Lägg P/S-snitt (Q1..Q4) om möjliga kvartal finns
    ps_avg = _compute_ps_avg(out)
    if ps_avg is not None:
        out["P/S-snitt (Q1..Q4)"] = ps_avg

    # Risklabel (om Market Cap finns)
    mc = to_float(out.get("Market Cap", None))
    if mc is not None and mc > 0:
        out["Risklabel"] = risk_label_from_mcap(mc)

    return out

# ----------------------------------------------------------------------

def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """
    Full uppdatering av en ticker. Returnerar (df_out, logtext).
    - Hämtar från Yahoo, FMP och SEC (om SEC-modul finns).
    - Mergar enligt FIELD_PRIORITY.
    - Stämplar TS-kolumner: 'TS Full' (alltid) och 'TS Kurs' om Kurs ändrades.
    - Skapar rad om tickern inte finns sedan tidigare.
    """
    if df is None:
        df = pd.DataFrame(columns=FINAL_COLS)
    df = ensure_schema(df.copy(), FINAL_COLS)

    tkr = str(ticker).upper().strip()
    idxs = df.index[df["Ticker"].astype(str).str.upper() == tkr].tolist()
    if not idxs:
        # skapa ny rad om den inte finns
        newrow = {c: None for c in FINAL_COLS}
        newrow["Ticker"] = tkr
        df = pd.concat([df, pd.DataFrame([newrow])], ignore_index=True)
        ridx = df.index[-1]
    else:
        ridx = idxs[0]

    # Ursprungsvärden (för att upptäcka ändring på Kurs)
    old_price = to_float(df.at[ridx, "Kurs"]) if "Kurs" in df.columns else None

    # -- Hämta källor --
    yv = yf_get_all(tkr) or {}
    fv = fmp_get_all(tkr) or {}
    sv = {}
    if sec_get_all is not None:
        try:
            sv = sec_get_all(tkr) or {}
        except Exception:
            sv = {}

    y_count = len([k for k in yv.keys() if not str(k).startswith("__")])
    f_count = len([k for k in fv.keys() if not str(k).startswith("__")])
    s_count = len([k for k in sv.keys() if not str(k).startswith("__")]) if sv else 0

    merged = _merge_sources(yv, fv, sv)

    # -- Skriv in i df --
    changed_fields: List[str] = []
    for k, v in merged.items():
        if k not in df.columns:
            # lägg inte till nya okända kolumner här; håll oss till FINAL_COLS
            continue
        prev = df.at[ridx, k]
        # sätt om skillnad eller tidigare tomt
        if (prev is None) or (isinstance(prev, float) and math.isnan(prev)) or (str(prev) != str(v)):
            df.at[ridx, k] = v
            changed_fields.append(k)

    # TS-stämplar
    now = now_stamp()
    if "TS Full" in df.columns:
        df.at[ridx, "TS Full"] = now

    new_price = to_float(df.at[ridx, "Kurs"]) if "Kurs" in df.columns else None
    if new_price is not None and (old_price is None or float(new_price) != float(old_price)):
        if "TS Kurs" in df.columns:
            df.at[ridx, "TS Kurs"] = now

    # Loggtext
    upd_n = len(changed_fields)
    log = f"{tkr}: Yahoo {y_count} fält, FMP {f_count} fält, SEC {s_count} fält → uppdaterade {upd_n} fält."
    if upd_n > 0:
        log += " (" + ", ".join(sorted(changed_fields)) + ")"

    return df, log
