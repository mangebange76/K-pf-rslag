# -*- coding: utf-8 -*-
"""
Orchestrator för full uppdatering.
- Primär källa: Yahoo (snapshot).
- Sekundära källor (om aktiverade i config och modulerna finns): FMP, SEC.
- Skriver tillbaka direkt i df-raden och stämplar TS-fält.
Returnerar (df, logtext).
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Any

import pandas as pd

from ..utils import now_stamp, risk_label_from_mcap

# Konfig-toggles (frivilliga)
try:
    from ..config import USE_YAHOO, USE_FMP, USE_SEC
except Exception:
    USE_YAHOO, USE_FMP, USE_SEC = True, False, False

# Yahoo är alltid fallback
from .yahoo import get_snapshot

# Dessa är valfria – finns de inte så kör vi bara Yahoo
try:
    from .fmp import get_company_metrics as fmp_metrics  # type: ignore
except Exception:  # pragma: no cover
    fmp_metrics = None  # type: ignore

try:
    from .sec import get_shares_outstanding as sec_shares  # type: ignore
except Exception:  # pragma: no cover
    sec_shares = None  # type: ignore


def _apply_row(df: pd.DataFrame, ridx, data: Dict[str, Any], updated: List[str]) -> None:
    for k, v in data.items():
        try:
            df.loc[ridx, k] = v
            updated.append(k)
        except Exception:
            pass


def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, str]:
    """
    Uppdaterar en rad i df för given ticker.
    - Hämtar snapshot från Yahoo (pris, valuta, namn, sektor, mcap, p/s, marginaler, etc).
    - Om FMP/SEC är aktiverade och moduler finns → kompletterar med fler fält.
    - Sätter Risklabel från Market Cap.
    - Stämplar "TS Full" och (om pris uppdaterats) "TS Kurs".
    """
    if df is None or df.empty:
        return df, "Tomt df"

    # hitta raden
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    idx = df.index[mask]
    if len(idx) == 0:
        return df, f"{ticker}: finns inte i tabellen"

    ridx = idx[0]
    updated: List[str] = []
    logs: List[str] = []

    # --- Yahoo ---------------------------------------------------------------
    if USE_YAHOO:
        y = {}
        try:
            y = get_snapshot(ticker) or {}
        except Exception as e:  # pragma: no cover
            logs.append(f"Yahoo: fel {e}")
            y = {}

        if y:
            _apply_row(df, ridx, y, updated)
            logs.append(f"Yahoo: {len(y)} fält")

    # --- SEC (endast aktier i miljoner om tillgängligt) ---------------------
    if USE_SEC and sec_shares is not None:
        try:
            sec = sec_shares(ticker) or {}
            if sec:
                _apply_row(df, ridx, sec, updated)
                logs.append("SEC: aktier")
        except Exception as e:  # pragma: no cover
            logs.append(f"SEC: fel {e}")

    # --- FMP (valfria nyckeltal) --------------------------------------------
    if USE_FMP and fmp_metrics is not None:
        try:
            f = fmp_metrics(ticker) or {}
            if f:
                _apply_row(df, ridx, f, updated)
                logs.append(f"FMP: {len(f)} fält")
        except Exception as e:  # pragma: no cover
            logs.append(f"FMP: fel {e}")

    # --- Risklabel från Market Cap ------------------------------------------
    try:
        mcap = df.loc[ridx].get("Market Cap")
        if pd.notna(mcap):
            df.loc[ridx, "Risklabel"] = risk_label_from_mcap(mcap)
    except Exception:
        pass

    # --- TS-stämplar --------------------------------------------------------
    df.loc[ridx, "TS Full"] = now_stamp()
    if "Kurs" in updated:
        df.loc[ridx, "TS Kurs"] = now_stamp()

    if not updated:
        return df, f"{ticker}: inga fält uppdaterades"
    return df, f"{ticker}: uppdaterade {len(updated)} fält ({', '.join(logs)})"
