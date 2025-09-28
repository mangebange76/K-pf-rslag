# stockapp/storage.py
# -*- coding: utf-8 -*-
"""
Högre nivå kring Google Sheets-lagring för appen.

Publika funktioner:
- hamta_data() -> pd.DataFrame
- spara_data(df, do_snapshot=False, snapshot_title=None, protect_from_wipe=True, dedupe=True) -> pd.DataFrame
- snapshot_now(df, title=None) -> str
- check_duplicate_tickers(df) -> List[str]
- merge_updates(df, updates) -> pd.DataFrame

All I/O mot Sheets går via stockapp.sheets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

from .config import FINAL_COLS, SHEET_NAME
from .utils import (
    ensure_final_cols,
    norm_ticker,
    ts_str,
    AppUserError,
)
from .sheets import (
    read_portfolio_df,
    save_portfolio_df,
    write_table,
    read_table,
    get_spreadsheet,
    get_ws,
)


# ---------------------------------------------------------------------
# Hjälp: dubbletter & normalisering
# ---------------------------------------------------------------------
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ slutkolumner och normalisera Ticker."""
    work = ensure_final_cols(df.copy())
    if "Ticker" in work.columns:
        work["Ticker"] = work["Ticker"].map(norm_ticker)
    return work


def check_duplicate_tickers(df: pd.DataFrame) -> List[str]:
    """Returnerar en lista av Ticker som förekommer fler än en gång (ignorerar tomma)."""
    if "Ticker" not in df.columns:
        return []
    ser = df["Ticker"].astype(str).str.strip().str.upper()
    ser = ser[ser != ""]
    dup = ser[ser.duplicated(keep=False)]
    return sorted(dup.unique().tolist())


# ---------------------------------------------------------------------
# Läs & spara
# ---------------------------------------------------------------------
def hamta_data() -> pd.DataFrame:
    """
    Läser huvudarket (SHEET_NAME) och garanterar FINAL_COLS.
    Använd denna i appen istället för att prata direkt med gspread.
    """
    df = read_portfolio_df()
    return _normalize_df(df)


def spara_data(
    df: pd.DataFrame,
    do_snapshot: bool = False,
    snapshot_title: Optional[str] = None,
    protect_from_wipe: bool = True,
    dedupe: bool = True,
) -> pd.DataFrame:
    """
    Sparar hela portfölj-tabellen.

    Skydd:
      - protect_from_wipe=True: Om df är tomt men befintligt ark INTE är tomt -> avbryt, kasta AppUserError.
      - dedupe=True: Dubbletter av Ticker är inte tillåtna -> avbryt, kasta AppUserError.

    Returnerar df (normaliserat) vid lyckad skrivning.
    """
    work = _normalize_df(df)

    # Blockera oavsiktlig wipe
    if protect_from_wipe:
        if work.empty:
            old = read_portfolio_df()
            if not old.empty:
                raise AppUserError(
                    "Skrivning avbruten: försökte spara TOM tabell medan befintlig databas innehåller rader."
                )

    # Dubblettskydd
    if dedupe:
        dups = check_duplicate_tickers(work)
        if dups:
            raise AppUserError(
                f"Dubbletter av Ticker ej tillåtna: {', '.join(dups)}. "
                "Rätta till innan sparning."
            )

    # Spara (med ev. snapshot)
    save_portfolio_df(work, do_snapshot=do_snapshot, snapshot_title=snapshot_title)
    return work


# ---------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------
def snapshot_now(df: pd.DataFrame, title: Optional[str] = None) -> str:
    """
    Skapar en snapshot av df i ett eget ark och returnerar arkets namn.
    """
    name = title or f"snapshot_{ts_str()}"
    work = _normalize_df(df)
    save_portfolio_df(work, do_snapshot=True, snapshot_title=name)
    return name


# ---------------------------------------------------------------------
# Smidig batch-uppdatering av värden i minnet
# ---------------------------------------------------------------------
def merge_updates(df: pd.DataFrame, updates: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """
    Tar en df och en uppsättning uppdateringar per ticker:
      updates = {
         "NVDA": {"Kurs (valuta)": 178.4, "Senast uppdaterad": "2025-09-28T12:00:00Z"},
         "AAPL": {"P/S Q1": 7.2}
      }
    Returnerar en NY DataFrame med fälten uppdaterade (sparar inte till Sheets).
    """
    work = _normalize_df(df)
    if "Ticker" not in work.columns or work.empty or not updates:
        return work

    # Skapa snabb index
    idx_map = {t: i for i, t in enumerate(work["Ticker"].tolist()) if t}

    for raw_tkr, fields in updates.items():
        tkr = norm_ticker(raw_tkr)
        ridx = idx_map.get(tkr, None)
        if ridx is None:
            # Ticker finns inte i df => lägg in ny rad
            new_row = {c: "" for c in work.columns}
            new_row["Ticker"] = tkr
            for k, v in (fields or {}).items():
                if k in work.columns:
                    new_row[k] = v
            work.loc[len(work)] = new_row
            idx_map[tkr] = int(len(work) - 1)
        else:
            for k, v in (fields or {}).items():
                if k in work.columns:
                    work.iat[ridx, work.columns.get_loc(k)] = v

    # Sista koll dubbletter
    dups = check_duplicate_tickers(work)
    if dups:
        raise AppUserError(
            f"Uppdateringar skapade dubbletter av Ticker: {', '.join(dups)}."
        )

    return work
