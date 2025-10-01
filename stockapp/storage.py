# stockapp/storage.py
# -*- coding: utf-8 -*-
"""
Läser/skriv­er portföljdatan mot Google Sheets.
Frikopplad från övriga moduler (inga imports från utils) för att undvika cirkulära beroenden.

Publikt API:
- hamta_data() -> pd.DataFrame
- spara_data(df: pd.DataFrame) -> None
"""

from __future__ import annotations

import time
from typing import List, Any

import pandas as pd
import streamlit as st

from .sheets import get_ws


# ---------------------------------------------------------------------
# Lokal backoff (ingen utils-dependency)
# ---------------------------------------------------------------------
def _with_backoff(fn, *args, **kwargs):
    delay = 0.5
    last_err: Exception | None = None
    for _ in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # vi vill bubbla upp sista felet
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    if last_err:
        raise last_err
    raise RuntimeError("Okänt fel i _with_backoff")


# ---------------------------------------------------------------------
# Läs/skriv
# ---------------------------------------------------------------------
def hamta_data() -> pd.DataFrame:
    """
    Läser alla rader från aktuellt worksheet.
    Förväntar att första raden i arket är kolumnrubriker.
    Returnerar alltid en DataFrame (kan vara tom).
    """
    ws = get_ws()  # använder default från sheets.py (SHEET_NAME i secrets)
    # get_all_records läser med rubrikrad som header och resten som data
    rows: List[dict[str, Any]] = _with_backoff(ws.get_all_records, empty_value="")
    if not rows:
        # Om helt tomt (eller bara header i arket), försök hämta header separat
        try:
            header = _with_backoff(ws.row_values, 1)
        except Exception:
            header = []
        if header:
            return pd.DataFrame(columns=header)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalisera kolumnnamn (trimma whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def spara_data(df: pd.DataFrame) -> None:
    """
    Skriver hela DataFrame till bladet.
    - Rensar arket
    - Skriver header + värden
    Bevarar kolumnordningen enligt df.columns.
    """
    if df is None:
        return

    ws = get_ws()
    # Säkerställ strängrubriker
    headers = [str(c) for c in df.columns.tolist()]

    # Konvertera DataFrame till listor
    if len(df) > 0:
        values = df.astype(object).where(pd.notna(df), "").values.tolist()
    else:
        values = []

    body: List[List[Any]] = [headers] + values

    # Skriv med backoff (clear + update)
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)
