# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .sheets import get_ws
from .config import FINAL_COLS, SHEET_NAME
from .utils import ensure_schema, with_backoff, dedupe_tickers


TEXT_LIKE = {"Ticker", "Bolagsnamn", "Valuta", "Sektor", "Risklabel"}


def _find_header_row(rows: List[List[str]]) -> int:
    """
    Hitta rubrikraden. Vi letar efter en rad som innehåller 'Ticker' (case-insensitive).
    Returnerar radindex (0-baserat). Om ingen hittas: 0.
    """
    for i, r in enumerate(rows):
        joined = " ".join([str(x) for x in r]).lower()
        if "ticker" in joined:
            return i
    return 0


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Försöker konvertera numeriska kolumner (ej text & ej TS-kolumner) till float."""
    for col in df.columns:
        if col in TEXT_LIKE or col.endswith(" TS"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def hamta_data() -> pd.DataFrame:
    """
    Läser HE-LA arket robust:
      - Hämtar alla värden
      - Hittar rubrikrad ('Ticker')
      - Bygger DataFrame och fyller på saknade FINAL_COLS
      - Konverterar numeriska kolumner
      - Tar bort uppenbart tomma rader (saknar Ticker)
    """
    ws = get_ws(SHEET_NAME)
    values: List[List[str]] = with_backoff(ws.get_all_values)

    if not values:
        # Tomt ark => returnera tomt med korrekt schema
        return pd.DataFrame(columns=FINAL_COLS)

    header_idx = _find_header_row(values)
    headers = [str(h).strip() for h in values[header_idx]]
    body = values[header_idx + 1 :]

    # Trimma body till header-längd
    trimmed = [row[: len(headers)] for row in body]

    df = pd.DataFrame(trimmed, columns=headers)
    # Rensa tomma strängar
    df = df.replace({"": np.nan})

    # Ta bort helt tomma rader
    if "Ticker" in df.columns:
        df = df[~df["Ticker"].isna() & (df["Ticker"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    # Säkerställ schema
    df = ensure_schema(df, FINAL_COLS)

    # Datatyper
    df = _coerce_types(df)

    # Dubblettskydd i minnet (appens spar-logik får bestämma senare om sammanfogning)
    df, _ = dedupe_tickers(df)

    return df


def spara_data(df: pd.DataFrame, do_snapshot: bool = False) -> None:
    """
    Skriver tillbaka hela tabellen:
      - Kolumnordning enligt FINAL_COLS
      - Tomma celler -> "" (inte NaN)
    """
    ws = get_ws(SHEET_NAME)
    # Se till att kolumner finns i rätt ordning
    out = ensure_schema(df.copy(), FINAL_COLS)
    out = out[FINAL_COLS]

    # Gör om NaN -> ""
    out = out.replace({np.nan: ""})
    values = [list(out.columns)]
    values += out.astype(str).values.tolist()

    with_backoff(ws.clear)
    with_backoff(ws.update, values)
