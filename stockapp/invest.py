# -*- coding: utf-8 -*-
"""
Investeringsförslag med bläddringsfunktion (Prev/Next) och sector-aware scoring.
Visar ett bolag åt gången + expander med detaljerade nyckeltal.
"""

from __future__ import annotations
from typing import Dict, Any, List
import math

import numpy as np
import pandas as pd
import streamlit as st


# ---------- scoring-hjälp

DEFAULT_WEIGHTS = {
    # "allmän" baseline
    "P/S": -2.0,                 # lägre bättre (negativ vikt)
    "P/B": -1.0,
    "Debt/Equity": -1.5,
    "Net debt / EBITDA": -1.5,
    "Gross margin (%)": +1.0,
    "Operating margin (%)": +1.5,
    "Net margin (%)": +1.5,
    "FCF Yield (%)": +2.0,
    "Dividend yield (%)": +1.0,
    # bonus för uppsida mot P/S-snitt (om du har en egen beräkning senare)
    "UpsidePS": +1.5,
}

SECTOR_OVERRIDES = {
    "Technology": {
        "Gross margin (%)": +1.5,
        "Operating margin (%)": +2.0,
        "P/S": -2.5,
        "FCF Yield (%)": +1.5,
        "Dividend yield (%)": +0.5,
    },
    "Financial Services": {
        "P/B": -2.0,
        "Debt/Equity": -1.0,
        "Net margin (%)": +1.0,
        "Dividend yield (%)": +1.5,
    },
    "Energy": {
        "Net debt / EBITDA": -2.0,
        "FCF Yield (%)": +2.0,
        "Dividend yield (%)": +1.5,
        "P/S": -1.0,
    },
    "Healthcare": {
        "P/S": -2.0,
        "Gross margin (%)": +1.5,
        "Operating margin (%)": +1.5,
        "Net margin (%)": +1.0,
    },
    "Consumer Defensive": {
        "Dividend yield (%)": +1.8,
        "P/B": -1.5,
        "Operating margin (%)": +1.5,
    },
    "Industrials": {
        "Net debt / EBITDA": -1.8,
        "Operating margin (%)": +1.5,
        "P/S": -1.5,
    },
}


def _metric_score(val: float, inverse: bool = False, lo: float = None, hi: float = None) -> float:
    """
    Normaliserar ett tal till 0..100 grovt.
    - inverse=True: lägre är bättre (t.ex. P/S, D/E, NetDebt/EBITDA)
    - lo/hi kan sätta mjuka gränser för klippning
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    try:
        x = float(val)
    except Exception:
        return 0.0

    if lo is None and hi is None:
        # generiska default
        if inverse:
            # 0 -> 100, 10 -> 0
            x = max(0.0, min(10.0, x))
            return (10.0 - x) * 10.0
        else:
            # 0 -> 0, 30 -> 100
            x = max(0.0, min(30.0, x))
            return x / 30.0 * 100.0

    # Med gränser
    x = max(lo, min(hi, x))
    if inverse:
        return (hi - x) / (hi - lo) * 100.0
    return (x - lo) / (hi - lo) * 100.0


def _sector_weights(sector: str) -> Dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if sector and sector in SECTOR_OVERRIDES:
        w.update(SECTOR_OVERRIDES[sector])
    return w


def _compute_upsides(row: pd.Series) -> float | None:
    """
    Enkel PS-baserad uppsida om du saknar egen riktkursmodell:
    använder P/S-snitt av (P/S, P/S Q1..Q4) om de finns.
    """
    ps_vals = []
    for k in ("P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        v = row.get(k)
       
