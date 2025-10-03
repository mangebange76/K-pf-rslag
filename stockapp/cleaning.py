# stockapp/cleaning.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Tuple, List, Set

# Kolumner vi ALDRIG tar bort (appens kärna och strategier)
PROTECTED_COLS: Set[str] = {
    "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Senast manuellt uppdaterad",
    "Utestående aktier (milj.)","Market Cap",
    # Tillväxt (P/S)
    "P/S (TTM)","Revenue TTM (M)","Revenue growth (%)","P/S-snitt",
    # Utdelning
    "Årlig utdelning","Dividend yield (%)","Payout ratio (%)",
    # EV/EBITDA
    "EV/EBITDA (ttm)","_y_ev_now","_y_ebitda_now",
    # P/B-strategier
    "P/B","Book value / share","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
    # Kompatibilitet från äldre modell
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år","Antal aktier","CAGR 5 år (%)",
}

def _is_numeric(series: pd.Series) -> bool:
    try:
        pd.to_numeric(series, errors="raise")
        return True
    except Exception:
        return False

def _nonempty_mask(series: pd.Series) -> pd.Series:
    """True där värdet räknas som 'fyllt'."""
    if _is_numeric(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return vals != 0.0
    s = series.astype(str).str.strip()
    return s != ""

def column_coverage(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    mask = _nonempty_mask(df[col])
    return float(mask.mean())

def prune_columns(
    df: pd.DataFrame,
    min_coverage: float = 0.05,
    extra_protected: Iterable[str] = (),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ta bort kolumner med täckning < min_coverage (andel icke-tomma värden).
    Skyddar PROTECTED_COLS + extra_protected.
    Returnerar (ny_df, borttagna_kolumner)
    """
    if df is None or df.empty:
        return df.copy(), []

    protected = set(PROTECTED_COLS) | set(extra_protected)
    drop_cols: List[str] = []
    for c in df.columns:
        if c in protected:
            continue
        cov = column_coverage(df, c)
        if cov < float(min_coverage):
            drop_cols.append(c)

    cleaned = df.drop(columns=drop_cols, errors="ignore").copy()
    return cleaned, drop_cols

def smart_dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Slår ihop dubbletter av Ticker.
    - Ticker normaliseras (upper/trim)
    - För numeriska fält: behåll sista icke-nollan (annars första)
    - För textfält: behåll sista icke-tomma (annars första)
    Returnerar (ny_df, antal_sammanslagningar)
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        return df.copy(), 0

    tmp = df.copy()
    tmp["Ticker"] = tmp["Ticker"].astype(str).str.upper().str.strip()

    # Inga dubbletter → klart
    if tmp["Ticker"].nunique() == len(tmp):
        return tmp, 0

    # Förbered typkarta
    numeric_cols = []
    for c in tmp.columns:
        if c == "Ticker": continue
        if _is_numeric(tmp[c]):
            numeric_cols.append(c)

    def merge_group(g: pd.DataFrame) -> pd.Series:
        out = {}
        for c in g.columns:
            if c == "Ticker":
                out[c] = g[c].iloc[-1]
                continue
            s = g[c]
            if c in numeric_cols:
                s_num = pd.to_numeric(s, errors="coerce").fillna(0.0)
                nz = s_num[s_num != 0.0]
                out[c] = nz.iloc[-1] if not nz.empty else s_num.iloc[0]
            else:
                s_str = s.astype(str).str.strip()
                nz = s_str[s_str != ""]
                out[c] = nz.iloc[-1] if not nz.empty else s_str.iloc[0]
        return pd.Series(out)

    merged = tmp.groupby("Ticker", as_index=False).apply(merge_group, include_groups=False)
    merged = merged.reset_index(drop=True)
    merges = len(tmp) - len(merged)
    return merged, merges
