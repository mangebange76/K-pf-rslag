from __future__ import annotations
from typing import List, Set, Any
import pandas as pd
import streamlit as st

CANON_COLS: List[str] = [
    "Ticker","Bolagsnamn","Sektor","Valuta",
    "Antal aktier","GAV (SEK)","Aktuell kurs","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
    "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
    "Senast manuellt uppdaterad","Senast auto uppdaterad","Auto källa","Senast beräknad",
    "DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
    "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",
    "Score Total (Idag)","Score Total (1 år)","Score Total (2 år)","Score Total (3 år)",
    "Score Growth (Idag)","Score Dividend (Idag)","Score Financials (Idag)",
    "Score Growth (1 år)","Score Dividend (1 år)","Score Financials (1 år)",
    "Score Growth (2 år)","Score Dividend (2 år)","Score Financials (2 år)",
    "Score Growth (3 år)","Score Dividend (3 år)","Score Financials (3 år)",
    "Div_Frekvens/år","Div_Månader","Div_Vikter","Uppsida (%)",
]

NUMERIC_COLS: Set[str] = {
    "Antal aktier","GAV (SEK)","Aktuell kurs","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
    "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
    "DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
    "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",
    "Score Total (Idag)","Score Total (1 år)","Score Total (2 år)","Score Total (3 år)",
    "Score Growth (Idag)","Score Dividend (Idag)","Score Financials (Idag)",
    "Score Growth (1 år)","Score Dividend (1 år)","Score Financials (1 år)",
    "Score Growth (2 år)","Score Dividend (2 år)","Score Financials (2 år)",
    "Score Growth (3 år)","Score Dividend (3 år)","Score Financials (3 år)",
    "Div_Frekvens/år","Uppsida (%)",
}

def _default_for(col: str) -> Any:
    if col == "Valuta": return "USD"
    return 0.0 if col in NUMERIC_COLS else ""

def _coerce_num(s: pd.Series) -> pd.Series:
    ser = s.astype(str)\
            .str.replace("\u00a0"," ", regex=False)\
            .str.replace("%","", regex=False)\
            .str.replace(",", "_", regex=False)\
            .str.replace(".", "", regex=False)\
            .str.replace("_",".", regex=False)\
            .str.strip()
    return pd.to_numeric(ser, errors="coerce").fillna(0.0)

def _as_canon(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame({c: [_default_for(c)]*n for c in CANON_COLS})
    for c in CANON_COLS:
        if c in df.columns:
            out[c] = _coerce_num(df[c]) if c in NUMERIC_COLS else df[c].astype(str)
    return out

def enforce_schema(raw_df: pd.DataFrame, ws_title: str, write_back: bool = True) -> pd.DataFrame:
    from .sheets import ws_write_df

    cols = [str(c).strip() for c in list(raw_df.columns)]
    if "Ticker" not in cols:
        if len(cols) == 1:
            raw_df = raw_df.rename(columns={cols[0]: "Ticker"})
        else:
            raw_df = pd.DataFrame({"Ticker": []})
    keep = [c for c in raw_df.columns if c in CANON_COLS]
    df_keep = raw_df[keep].copy()

    df_out = _as_canon(df_keep)

    need_write = list(raw_df.columns) != CANON_COLS
    key = f"_schema_applied__{ws_title}"
    if write_back and need_write and not st.session_state.get(key, False):
        try:
            ws_write_df(ws_title, df_out)
            st.session_state[key] = True
            st.sidebar.success(f"Struktur byggd i '{ws_title}'.")
        except Exception as e:
            st.sidebar.warning(f"Kunde inte skriva struktur: {e}")
    return df_out
