# stockapp/schema.py
from __future__ import annotations
from typing import List, Set, Dict, Any
import pandas as pd
import streamlit as st

# De enda kolumner appen ska ha (ordning = sanning)
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

# Dessa ska vara numeriska (allt annat sträng)
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

# Standardvärden
def _default_for(col: str) -> Any:
    if col == "Valuta":
        return "USD"  # default
    if col in NUMERIC_COLS:
        return 0.0
    return ""

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # Rensa bort tusental/komma, tolka robust till float
    ser = s.astype(str)\
            .str.replace("\u00a0"," ", regex=False)\
            .str.replace("%","", regex=False)\
            .str.replace(",","_", regex=False)\
            .str.replace(".","", regex=False)\
            .str.replace("_",".", regex=False)\
            .str.strip()
    return pd.to_numeric(ser, errors="coerce").fillna(0.0)

def _as_canon(df: pd.DataFrame) -> pd.DataFrame:
    # Skapa nytt DF med exakt rätt kolumner och datatyper
    n = len(df)
    out = pd.DataFrame({c: [_default_for(c)]*n for c in CANON_COLS})
    # Kopiera över de få fält som ev. redan finns (exakt namn)
    for c in CANON_COLS:
        if c in df.columns:
            if c in NUMERIC_COLS:
                out[c] = _coerce_numeric_series(df[c])
            else:
                out[c] = df[c].astype(str)
    return out

def enforce_schema(raw_df: pd.DataFrame, ws_title: str, *, write_back: bool = True) -> pd.DataFrame:
    """
    * Kräver minst kolumnen 'Ticker'. Allt annat ignoreras och/eller byggs upp.
    * Returnerar DataFrame exakt enligt CANON_COLS.
    * Skriver tillbaka till Google Sheets EN gång om kolumnordning/namn inte matchar.
    """
    from .sheets import ws_write_df  # lokal import för att undvika cirkulärt
    # 1) Säkerställ att Ticker finns
    cols = [str(c).strip() for c in list(raw_df.columns)]
    if "Ticker" not in cols:
        # Om användaren råkat lämna enda kolumnen utan namn, försök anta att första är tickers
        if len(cols) == 1 and cols[0] != "Ticker":
            df_tmp = raw_df.rename(columns={cols[0]: "Ticker"}).copy()
        else:
            # Skapa tom mall
            df_tmp = pd.DataFrame({"Ticker": []})
    else:
        df_tmp = raw_df.copy()

    # 2) Endast Ticker + ev. riktiga canon-fält kopieras – allt annat ignoreras
    keep = [c for c in df_tmp.columns if c in CANON_COLS]
    df_keep = df_tmp[keep].copy()

    # 3) Bygg exakt rätt schema
    df_out = _as_canon(df_keep)

    # 4) Skriv tillbaka om nödvändigt (rubrikordningen måste vara exakt CANON_COLS)
    need_write = list(df_tmp.columns) != CANON_COLS
    key = f"_schema_applied__{ws_title}"
    already = st.session_state.get(key, False)

    if write_back and need_write and not already:
        try:
            ws_write_df(ws_title, df_out)  # skriver rubriker + data exakt i ordning
            st.session_state[key] = True
            st.sidebar.success(f"Byggde upp bladet '{ws_title}' med kanoniska kolumner.")
        except Exception as e:
            st.sidebar.warning(f"Kunde inte skriva upp strukturen: {e}")

    return df_out
