# schema_utils.py
import streamlit as st
import pandas as pd
import numpy as np

FINAL_COLS = [
    "Ticker","Bolagsnamn","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Valuta","Årlig utdelning","Aktuell kurs",
    "CAGR 5 år (%)","P/S-snitt",
    "Senast manuellt uppdaterad","Senast auto uppdaterad",
    "TS P/S","TS Omsättning","TS Utestående aktier",
    "Källa Aktuell kurs","Källa Utestående aktier",
    "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","snitt","utestående"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Riktkurs 2026":"Riktkurs om 1 år",
        "Riktkurs 2027":"Riktkurs om 2 år",
        "Riktkurs 2028":"Riktkurs om 3 år",
        "Riktkurs om idag":"Riktkurs idag"
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns: df[new] = 0.0
            nv = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            ov = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (nv == 0.0) & (ov > 0.0)
            df.loc[mask, new] = ov[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    str_cols = [
        "Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad",
        "TS P/S","TS Omsättning","TS Utestående aktier",
        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
        "Källa Aktuell kurs","Källa Utestående aktier",
        "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"
    ]
    for c in str_cols:
        if c in df.columns: df[c] = df[c].astype(str)
    return df
