# app.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from sheets_utils import (
    hamta_data,
    spara_data,
    skapa_snapshot_om_saknas,
)

from views import (
    hamta_valutakurser_sidebar,
    visa_hamtlogg_panel,
    spara_logg_till_sheets,
    massuppdatera,
    lagg_till_eller_uppdatera,
    analysvy,
    visa_investeringsforslag,
)

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------- Kolumnschema ----------
FINAL_COLS = [
    # Bas
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning",
    "Utestående aktier", "Antal aktier",
    # P/S & kvartal
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S Q1 datum", "P/S Q2 datum", "P/S Q3 datum", "P/S Q4 datum",
    "Källa Aktuell kurs", "Källa Utestående aktier", "Källa P/S",
    "Källa P/S Q1", "Källa P/S Q2", "Källa P/S Q3", "Källa P/S Q4",
    # Omsättning & riktkurser (miljoner och per aktie)
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    # Derivat/övrigt
    "CAGR 5 år (%)", "P/S-snitt",
    # Tidsstämplar & meta
    "Senast manuellt uppdaterad", "Senast auto uppdaterad",
    "TS P/S", "TS Utestående aktier", "TS Omsättning",
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "Riktkurs 2026": "Riktkurs om 1 år",
        "Riktkurs 2027": "Riktkurs om 2 år",
        "Riktkurs 2028": "Riktkurs om 3 år",
        "Riktkurs om idag": "Riktkurs idag",
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns:
                df[new] = 0.0
            new_vals = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            old_vals = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = [
        "Utestående aktier","Antal aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S-snitt",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in [
        "Ticker","Bolagsnamn","Valuta",
        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
        "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
        "Senast manuellt uppdaterad","Senast auto uppdaterad",
        "TS P/S","TS Utestående aktier","TS Omsättning",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# Minimal beräkningsfallback (om du inte har en egen calc-modul)
def _uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, rad in df.iterrows():
        # P/S-snitt (ignorera nollor)
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(float(np.mean(ps_clean)) if ps_clean else 0.0, 2)
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från “nästa år”
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)

        # Riktkurser — OBS: omsättning & utest. aktier i “miljoner”
        ps_use = ps_snitt if ps_snitt > 0 else float(rad.get("P/S", 0.0))
        aktier_ut_mn = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut_mn > 0 and ps_use > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))     * ps_use) / aktier_ut_mn, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * ps_use) / aktier_ut_mn, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])      * ps_use) / aktier_ut_mn, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])      * ps_use) / aktier_ut_mn, 2)
        else:
            for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i, k] = 0.0
    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: Valutor, Sheets-status & hämtlogg
    user_rates = hamta_valutakurser_sidebar()
    visa_hamtlogg_panel()
    if st.sidebar.button("⬆️ Spara hämtlogg"):
        spara_logg_till_sheets()

    # Läs data (med automatisk lokal fallback om Sheets inte nås)
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # Säkerställ schema och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 📸 Snapshot vid start (en gång per session/dag i utility-modulen)
    if "did_bootstrap_snapshot" not in st.session_state:
        ok, msg = skapa_snapshot_om_saknas(df)
        st.session_state["did_bootstrap_snapshot"] = True
        st.sidebar.caption(msg)

    # Global massuppdateringsknapp (Yahoo/SEC)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag"])

    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        # lätta derivat lokalt för visning (sparas ej här)
        df = _uppdatera_berakningar(df)
        visa_investeringsforslag(df, user_rates)

if __name__ == "__main__":
    main()
