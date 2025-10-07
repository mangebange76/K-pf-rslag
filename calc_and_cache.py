# calc_and_cache.py
import streamlit as st
import pandas as pd
import numpy as np

# —— Kärnberäkningar (P/S-snitt, omsättning år2/3 med clamp, riktkurser)
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict | None = None) -> pd.DataFrame:
    """
    Uppdaterar derivatkolumner i datatabellen:
      - 'P/S-snitt' = snitt av positiva {Q1..Q4}
      - 'Omsättning om 2 år', 'Omsättning om 3 år' = växer från 'Omsättning nästa år' med CAGR-clamp
      - 'Riktkurs {idag,1,2,3 år}' = (Omsättning * P/S-snitt) / Utestående aktier
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    for i, rad in df.iterrows():
        # 1) P/S-snitt av positiva kvartal
        ps_vals = [
            float(rad.get("P/S Q1", 0) or 0),
            float(rad.get("P/S Q2", 0) or 0),
            float(rad.get("P/S Q3", 0) or 0),
            float(rad.get("P/S Q4", 0) or 0),
        ]
        ps_clean = [v for v in ps_vals if v > 0]
        df.at[i, "P/S-snitt"] = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

        # 2) CAGR clamp: >100% → 50%, <0% → 2%
        cagr = float(rad.get("CAGR 5 år (%)", 0.0) or 0.0)
        if cagr > 100.0:
            just_cagr = 50.0
        elif cagr < 0.0:
            just_cagr = 2.0
        else:
            just_cagr = cagr
        g = just_cagr / 100.0

        # 3) Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev. manuellt ifyllda
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0) or 0.0)
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0) or 0.0)

        # 4) Riktkurser
        ps_snitt = float(df.at[i, "P/S-snitt"])
        aktier_ut = float(rad.get("Utestående aktier", 0.0) or 0.0)  # i miljoner
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0) or 0.0)     * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0) or 0.0) * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])             * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])             * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df


# —— Cache för investeringsförslag (dyr del som används vid bläddring i UI)
@st.cache_data(show_spinner=False, ttl=300)
def bygg_forslag_cache(df_json: str, riktkurs_val: str, subset: str, kapital_sek: float) -> pd.DataFrame:
    """
    Returnerar en förberäknad DataFrame med de kolumner som behövs i investeringsvyn:
      ["Ticker","Bolagsnamn","Aktuell kurs","Valuta", riktkurs_val, "Potential (%)","Diff till mål (%)"]
    Cachas i 5 minuter för att minimera lagg vid UI-bläddring.
    """
    if not df_json:
        return pd.DataFrame()

    df = pd.read_json(df_json, orient="split")

    # Filtrera subset
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0]

    # Nödvändiga villkor
    if riktkurs_val not in base.columns:
        return pd.DataFrame()

    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        return base

    # Nyckelmetrik
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    # Endast kolumner som UI:t behöver
    cols = ["Ticker","Bolagsnamn","Aktuell kurs","Valuta", riktkurs_val, "Potential (%)","Diff till mål (%)"]
    return base[cols].reset_index(drop=True)
