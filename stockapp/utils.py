# stockapp/utils.py
# -*- coding: utf-8 -*-

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Tidsstämplade fält (kolumn -> TS-kolumn)
# ------------------------------------------------------------
TS_FIELDS: Dict[str, str] = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

# ------------------------------------------------------------
# Slutlig kolumnlista (minimikrav för appen)
# Obs: fler kolumner kan finnas i databasen; appen klarar det.
# ------------------------------------------------------------
FINAL_COLS: List[str] = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Portfölj-relaterat
    "GAV (SEK)",

    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# ------------------------------------------------------------
# Datum-hjälpare
# ------------------------------------------------------------
def _today_str() -> str:
    # Appen kör i Stockholm; förenklad lokal tid utan pytz (ok för stämplar YYYY-MM-DD)
    return datetime.now().strftime("%Y-%m-%d")


# ------------------------------------------------------------
# Schemahjälpare
# ------------------------------------------------------------
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att minimikolumner finns och fyll med rimliga defaultvärden.
    Tar bort dubblett-kolumner och returnerar en kopia.
    """
    df = df.copy()

    for kol in FINAL_COLS:
        if kol not in df.columns:
            # Numeriska default?
            if any(x in kol.lower() for x in [
                "kurs", "omsättning", "p/s", "utdelning", "cagr", "antal", "riktkurs", "aktier", "snitt", "gav", "mcap"
            ]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa"):
                df[kol] = ""
            else:
                df[kol] = ""

    # Ta bort dubbletter och behåll första
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Migrera ev. äldre namn på riktkurskolumner till de nya.
    """
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
    """
    Säkerställ numeriska/sträng-typer enligt förväntan.
    """
    df = df.copy()

    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "GAV (SEK)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)

    return df

# ------------------------------------------------------------
# Tidsstämplar – äldsta TS över spårade fält
# ------------------------------------------------------------
def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    Returnerar äldsta (minsta) tidsstämpeln bland alla TS_-kolumner för en rad.
    None om inga tidsstämplar finns eller ej parsbara.
    """
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None


def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lägger till hjälpkolumner:
      - _oldest_any_ts (Timestamp eller NaT)
      - _oldest_any_ts_fill (för sortering: NaT ersätts med framtidsdatum)
    """
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# ------------------------------------------------------------
# Beräkningar (P/S-snitt, riktkurser, framtida omsättning)
# ------------------------------------------------------------
def uppdatera_berakningar(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning om 2 & 3 år från 'Omsättning nästa år' med CAGR clamp (>100% → 50%, <0% → 2%)
      - Riktkurser (idag/1/2/3 år) utifrån P/S-snitt och Utestående aktier
    Not: user_rates tas med för signaturkompatibilitet, används ej här.
    """
    df = df.copy()
    for i, rad in df.iterrows():
        # P/S-snitt – använd endast positiva
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        try:
            ps_clean = [float(x) for x in ps_vals if pd.notna(x) and float(x) > 0]
        except Exception:
            ps_clean = []
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        try:
            cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        except Exception:
            cagr = 0.0
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år
        try:
            oms_next = float(rad.get("Omsättning nästa år", 0.0))
        except Exception:
            oms_next = 0.0
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll befintliga om finns, annars 0
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0) or 0.0)
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0) or 0.0)

        # Riktkurser – kräver Utestående aktier (miljoner) och P/S-snitt
        try:
            aktier_ut_milj = float(rad.get("Utestående aktier", 0.0))
        except Exception:
            aktier_ut_milj = 0.0

        if aktier_ut_milj > 0 and ps_snitt > 0:
            denom = aktier_ut_milj  # omsättningar antas vara i "miljoner" i databasen
            def _rk(val: float) -> float:
                try:
                    v = float(val)
                    return round((v * ps_snitt) / denom, 2) if v > 0 else 0.0
                except Exception:
                    return 0.0

            df.at[i, "Riktkurs idag"]    = _rk(rad.get("Omsättning idag", 0.0))
            df.at[i, "Riktkurs om 1 år"] = _rk(rad.get("Omsättning nästa år", 0.0))
            df.at[i, "Riktkurs om 2 år"] = _rk(df.at[i, "Omsättning om 2 år"])
            df.at[i, "Riktkurs om 3 år"] = _rk(df.at[i, "Omsättning om 3 år"])
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df
