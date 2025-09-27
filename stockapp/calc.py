# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .utils import hamta_valutakurs
from .config import TS_FIELDS

def _clamp_cagr(cagr_pct: float) -> float:
    if cagr_pct > 100.0: return 50.0
    if cagr_pct < 0.0: return 2.0
    return float(cagr_pct)

def recompute_row(rad: pd.Series, user_rates: dict) -> dict:
    out = {}

    # P/S-snitt
    ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
    ps_clean = [float(x) for x in ps_vals if float(x) > 0]
    out["P/S-snitt"] = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

    # CAGR clamp
    cagr = float(rad.get("CAGR 5 år (%)", 0.0))
    g = _clamp_cagr(cagr) / 100.0

    # Omsättning 2 & 3 år från "Omsättning nästa år"
    oms_next = float(rad.get("Omsättning nästa år", 0.0))
    if oms_next > 0:
        out["Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
        out["Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
    else:
        out["Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
        out["Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

    # Riktkurser (Revenue anges i miljoner av bolagets valuta; Utestående aktier i miljoner)
    ps_snitt = out["P/S-snitt"]
    shares_m = float(rad.get("Utestående aktier", 0.0))
    if shares_m > 0 and ps_snitt > 0:
        out["Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / shares_m, 2)
        out["Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / shares_m, 2)
        out["Riktkurs om 2 år"] = round((float(out["Omsättning om 2 år"])            * ps_snitt) / shares_m, 2)
        out["Riktkurs om 3 år"] = round((float(out["Omsättning om 3 år"])            * ps_snitt) / shares_m, 2)
    else:
        out["Riktkurs idag"] = out["Riktkurs om 1 år"] = out["Riktkurs om 2 år"] = out["Riktkurs om 3 år"] = 0.0

    # Market cap (valuta) & SEK
    px = float(rad.get("Aktuell kurs", 0.0))
    vx = hamta_valutakurs(rad.get("Valuta","USD"), user_rates)
    if shares_m > 0 and px > 0:
        mcap_val = shares_m * 1e6 * px
        out["Market Cap (valuta)"] = float(mcap_val)
        out["Market Cap (SEK)"] = float(mcap_val * vx)
    else:
        out["Market Cap (valuta)"] = float(rad.get("Market Cap (valuta)", 0.0))
        out["Market Cap (SEK)"] = float(out["Market Cap (valuta)"] * vx)

    # FCF-margin, Monthly burn, Runway
    fcf = float(rad.get("Free Cash Flow", 0.0))
    # omsättning idag i miljoner av prisvaluta => konvertera till samma bas som FCF?
    # yfinance FCF är ofta i basvaluta (USD). Vi lämnar margin tom om saknas TR.
    tr_m = float(rad.get("Omsättning idag", 0.0)) * 1e6  # approx i prisvaluta
    if tr_m > 0:
        out["FCF Margin (%)"] = round((fcf / tr_m) * 100.0, 2)
    else:
        out["FCF Margin (%)"] = float(rad.get("FCF Margin (%)", 0.0))
    monthly_burn = 0.0
    if fcf < 0:
        monthly_burn = abs(fcf) / 12.0
    out["Monthly Burn"] = monthly_burn
    cash = float(rad.get("Cash & Equivalents", 0.0))
    if monthly_burn > 0:
        out["Runway (quarters)"] = round(cash / (monthly_burn * 3.0), 2)
    else:
        out["Runway (quarters)"] = 99.0 if cash > 0 else 0.0

    # Risklabel (baserat på Market Cap (valuta)) — sätts i views/ eller separat funktion om du vill
    return out

def recompute_all(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    work = df.copy()
    for i, row in work.iterrows():
        updates = recompute_row(row, user_rates)
        for k, v in updates.items():
            work.at[i, k] = v
    return work

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, force_stamp_ts: bool = True) -> bool:
    """
    Skriver fälten som kommer i new_vals (även om samma värde) för att kunna stämpla TS.
    Sätter 'Senast auto-uppdaterad' + källa, samt TS_ för spårade fält.
    """
    changed_any = False
    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        prev = df.at[row_idx, f]
        df.at[row_idx, f] = v
        changed_any = changed_any or (str(prev) != str(v))
        # stämpla TS om fältet är TS-tracked och värde skickats
        ts_col = TS_FIELDS.get(f)
        if ts_col and force_stamp_ts:
            from .utils import now_stamp
            try:
                df.at[row_idx, ts_col] = now_stamp()
            except Exception:
                pass
    # notera auto-uppdaterad + källa
    if new_vals:
        from .utils import now_stamp
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    return changed_any or bool(new_vals)
