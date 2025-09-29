# stockapp/compute.py
# -*- coding: utf-8 -*-
"""
Beräknings- och inmatningshjälpare för DataFrame:
- apply_auto_updates_to_row(df, ridx, vals, source, changes_map, always_stamp)
- uppdatera_berakningar(df, user_rates)
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime as _dt

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Små hjälpare
# ------------------------------------------------------------
def _now_ts() -> str:
    return _dt.now().isoformat(timespec="seconds")

def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except Exception:
        return False

def _to_float(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return float(default)
    try:
        s = str(x).strip().replace(" ", "").replace("\u00A0", "")
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return float(default)

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

def _fmt_thousands(n: float) -> str:
    try:
        i = int(round(float(n)))
        return f"{i:,}".replace(",", " ")
    except Exception:
        return str(n)

def _format_mcap_sv(n: float) -> str:
    """
    Formatera marknadsvärde i svenska enheter:
    - tn (triljoner) >= 1e12
    - mdr (miljarder) >= 1e9
    - mn (miljoner) >= 1e6
    annars tusentals-sep
    """
    try:
        v = float(n)
    except Exception:
        return ""

    sign = "-" if v < 0 else ""
    v = abs(v)

    if v >= 1e12:
        return f"{sign}{v/1e12:.2f} tn"
    if v >= 1e9:
        return f"{sign}{v/1e9:.2f} mdr"
    if v >= 1e6:
        return f"{sign}{v/1e6:.2f} mn"
    return f"{sign}{_fmt_thousands(v)}"

def _fx_to_sek(cur: str, rates: Dict[str, float]) -> float:
    """
    Returnerar kurs (cur -> SEK) från user_rates; fallback 1.0 om okänd.
    """
    if not cur:
        return 1.0
    try:
        return float(rates.get(cur.upper(), 1.0))
    except Exception:
        return 1.0

def _fx_cur_to_usd(cur: str, rates: Dict[str, float]) -> float:
    """
    Omvandla 1 enhet 'cur' till USD baserat på kurserna valuta->SEK i 'rates'.
    cur_to_usd = (cur->SEK) / (USD->SEK).
    """
    cur = (cur or "USD").upper()
    usd_sek = _fx_to_sek("USD", rates)
    cur_sek = _fx_to_sek(cur, rates)
    if usd_sek <= 0:
        return 1.0
    return cur_sek / usd_sek

# ------------------------------------------------------------
# Inmatning/TS/Källa
# ------------------------------------------------------------
def apply_auto_updates_to_row(
    df: pd.DataFrame,
    ridx: int,
    vals: Dict[str, Any],
    source: str = "Auto",
    changes_map: Optional[Dict[str, List[str]]] = None,
    always_stamp: bool = True,
) -> List[str]:
    """
    Skriver in fält på rad 'ridx'. Skapar saknade kolumner.
    För varje uppdaterat fält 'X' sätts:
      - kolumn 'X'
      - kolumn 'X [TS]'  -> current ISO-datetime
      - kolumn 'X [Källa]' -> source
    Om 'always_stamp' = True stämplas TS/Källa även om värdet är oförändrat.

    Returnerar lista över fältnamn som faktiskt ändrades (värdet nytt vs gammalt).
    Loggar ändringarna i 'changes_map' per ticker om den skickas in.
    """
    changed: List[str] = []
    if not isinstance(df, pd.DataFrame) or ridx not in df.index:
        return changed

    # ticker behövs för logg
    tkr = ""
    if "Ticker" in df.columns:
        try:
            tkr = str(df.loc[ridx, "Ticker"])
        except Exception:
            tkr = ""

    for col, new_val in vals.items():
        ts_col = f"{col} [TS]"
        src_col = f"{col} [Källa]"
        _ensure_columns(df, [col, ts_col, src_col])

        old_val = df.loc[ridx, col]
        is_num = isinstance(new_val, (int, float, np.number)) or (
            isinstance(new_val, str) and new_val.replace(",", ".").replace(" ", "").replace("\u00A0","").replace("-", "", 1).replace(".", "", 1).isdigit()
        )

        # normalisera numeriskt
        if is_num:
            new_val_norm = _to_float(new_val, default=np.nan)
            old_val_norm = _to_float(old_val, default=np.nan)
            different = (np.isnan(old_val_norm) and not np.isnan(new_val_norm)) or \
                        (not np.isnan(old_val_norm) and np.isnan(new_val_norm)) or \
                        (not np.isnan(old_val_norm) and not np.isnan(new_val_norm) and abs(old_val_norm - new_val_norm) > 1e-12)
        else:
            new_val_norm = new_val
            different = (str(old_val) != str(new_val))

        if different:
            df.loc[ridx, col] = new_val_norm
            changed.append(col)

        # Stämpla alltid om begärt
        if different or always_stamp:
            df.loc[ridx, ts_col] = _now_ts()
            df.loc[ridx, src_col] = source

    # Logg
    if changes_map is not None and tkr:
        changes_map.setdefault(tkr, []).extend(changed)

    return changed

# ------------------------------------------------------------
# Beräkningar / härledda kolumner
# ------------------------------------------------------------
def _calc_ps_avg_4q(row: pd.Series) -> float:
    """
    Medelvärde av P/S Q1..Q4 om de finns; annars fallback: P/S.
    """
    vals = []
    for k in ("P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        if k in row and not _is_nan(row[k]) and _to_float(row[k]) > 0:
            vals.append(_to_float(row[k]))
    if vals:
        return float(np.mean(vals))
    # fallback
    if "P/S" in row and not _is_nan(row["P/S"]) and _to_float(row["P/S"]) > 0:
        return _to_float(row["P/S"])
    return np.nan

def _pick_current_revenue_mn(row: pd.Series) -> float:
    """
    Välj manuell prognos i första hand:
    - 'Omsättning i år (förv.)' (miljoner)
    fallback: 'Omsättning idag' (miljoner)
    """
    for k in ("Omsättning i år (förv.)", "Omsättning idag"):
        if k in row and not _is_nan(row[k]) and _to_float(row[k]) > 0:
            return _to_float(row[k])
    return np.nan

def _shares_count(row: pd.Series) -> float:
    """
    Antal aktier (stycken). I DF lagras 'Utestående aktier' i **miljoner**.
    """
    if "Utestående aktier" in row and not _is_nan(row["Utestående aktier"]):
        m = _to_float(row["Utestående aktier"])
        if m > 0:
            return m * 1e6
    return np.nan

def _cur(row: pd.Series) -> str:
    v = str(row.get("Valuta", "USD") or "USD").upper()
    return v if v in ("USD","EUR","CAD","NOK","SEK","GBP","DKK","CHF","JPY") else v

def _calc_target_price(row: pd.Series, user_rates: Dict[str, float]) -> Tuple[float, float]:
    """
    Beräknar riktkurs i bolagets valuta och i SEK.
    target_mcap = (förv. omsättning i HELA VALUTAN) * P/S-snitt
                 = (mn * 1e6) * ps_avg
    target_price = target_mcap / shares
    """
    ps_avg = _to_float(row.get("P/S snitt (4Q)", np.nan), default=np.nan)
    rev_mn = _to_float(_pick_current_revenue_mn(row), default=np.nan)
    shares = _to_float(_shares_count(row), default=np.nan)
    cur = _cur(row)

    if np.isnan(ps_avg) or ps_avg <= 0 or np.isnan(rev_mn) or rev_mn <= 0 or np.isnan(shares) or shares <= 0:
        return (np.nan, np.nan)

    target_mcap_cur = rev_mn * 1e6 * ps_avg  # i bolagets valuta
    target_px_cur = target_mcap_cur / shares

    # Till SEK
    cur_to_sek = _fx_to_sek(cur, user_rates)
    target_px_sek = target_px_cur * cur_to_sek

    return (float(target_px_cur), float(target_px_sek))

def _calc_uppsida(row: pd.Series) -> float:
    px = _to_float(row.get("Aktuell kurs", np.nan), default=np.nan)
    targ = _to_float(row.get("Riktkurs (valuta)", np.nan), default=np.nan)
    if np.isnan(px) or px <= 0 or np.isnan(targ) or targ <= 0:
        return np.nan
    return (targ / px - 1.0) * 100.0

def _calc_risklabel(row: pd.Series, user_rates: Dict[str, float]) -> str:
    """
    Risk-etikett baserat på MCAP i USD.
    Gränser:
      Micro < 300M
      Small 300M–2B
      Mid   2B–10B
      Large ≥ 10B
    """
    mcap = _to_float(row.get("MCAP nu", np.nan), default=np.nan)
    if np.isnan(mcap) or mcap <= 0:
        return ""
    cur = _cur(row)
    cur_to_usd = _fx_cur_to_usd(cur, user_rates)
    mcap_usd = mcap * cur_to_usd

    if mcap_usd < 3e8:
        return "Microcap"
    if mcap_usd < 2e9:
        return "Smallcap"
    if mcap_usd < 1e10:
        return "Midcap"
    return "Largecap"

# ------------------------------------------------------------
# Publik: uppdatera_berakningar
# ------------------------------------------------------------
def uppdatera_berakningar(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Räknar härledda fält på hela DF:t. Skapar kolumner om de saknas.

    Skapar/uppdaterar:
      - P/S snitt (4Q)
      - Riktkurs (valuta)
      - Riktkurs (SEK)
      - Uppsida (%)
      - MCAP (fmt)
      - Risklabel

    Antaganden:
      - 'Utestående aktier' är lagrat i miljoner
      - 'Omsättning i år (förv.)' resp. 'Omsättning idag' i **miljoner** av bolagets valuta
      - 'MCAP nu' i bolagets valuta
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else df

    work = df.copy()

    # Säkerställ kolumner
    need_cols = [
        "P/S snitt (4Q)",
        "Riktkurs (valuta)",
        "Riktkurs (SEK)",
        "Uppsida (%)",
        "MCAP (fmt)",
        "Risklabel",
    ]
    _ensure_columns(work, need_cols)

    # P/S snitt (4Q)
    work["P/S snitt (4Q)"] = work.apply(_calc_ps_avg_4q, axis=1)

    # Riktkurser
    targ_val = []
    targ_sek = []
    for _, row in work.iterrows():
        tv, ts = _calc_target_price(row, user_rates)
        targ_val.append(tv)
        targ_sek.append(ts)
    work["Riktkurs (valuta)"] = targ_val
    work["Riktkurs (SEK)"] = targ_sek

    # Uppsida
    work["Uppsida (%)"] = work.apply(_calc_uppsida, axis=1)

    # MCAP (fmt)
    if "MCAP nu" in work.columns:
        work["MCAP (fmt)"] = work["MCAP nu"].apply(_format_mcap_sv)
    else:
        work["MCAP (fmt)"] = ""

    # Risklabel
    work["Risklabel"] = work.apply(lambda r: _calc_risklabel(r, user_rates), axis=1)

    return work
