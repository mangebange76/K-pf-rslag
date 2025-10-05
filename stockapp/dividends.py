# stockapp/dividends.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Hjälpare: månader (sv/eng) och parsning
# ------------------------------------------------------------
_SV_MN = ["jan","feb","mar","apr","maj","jun","jul","aug","sep","okt","nov","dec"]
_EN_MN = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
_SV_FULL = ["januari","februari","mars","april","maj","juni","juli","augusti","september","oktober","november","december"]

def _to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(",", ".")
        if s == "" or s.lower() in {"na","n/a","none","null","—"}:
            return 0.0
        return float(s)
    except Exception:
        return 0.0

def _month_name_sv(m: int) -> str:
    if 1 <= m <= 12:
        return _SV_FULL[m-1]
    return ""

def _parse_month_token(tok: str) -> int:
    """
    '3' -> 3
    'Mar'/'mar' -> 3
    'Mär' (ovanligt) -> ignoreras → 0
    'okt' -> 10, 'dec' -> 12
    """
    if not tok:
        return 0
    s = tok.strip().lower()
    s = re.sub(r"[^\w]", "", s)  # ta bort skiljetecken
    # siffra?
    if re.match(r"^\d{1,2}$", s):
        m = int(s)
        return m if 1 <= m <= 12 else 0
    # sv/eng kort
    if s in _SV_MN:
        return _SV_MN.index(s) + 1
    if s in _EN_MN:
        return _EN_MN.index(s) + 1
    # svenska fulla (om någon matar in så)
    if s in _SV_FULL:
        return _SV_FULL.index(s) + 1
    return 0

def _parse_months_field(s: str) -> List[int]:
    """
    Tar emot t.ex:
      "Mar, Jun, Sep, Dec"
      "3,6,9,12"
      "mar-apr-jul-okt"
    Returnerar lista med unika månader [1..12] i ordning.
    """
    if not s:
        return []
    # Splitta på icke-alfanumeriskt
    toks = re.split(r"[,\s;/\-]+", str(s))
    out: List[int] = []
    for t in toks:
        m = _parse_month_token(t)
        if m and m not in out:
            out.append(m)
    return out

def _parse_weights_field(s: str) -> List[float]:
    """
    Tar emot t.ex:
      "40,30,20,10" (procent)
      "0.25,0.25,0.25,0.25" (andelar)
    Normaliserar till summan 1.0.
    """
    if not s:
        return []
    toks = re.split(r"[,\s;/]+", str(s))
    vals = []
    for t in toks:
        v = _to_float(t)
        vals.append(v)
    if not vals:
        return []
    # Om det ser ut som procent (>1.0) -> skala ned
    if any(v > 1.0 for v in vals):
        vals = [v / 100.0 for v in vals]
    ssum = sum(vals)
    if ssum <= 0:
        return []
    return [v / ssum for v in vals]

def _infer_months_by_freq(freq: int) -> List[int]:
    """
    Om vi saknar explicit schema – välj hyggliga default-månader.
    12: varje månad
     4: kvartalsvis (Mar/Jun/Sep/Dec)
     2: halvårsvis (Jun/Dec)
     1: årsvis (Dec)
     3: var 4:e månad (Feb/Jun/Okt) – enkel spridning
     6: varannan månad (Feb-Apr-Jun-Aug-Okt-Dec)
    Annars: jämnt utspritt över året start i Mars.
    """
    if freq <= 0:
        return []
    if freq >= 12:
        return list(range(1, 13))
    presets = {
        4: [3, 6, 9, 12],
        2: [6, 12],
        1: [12],
        3: [2, 6, 10],
        6: [2, 4, 6, 8, 10, 12],
    }
    if freq in presets:
        return presets[freq]
    # generiskt: börja i mars (3) och sprid jämnt
    step = max(1, round(12 / freq))
    res = []
    cur = 3
    for _ in range(freq):
        res.append(((cur - 1) % 12) + 1)
        cur += step
    # unika och sortera
    res = sorted(list(dict.fromkeys(res)))
    return res[:freq]

def _equal_weights(n: int) -> List[float]:
    if n <= 0:
        return []
    return [1.0 / n] * n

def _normalize_weights(w: List[float], n: int) -> List[float]:
    """Säkerställ längd n + normalisera (fallback till lika)."""
    if n <= 0:
        return []
    if not w or len(w) != n:
        return _equal_weights(n)
    ssum = sum(w)
    if ssum <= 0:
        return _equal_weights(n)
    return [v / ssum for v in w]


# ------------------------------------------------------------
# Publik API
# ------------------------------------------------------------
def build_dividend_calendar(
    df_in: pd.DataFrame,
    rates: Dict[str, float],
    months_forward: int = 12,
    write_back_schedule: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bygger en utdelningskalender från df_in för nästa N månader.

    Kräver (per rad):
      - Ticker, Bolagsnamn, Valuta
      - Antal aktier
      - Årlig utdelning  (fallback: DA (%) * Aktuell kurs / 100)
      - Div_Frekvens/år, Div_Månader, Div_Vikter  (om saknas: heuristik)

    Parametrar:
      rates: dict, t.ex {"USD": 10.5, "SEK": 1.0, ...}
      months_forward: antal månader framåt att generera
      write_back_schedule: om infer → skriv tillbaka (i df_out)

    Return:
      (summary_df, detail_df, df_out)
        summary_df: kolumner ["År","Månad","Månad (sv)","Summa (SEK)"]
        detail_df : kolumner ["År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier",
                              "Valuta","Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"]
        df_out    : df med ev. ifyllda Div_Månader/Div_Vikter/Div_Frekvens/år
    """
    df = df_in.copy()

    # Säkerställ kolumner
    for col in ["Ticker","Bolagsnamn","Valuta","Antal aktier","Årlig utdelning","DA (%)","Aktuell kurs",
                "Div_Frekvens/år","Div_Månader","Div_Vikter"]:
        if col not in df.columns:
            # initiera lämplig typ
            df[col] = 0.0 if any(k in col.lower() for k in ["%", "frekvens", "aktier", "utdelning", "kurs"]) else ""

    # Aktuell utgångspunkt
    now = pd.Timestamp.now(tz="Europe/Stockholm")
    # Lista över kommande (År,Månad)
    months: List[Tuple[int,int]] = []
    cur_y = int(now.year)
    cur_m = int(now.month)
    for i in range(months_forward):
        m = ((cur_m - 1 + i) % 12) + 1
        y = cur_y + ((cur_m - 1 + i) // 12)
        months.append((y, m))

    detail_rows: List[dict] = []
    updated_rows: List[int] = []

    for idx, r in df.iterrows():
        try:
            shares = _to_float(r.get("Antal aktier", 0.0))
            if shares <= 0:
                continue

            # Valuta → SEK-kurs
            cur = str(r.get("Valuta", "") or "SEK").upper()
            fx = float(rates.get(cur, 1.0))

            # Årlig utd per aktie (valuta)
            dps_annual = _to_float(r.get("Årlig utdelning", 0.0))
            if dps_annual <= 0:
                # fallback: DA (%) * Aktuell kurs
                da = _to_float(r.get("DA (%)", 0.0))
                px = _to_float(r.get("Aktuell kurs", 0.0))
                if da > 0 and px > 0:
                    dps_annual = (da / 100.0) * px
            if dps_annual <= 0:
                continue  # inget att göra

            # Frekvens / schema / vikter
            months_field = str(r.get("Div_Månader", "") or "").strip()
            weights_field = str(r.get("Div_Vikter", "") or "").strip()
            freq_val = _to_float(r.get("Div_Frekvens/år", 0.0))

            months_list = _parse_months_field(months_field)
            if not months_list:
                # om vi har frekvens -> infer
                f = int(freq_val) if freq_val > 0 else 0
                if f <= 0:
                    # default-frekvens om helt saknas: kvartalsvis
                    f = 4
                months_list = _infer_months_by_freq(f)

            # Om frekvens saknades eller mismatch, sätt den från months_list
            if freq_val <= 0 or int(freq_val) != len(months_list):
                freq_val = float(len(months_list))

            weights = _parse_weights_field(weights_field)
            weights = _normalize_weights(weights, len(months_list))

            # Skapa mapping månad -> vikt
            mv = {m: weights[i] for i, m in enumerate(months_list)}

            # För varje framtida månad – har vi utdelning då?
            for (yy, mm) in months:
                w = mv.get(mm, 0.0)
                if w <= 0:
                    continue
                dps_this = dps_annual * w  # per aktie i bolagets valuta
                total_val = shares * dps_this  # i bolagets valuta
                total_sek = total_val * fx

                detail_rows.append({
                    "År": yy,
                    "Månad": mm,
                    "Månad (sv)": _month_name_sv(mm),
                    "Ticker": str(r.get("Ticker","")).upper(),
                    "Bolagsnamn": str(r.get("Bolagsnamn","")),
                    "Antal aktier": float(shares),
                    "Valuta": cur,
                    "Per utbetalning (valuta)": round(float(dps_this), 6),
                    "SEK-kurs": round(float(fx), 6),
                    "Summa (SEK)": round(float(total_sek), 2),
                })

            # Skriv tillbaka inferat schema om användaren bett om det
            if write_back_schedule:
                # Endast om något var tomt / mismatch – spara tillbaka
                new_months_str = ",".join([_SV_MN[m-1].capitalize() for m in months_list])
                new_weights_str = ",".join([str(round(w, 6)) for w in weights])
                changed = False
                if str(r.get("Div_Månader","")).strip() != new_months_str:
                    df.at[idx, "Div_Månader"] = new_months_str
                    changed = True
                if str(r.get("Div_Vikter","")).strip() != new_weights_str:
                    df.at[idx, "Div_Vikter"] = new_weights_str
                    changed = True
                if _to_float(r.get("Div_Frekvens/år", 0.0)) != float(len(months_list)):
                    df.at[idx, "Div_Frekvens/år"] = float(len(months_list))
                    changed = True
                if changed:
                    updated_rows.append(idx)

        except Exception:
            # per-rad-fel ignoreras — vi fortsätter
            continue

    # Bygg DataFrames
    if detail_rows:
        det = pd.DataFrame(detail_rows)
        summ = (
            det.groupby(["År","Månad"], as_index=False)["Summa (SEK)"]
            .sum()
            .sort_values(["År","Månad"])
        )
        summ["Månad (sv)"] = summ["Månad"].map(_month_name_sv)
        summ = summ[["År","Månad","Månad (sv)","Summa (SEK)"]]
        det = det.sort_values(["År","Månad","Ticker","Bolagsnamn"]).reset_index(drop=True)
    else:
        summ = pd.DataFrame(columns=["År","Månad","Månad (sv)","Summa (SEK)"])
        det  = pd.DataFrame(columns=["År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta",
                                     "Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"])

    return summ, det, df
