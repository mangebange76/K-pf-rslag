# stockapp/dividends.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# -------------------------------
# Datum & språk
# -------------------------------
SV_MONTHS = {
    1: "Januari", 2: "Februari", 3: "Mars", 4: "April",
    5: "Maj", 6: "Juni", 7: "Juli", 8: "Augusti",
    9: "September", 10: "Oktober", 11: "November", 12: "December"
}

def month_name_sv(m: int) -> str:
    return SV_MONTHS.get(int(m), str(m))


# -------------------------------
# YF helpers
# -------------------------------
def _fetch_dividends_series(ticker: str) -> pd.Series:
    """
    Hämtar historiska utdelningar (per datum) via yfinance.
    Returnerar en Series med DateTimeIndex och 'dividend per share' som values.
    Tom series om inget finns.
    """
    try:
        t = yf.Ticker(ticker)
        s = t.dividends  # pandas Series
        if isinstance(s, pd.Series) and not s.empty:
            s = s.dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
    except Exception:
        pass
    return pd.Series(dtype=float)


def _annual_dividend_from_series(div_s: pd.Series) -> float:
    """
    Summera senaste 12 månadernas utdelningar som proxy för 'årlig utdelning'.
    Fallback: summera senaste 4 utbetalningar.
    """
    if div_s.empty:
        return 0.0
    s = div_s.sort_index()
    end = s.index.max()
    start = end - pd.DateOffset(years=1)
    last_12m = s[(s.index > start) & (s.index <= end)]
    if not last_12m.empty:
        return float(last_12m.sum())
    return float(s.iloc[-4:].sum())


# -------------------------------
# Schema-inferens (frekvens, månader, vikter)
# -------------------------------
def _infer_frequency(div_s: pd.Series) -> int:
    """Avsluta frekvens (antal utbetalningar/år) utifrån intervall i månader."""
    if div_s.empty:
        return 4  # default kvartalsvis
    s = div_s.sort_index().iloc[-12:]  # senaste ~12 datapunkter
    idx = s.index.sort_values()
    diffs = []
    for i in range(1, len(idx)):
        d = (idx[i].year - idx[i-1].year) * 12 + (idx[i].month - idx[i-1].month)
        if d > 0:
            diffs.append(d)
    if not diffs:
        return 1
    # närmaste av typiska steg
    target_steps = [1, 2, 3, 4, 6, 12]
    rounded = []
    for d in diffs:
        nearest = min(target_steps, key=lambda k: abs(k - d))
        rounded.append(nearest)
    if not rounded:
        return 1
    top = pd.Series(rounded).value_counts().idxmax()
    if top == 1:   return 12
    if top == 2:   return 6
    if top == 3:   return 4
    if top == 4:   return 3  # ovanligt, de flesta hamnar på 3/år → vi degraderar till 3
    if top == 6:   return 2
    return 1


def _monthly_profile(div_s: pd.Series, freq: int) -> Tuple[List[int], List[float]]:
    """
    Skapa månadsprofil (månader + vikter) från historiken:
      - Summera utdelningsbelopp per månad (sista 3 åren om möjligt)
      - Välj de 'freq' mest sannolika månaderna
      - Normalisera vikterna så att de summerar till 1.0
    Ger både ojämna vikter (t.ex. hög Q4) och robusta månader.
    """
    if div_s.empty:
        # Standardmönster: kvartalsvis i Mar/Jun/Sep/Dec, månatlig = alla månader
        if freq == 12:
            months = list(range(1, 13))
            weights = [1/12.0] * 12
        elif freq == 6:
            months = [1,3,5,7,9,11]
            weights = [1/6.0] * 6
        elif freq == 4:
            months = [3,6,9,12]
            weights = [0.25, 0.25, 0.25, 0.25]
        elif freq == 3:
            months = [4,8,12]
            weights = [1/3.0, 1/3.0, 1/3.0]
        elif freq == 2:
            months = [6,12]
            weights = [0.5, 0.5]
        else:
            months = [6]
            weights = [1.0]
        return months, weights

    s = div_s.sort_index()
    end = s.index.max()
    start = end - pd.DateOffset(years=3)
    s3 = s[(s.index > start) & (s.index <= end)]
    if s3.empty:
        s3 = s

    by_month = s3.groupby(s3.index.month).sum()  # månads-summa
    if by_month.empty:
        # fallback jämt
        return _monthly_profile(pd.Series(dtype=float), freq)

    by_month = by_month[by_month > 0]
    if by_month.empty:
        return _monthly_profile(pd.Series(dtype=float), freq)

    # välj topp-månader
    top = by_month.sort_values(ascending=False).head(freq)
    months = sorted(top.index.tolist())

    # vikter = proportioner av månadssummor, normalisera
    weights = top.reindex(months).fillna(0.0).values.astype(float)
    total = float(weights.sum())
    if total <= 0:
        # jämt fallback
        weights = np.array([1.0 / len(months)] * len(months))
    else:
        weights = weights / total

    # Om vi fått färre än freq månader, fyll jämt
    if len(months) < freq:
        needed = freq - len(months)
        # välj jämna insprängningar
        add = []
        cur = (months[-1] if months else datetime.now().month)
        step = max(1, 12 // max(1, freq))
        for _ in range(needed):
            cur = ((cur - 1 + step) % 12) + 1
            if cur not in months:
                add.append(cur)
        months = sorted(list(set(months + add)))
        weights = np.array([1.0 / len(months)] * len(months))

    # säkerställ normalisering (floating noise)
    weights = (weights / weights.sum()).tolist()
    return months, weights


def infer_or_use_schedule_for_row(
    row: pd.Series,
    prefer_existing: bool = True
) -> Tuple[int, List[int], List[float]]:
    """
    Läser befintligt schema från raden (om finns) annars infererar från historik.
    Befintliga kolumner (om du lagt in dem i arket):
      - 'Div_Frekvens/år' (int)
      - 'Div_Månader'     (CSV, t.ex. "3,6,9,12")
      - 'Div_Vikter'      (CSV, t.ex. "0.25,0.25,0.25,0.25")
    Returnerar (freq, months, weights) där sum(weights)==1.
    """
    # 1) Befintliga fält?
    if prefer_existing:
        try:
            freq0 = int(float(row.get("Div_Frekvens/år", 0.0)))
        except Exception:
            freq0 = 0
        months0 = str(row.get("Div_Månader", "") or "").strip()
        weights0 = str(row.get("Div_Vikter", "") or "").strip()
        if freq0 > 0 and months0:
            try:
                m_list = [int(x) for x in months0.split(",") if x.strip()]
            except Exception:
                m_list = []
            try:
                w_list = [float(x) for x in weights0.split(",") if x.strip()]
            except Exception:
                w_list = []
            # normalisera vikter
            if m_list:
                if not w_list or len(w_list) != len(m_list) or sum(w_list) <= 0:
                    w_list = [1.0 / len(m_list)] * len(m_list)
                else:
                    s = float(sum(w_list))
                    w_list = [w / s for w in w_list]
                return int(freq0), m_list, w_list

    # 2) Annars inferera
    tkr = str(row.get("Ticker", "")).upper()
    div_s = _fetch_dividends_series(tkr)
    freq = _infer_frequency(div_s)
    months, weights = _monthly_profile(div_s, freq)
    return int(freq), months, weights


# -------------------------------
# Projektering
# -------------------------------
@dataclass
class ProjectedPayment:
    year: int
    month: int
    month_name: str
    ticker: str
    company: str
    currency: str
    shares: float
    per_share_div: float  # i aktiens valuta
    total_local: float
    rate_to_sek: float
    total_sek: float


def _project_months_ahead(start_dt: datetime, months_ahead: int) -> List[Tuple[int, int]]:
    out = []
    y, m = start_dt.year, start_dt.month
    for i in range(months_ahead):
        mm = ((m - 1 + i) % 12) + 1
        yy = y + ((m - 1 + i) // 12)
        out.append((yy, mm))
    return out


def build_dividend_calendar(
    df: pd.DataFrame,
    rates: Dict[str, float],
    months_forward: int = 12,
    write_back_schedule: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Skapar (summering_per_månad, detaljerade_utbetalningar, df_med_schema) för nästa 12 månader.

    - Läser/beräknar schema per rad (freq, månader, vikter).
    - Årlig utdelning tas från df['Årlig utdelning'] om >0, annars från historik (sum 12m / 4 senaste).
    - Viktar utbetalningarna ojämnt per månad enligt vikter.
    - Konverterar till SEK via 'rates' mha 'Valuta'.
    - Om write_back_schedule=True läggs/uppdateras:
        'Div_Frekvens/år', 'Div_Månader', 'Div_Vikter' i den returnerade df:n.

    Returnerar:
      (summ_df, detalj_df, df_out)
    """
    today = datetime.now()
    horizon = _project_months_ahead(today, months_forward)

    if df.empty:
        months = [{"År": y, "Månad": m, "Månad (sv)": month_name_sv(m), "Summa (SEK)": 0.0} for (y, m) in horizon]
        return pd.DataFrame(months), pd.DataFrame(columns=[
            "År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta",
            "Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"
        ]), df.copy()

    port = df[(df.get("Antal aktier", 0) > 0)].copy()
    if port.empty:
        months = [{"År": y, "Månad": m, "Månad (sv)": month_name_sv(m), "Summa (SEK)": 0.0} for (y, m) in horizon]
        return pd.DataFrame(months), pd.DataFrame(columns=[
            "År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta",
            "Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"
        ]), df.copy()

    # säkerställ schema-kolumner finns om vi vill skriva tillbaka
    df_out = df.copy()
    if write_back_schedule:
        if "Div_Frekvens/år" not in df_out.columns: df_out["Div_Frekvens/år"] = 0.0
        if "Div_Månader" not in df_out.columns:     df_out["Div_Månader"] = ""
        if "Div_Vikter" not in df_out.columns:      df_out["Div_Vikter"] = ""

    rows: List[ProjectedPayment] = []

    for idx, r in port.iterrows():
        tkr = str(r.get("Ticker", "")).upper()
        company = str(r.get("Bolagsnamn", "")) or tkr
        currency = str(r.get("Valuta", "USD")).upper() or "USD"
        shares = float(r.get("Antal aktier", 0.0)) or 0.0
        if shares <= 0:
            continue

        # schema
        freq, months, weights = infer_or_use_schedule_for_row(r, prefer_existing=True)

        # årlig utdelning
        annual_div = float(r.get("Årlig utdelning", 0.0)) or 0.0
        if annual_div <= 0:
            # hämta från historik
            div_s = _fetch_dividends_series(tkr)
            annual_div = _annual_dividend_from_series(div_s)

        if annual_div <= 0 or not months:
            # ingen utdelning/saknar schema
            continue

        # sanity: normalisera vikter
        if len(weights) != len(months) or sum(weights) <= 0:
            weights = [1.0 / len(months)] * len(months)
        else:
            s = float(sum(weights))
            weights = [w / s for w in weights]

        # Skriv tillbaka schema till DF om så önskas
        if write_back_schedule:
            df_out.loc[df_out["Ticker"] == r["Ticker"], "Div_Frekvens/år"] = float(freq)
            df_out.loc[df_out["Ticker"] == r["Ticker"], "Div_Månader"]     = ",".join(str(m) for m in months)
            df_out.loc[df_out["Ticker"] == r["Ticker"], "Div_Vikter"]      = ",".join(f"{w:.6f}" for w in weights)

        # Projektera betalningar
        month_weight_map = dict(zip(months, weights))
        for (yy, mm) in horizon:
            w = float(month_weight_map.get(int(mm), 0.0))
            if w <= 0:
                continue
            per_share = annual_div * w
            rate = float(rates.get(currency, 1.0))
            total_local = per_share * shares
            total_sek = total_local * rate
            rows.append(ProjectedPayment(
                year=yy, month=mm, month_name=month_name_sv(mm),
                ticker=tkr, company=company, currency=currency,
                shares=shares, per_share_div=per_share,
                total_local=total_local, rate_to_sek=rate, total_sek=total_sek
            ))

    # Bygg tabeller
    if not rows:
        months = [{"År": y, "Månad": m, "Månad (sv)": month_name_sv(m), "Summa (SEK)": 0.0} for (y, m) in horizon]
        return pd.DataFrame(months), pd.DataFrame(columns=[
            "År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta",
            "Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"
        ]), df_out

    det = pd.DataFrame([{
        "År": p.year,
        "Månad": p.month,
        "Månad (sv)": p.month_name,
        "Ticker": p.ticker,
        "Bolagsnamn": p.company,
        "Antal aktier": round(p.shares, 6),
        "Valuta": p.currency,
        "Per utbetalning (valuta)": round(p.per_share_div, 6),
        "SEK-kurs": round(p.rate_to_sek, 6),
        "Summa (SEK)": round(p.total_sek, 2),
    } for p in rows]).sort_values(by=["År","Månad","Ticker"]).reset_index(drop=True)

    summ = det.groupby(["År","Månad","Månad (sv)"], as_index=False)["Summa (SEK)"].sum()
    summ["Summa (SEK)"] = summ["Summa (SEK)"].round(2)

    # säkerställ alla månader i horisonten
    all_rows = []
    for (yy, mm) in horizon:
        mask = (summ["År"] == yy) & (summ["Månad"] == mm)
        if mask.any():
            all_rows.append(summ[mask].iloc[0].to_dict())
        else:
            all_rows.append({"År": yy, "Månad": mm, "Månad (sv)": month_name_sv(mm), "Summa (SEK)": 0.0})
    summ2 = pd.DataFrame(all_rows).sort_values(by=["År","Månad"]).reset_index(drop=True)

    return summ2, det, df_out
