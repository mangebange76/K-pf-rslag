from __future__ import annotations
from typing import Tuple, Dict
import pandas as pd
import numpy as np

def _parse_months(s: str) -> list[int]:
    if not s: return []
    out=[]
    for tok in str(s).replace(","," ").split():
        try:
            m=int(tok)
            if 1<=m<=12: out.append(m)
        except:
            pass
    return out

def _parse_weights(s: str, n: int) -> list[float]:
    if not s or str(s).strip()=="":
        return [1.0]*n
    vals=[]
    for tok in str(s).replace(","," ").split():
        try: vals.append(float(tok))
        except: pass
    if len(vals)!=n:
        return [1.0]*n
    return vals

def build_dividend_calendar(df: pd.DataFrame, rates: Dict[str,float], months_forward=12, write_back_schedule=True) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    base = df[(df["Antal aktier"]>0) & (df["Årlig utdelning"]>0)].copy()
    if base.empty:
        return (pd.DataFrame(), pd.DataFrame(), df)

    today = pd.Timestamp.today().normalize()
    month_list = [(today + pd.DateOffset(months=i)).to_period("M") for i in range(months_forward)]

    det_rows=[]
    for _, r in base.iterrows():
        tkr = str(r["Ticker"])
        name= str(r["Bolagsnamn"])
        cur = str(r["Valuta"]).upper()
        rate= float(rates.get(cur, 1.0))
        shares = float(r["Antal aktier"])
        annual = float(r["Årlig utdelning"])

        freq = int(float(r.get("Div_Frekvens/år",0.0)) or 0)
        months = _parse_months(str(r.get("Div_Månader","")))
        if freq<=0:
            if months: freq=len(months)
            else: freq=4; months=[3,6,9,12]

        if not months or len(months)!=freq:
            months = months if months else [3,6,9,12]
            if len(months)!=freq:
                freq=len(months)

        weights = _parse_weights(str(r.get("Div_Vikter","")), freq)
        wsum = sum(weights) if sum(weights)>0 else float(freq)
        per_payment = [annual * (w/wsum) for w in weights]

        for mi, m in enumerate(months):
            # hitta alla framtida månader som matchar m
            for p in month_list:
                if p.month == m:
                    gross_val = per_payment[mi] * shares
                    sek = gross_val * rate
                    det_rows.append([int(p.year), int(p.month), tkr, name, shares, cur, round(per_payment[mi],4), round(rate,6), round(sek,2)])

    det = pd.DataFrame(det_rows, columns=["År","Månad","Ticker","Bolagsnamn","Antal aktier","Valuta","Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"])
    if det.empty:
        return (pd.DataFrame(), pd.DataFrame(), df)
    det["Månad (sv)"] = det["Månad"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Maj",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Okt",11:"Nov",12:"Dec"})
    summ = det.groupby(["År","Månad","Månad (sv)"], as_index=False)["Summa (SEK)"].sum().sort_values(["År","Månad"])
    return (summ, det, df)
