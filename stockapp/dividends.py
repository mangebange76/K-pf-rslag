# stockapp/dividends.py
import pandas as pd
import numpy as np

MONTHS_SV = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Maj",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Okt",11:"Nov",12:"Dec"
}

def _to_float(x):
    try:
        s = str(x).strip().replace("\u00a0","").replace(" ","").replace(",",".")
        if s in ("","-","nan","None"):
            return 0.0
        return float(s)
    except:
        try:
            return float(x)
        except:
            return 0.0

def build_dividend_calendar(df_in: pd.DataFrame, rates: dict, months_forward: int = 12, write_back_schedule: bool = True):
    df = df_in.copy()

    # Säkerställ schemafält
    if "Div_Frekvens/år" not in df.columns: df["Div_Frekvens/år"] = 0.0
    if "Div_Månader"    not in df.columns: df["Div_Månader"]    = ""
    if "Div_Vikter"     not in df.columns: df["Div_Vikter"]     = ""

    # Default: om DA finns, anta kvartalsvis
    df["Div_Frekvens/år"] = df["Div_Frekvens/år"].map(_to_float)
    df.loc[df["Div_Frekvens/år"]<=0, "Div_Frekvens/år"] = 4.0
    # Default months om tomt: Jan/Apr/Jul/Okt
    df.loc[df["Div_Månader"].astype(str).str.strip()=="", "Div_Månader"] = "1,4,7,10"
    df.loc[df["Div_Vikter"].astype(str).str.strip()=="",  "Div_Vikter"]  = "1,1,1,1"

    det_rows = []
    # Generera 12 månader framåt från nu
    from datetime import datetime
    start = datetime.today()
    months = []
    y, m = start.year, start.month
    for i in range(months_forward):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1

    for _, r in df.iterrows():
        tkr = str(r.get("Ticker","")).strip()
        if not tkr: continue
        namn = str(r.get("Bolagsnamn","")).strip()
        antal = _to_float(r.get("Antal aktier",0.0))
        valuta = str(r.get("Valuta","SEK")).strip().upper()
        vx = float(rates.get(valuta, 1.0))
        arlig = _to_float(r.get("Årlig utdelning",0.0))
        freq  = int(max(1, _to_float(r.get("Div_Frekvens/år",4.0))))

        # parse månader/vikter
        try:
            ms = [int(x) for x in str(r.get("Div_Månader","")).replace(" ","").split(",") if str(x).strip()]
        except:
            ms = [1,4,7,10]
        try:
            ws = [float(x) for x in str(r.get("Div_Vikter","")).replace(" ","").split(",") if str(x).strip()]
        except:
            ws = [1,1,1,1]
        if len(ms) != len(ws):
            # fallback jämn viktning
            ms = [1,4,7,10]
            ws = [1,1,1,1]

        wsum = sum(ws) if ws else 1.0
        per_payment_currency = [(arlig * (w/wsum))/1.0 for w in ws]  # per aktie i bolagets valuta

        # expandera till månader
        sched = {mth:0.0 for mth in range(1,13)}
        for mon, val in zip(ms, per_payment_currency):
            sched[int(mon)] += val

        # skapa rader för 12 mån framåt
        for (yy, mm) in months:
            per_share_val = sched.get(mm, 0.0)
            if per_share_val <= 0: continue
            summa_sek = antal * per_share_val * vx
            det_rows.append({
                "År": yy,
                "Månad": mm,
                "Månad (sv)": MONTHS_SV.get(mm, str(mm)),
                "Ticker": tkr,
                "Bolagsnamn": namn,
                "Antal aktier": antal,
                "Valuta": valuta,
                "Per utbetalning (valuta)": round(per_share_val, 4),
                "SEK-kurs": round(vx, 4),
                "Summa (SEK)": round(summa_sek, 2),
            })

    det = pd.DataFrame(det_rows)
    if det.empty:
        summ = pd.DataFrame(columns=["År","Månad","Månad (sv)","Summa (SEK)"])
    else:
        summ = det.groupby(["År","Månad","Månad (sv)"], as_index=False)["Summa (SEK)"].sum().sort_values(["År","Månad"])

    df_out = df.copy()
    if write_back_schedule:
        # Fälten kan ha justerats; skriv tillbaka
        pass

    return summ, det, df_out
