# app.py
from __future__ import annotations

import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import gspread
import yfinance as yf
from google.oauth2.service_account import Credentials

# ---------- TID ----------
try:
    import pytz
    TZ = pytz.timezone("Europe/Stockholm")
    def now_stamp() -> str:
        return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
except Exception:
    def now_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

st.set_page_config(page_title="Aktieanalys & investeringsförslag", layout="wide")

# ---------- GOOGLE SHEETS ----------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last

def _open_ss():
    return client.open_by_url(SHEET_URL)

def _ws_main():
    return _open_ss().worksheet(SHEET_NAME)

def _ws_rates():
    ss = _open_ss()
    try:
        return ss.worksheet(RATES_SHEET)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET, rows=10, cols=3)
        ws = ss.worksheet(RATES_SHEET)
        ws.update([["Valuta","Kurs"]])
        return ws

# ---------- SVE-DECIMAL HJÄLPARE ----------
def to_float_swe(x) -> float:
    """Tål '10,6', '10.6', '1 234,56', '1.234,56' osv."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.number)):
        try: return float(x)
        except Exception: return 0.0
    s = str(x).strip()
    if s == "": return 0.0
    s = s.replace("\u00A0", " ")        # hårt mellanslag
    s = s.replace(" ", "")              # ta bort eventuella tusentals-mellanslag
    # Om båda tecken finns, anta '.'=tusen, ','=decimal
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # bara komma => decimal
        if "," in s:
            s = s.replace(",", ".")
        # bara punkt => redan decimal, låt vara
    try:
        return float(s)
    except Exception:
        return 0.0

def col_to_float_swe(series: pd.Series) -> pd.Series:
    return series.apply(to_float_swe)

# ---------- DATA I/O ----------
FINAL_COLS = [
    "Ticker","Bolagsnamn","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","GAV (SEK)","Valuta","Årlig utdelning","Aktuell kurs",
    "CAGR 5 år (%)","P/S-snitt",
    "Senast manuellt uppdaterad"
]

def hamta_data() -> pd.DataFrame:
    ws = _ws_main()
    rows = _with_backoff(ws.get_all_records)
    df = pd.DataFrame(rows)

    # Säkerställ schema
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad") else 0.0

    # Typer (svenska decimaler in)
    num_cols = [
        "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","GAV (SEK)","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
    ]
    for c in num_cols:
        df[c] = col_to_float_swe(df[c])
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        df[c] = df[c].astype(str)

    return df[FINAL_COLS].copy()

def spara_data(df: pd.DataFrame):
    """Skriv tillbaka som str (lämna decimalpunkt; GSheets visar fint ändå)."""
    ws = _ws_main()
    _with_backoff(ws.clear)
    values = [df.columns.tolist()] + df.astype(object).where(pd.notnull(df), "").astype(str).values.tolist()
    _with_backoff(ws.update, values)

# ---------- VALUTA ----------
STANDARD_RATES = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int) -> dict:
    ws = _ws_rates()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = to_float_swe(r.get("Kurs",""))
        if cur: out[cur] = val
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = _ws_rates()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        body.append([k, str(float(rates.get(k, STANDARD_RATES.get(k,1.0))))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

@st.cache_data(show_spinner=False, ttl=3600)
def hamta_valutakurser_automatiskt() -> dict:
    pairs = {"USD":"USDSEK=X","NOK":"NOKSEK=X","CAD":"CADSEK=X","EUR":"EURSEK=X"}
    got = {"SEK": 1.0}
    try:
        data = yf.download(tickers=" ".join(pairs.values()), period="1d", interval="1d",
                           progress=False, group_by="ticker", threads=True)
        for c, y in pairs.items():
            try:
                if isinstance(data, pd.DataFrame) and y in getattr(data.columns, "levels", [[y]])[0]:
                    ser = data[y]["Close"].dropna()
                else:
                    ser = data["Close"].dropna()
                if not ser.empty:
                    got[c] = float(ser.iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return got

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta: return 1.0
    v = user_rates.get(valuta.upper(), STANDARD_RATES.get(valuta.upper(), 1.0))
    try: return float(v)
    except Exception: return 1.0

# ---------- YAHOO ----------
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    try:
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            s = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                s = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if len(s) < 2: return 0.0
        s = s.sort_index()
        start, end = float(s.iloc[0]), float(s.iloc[-1])
        if start <= 0: return 0.0
        years = max(1, len(s)-1)
        return round(((end/start) ** (1/years) - 1) * 100, 2)
    except Exception:
        return 0.0

def hamta_yahoo_fält(ticker: str) -> dict:
    out = {"Bolagsnamn":"", "Aktuell kurs":0.0, "Valuta":"USD", "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        px = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        if px is not None: out["Aktuell kurs"] = float(px)

        cur = info.get("currency")
        if cur: out["Valuta"] = str(cur).upper()

        name = info.get("shortName") or info.get("longName") or ""
        if name: out["Bolagsnamn"] = str(name)

        dr = info.get("dividendRate", None)
        if dr is not None:
            out["Årlig utdelning"] = float(dr)

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---------- BERÄKNING ----------
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    df = df.copy()
    for i, r in df.iterrows():
        ps_vals = [r.get("P/S Q1",0), r.get("P/S Q2",0), r.get("P/S Q3",0), r.get("P/S Q4",0)]
        ps_clean = [to_float_swe(x) for x in ps_vals if to_float_swe(x) > 0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        cagr = to_float_swe(r.get("CAGR 5 år (%)", 0.0))
        cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = cagr / 100.0

        next_rev = to_float_swe(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            df.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)

        shares_m = to_float_swe(r.get("Utestående aktier", 0.0))
        if shares_m > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round(to_float_swe(r.get("Omsättning idag",0))     * ps_snitt / shares_m, 2)
            df.at[i, "Riktkurs om 1 år"] = round(next_rev                                      * ps_snitt / shares_m, 2)
            df.at[i, "Riktkurs om 2 år"] = round(to_float_swe(df.at[i,"Omsättning om 2 år"])  * ps_snitt / shares_m, 2)
            df.at[i, "Riktkurs om 3 år"] = round(to_float_swe(df.at[i,"Omsättning om 3 år"])  * ps_snitt / shares_m, 2)
        else:
            for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i, c] = 0.0
    return df

# ---------- MASSUPPDATERA ----------
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        total = len(df)
        miss = []
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            bad = []
            if data.get("Bolagsnamn"): df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else: bad.append("Bolagsnamn")
            if data.get("Aktuell kurs",0)>0: df.at[i,"Aktuell kurs"] = data["Aktuell kurs"]
            else: bad.append("Aktuell kurs")
            if data.get("Valuta"): df.at[i,"Valuta"] = data["Valuta"]
            else: bad.append("Valuta")
            if "Årlig utdelning" in data: df.at[i,"Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            else: bad.append("Årlig utdelning")
            if "CAGR 5 år (%)" in data: df.at[i,"CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)
            else: bad.append("CAGR 5 år (%)")
            if bad: miss.append(f"{tkr}: {', '.join(bad)}")
            time.sleep(0.3)
            bar.progress((i+1)/max(1,total))
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if miss:
            st.sidebar.warning("Vissa fält saknades:"); st.sidebar.text_area("Detaljer", "\n".join(miss), height=150)
    return df

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# ---------- VY: Lägg till / uppdatera ----------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista,
                              index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_info, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_info:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            antal  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0, step=1.0, format="%.0f")
            gav    = st.number_input("GAV (SEK)", value=float(bef.get("GAV (SEK)",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps   = st.number_input("P/S", value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps1  = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps2  = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps3  = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps4  = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            st.caption("Vid spar: Bolagsnamn/Valuta/Aktuell kurs/Utdelning/CAGR hämtas automatiskt.")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # DUBBLETTKONTROLL
        new_tkr = (ticker or "").strip().upper()
        cur_tkr = (bef.get("Ticker","") if not bef.empty else "").strip().upper()
        tkr_norm = df["Ticker"].astype(str).str.strip().str.upper()
        if bef.empty:
            if (tkr_norm == new_tkr).any():
                st.error(f"Tickern **{new_tkr}** finns redan. Välj den i listan för att redigera."); st.stop()
        else:
            if new_tkr != cur_tkr and (tkr_norm == new_tkr).any():
                st.error(f"Kan inte byta till **{new_tkr}** – den finns redan i en annan rad."); st.stop()

        ny = {
            "Ticker": new_tkr,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "GAV (SEK)": gav,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # sätt datum om manuella fält ändrats
        datum_satt = False
        if not bef.empty:
            before = {f: to_float_swe(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: to_float_swe(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_satt = True
        else:
            if any(to_float_swe(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_satt = True

        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"]==cur_tkr, k] = v
            if new_tkr != cur_tkr:
                df.loc[df["Ticker"]==cur_tkr, "Ticker"] = new_tkr
        else:
            tom = {c: ("" if c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"] else 0.0) for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_satt:
            df.loc[df["Ticker"]==new_tkr, "Senast manuellt uppdaterad"] = now_stamp()

        auto = hamta_yahoo_fält(new_tkr)
        if auto.get("Bolagsnamn"): df.loc[df["Ticker"]==new_tkr, "Bolagsnamn"] = auto["Bolagsnamn"]
        if auto.get("Valuta"):     df.loc[df["Ticker"]==new_tkr, "Valuta"] = auto["Valuta"]
        if auto.get("Aktuell kurs",0)>0: df.loc[df["Ticker"]==new_tkr, "Aktuell kurs"] = auto["Aktuell kurs"]
        if "Årlig utdelning" in auto:    df.loc[df["Ticker"]==new_tkr, "Årlig utdelning"] = float(auto.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in auto:      df.loc[df["Ticker"]==new_tkr, "CAGR 5 år (%)"] = float(auto.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")
        st.rerun()

    # ⏱️ Äldst manuellt uppdaterade – TOPP 10
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )
    return df

# ---------- VY: Analys ----------
def analysvy(df: pd.DataFrame, user_rates: dict):
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", 0, max(0, len(etiketter)-1), st.session_state.analys_idx, 1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","Antal aktier","GAV (SEK)","Årlig utdelning","Senast manuellt uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

# ---------- VY: Portfölj ----------
def visa_portfolj(df: pd.DataFrame, user_rates: dict):
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    port["Vinst/Förlust (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst/Förlust (%)"] = np.where(
        port["Anskaffningsvärde (SEK)"] > 0,
        (port["Vinst/Förlust (SEK)"] / port["Anskaffningsvärde (SEK)"]) * 100.0,
        0.0
    )
    tot = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(tot>0, round(port["Värde (SEK)"]/tot*100.0, 2), 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    st.markdown(f"**Totalt portföljvärde:** {round(tot,2)} SEK")
    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","GAV (SEK)","Anskaffningsvärde (SEK)","Aktuell kurs","Valuta","Växelkurs",
              "Värde (SEK)","Vinst/Förlust (SEK)","Vinst/Förlust (%)","Årlig utdelning","Total årlig utdelning (SEK)","Andel (%)"]],
        use_container_width=True
    )

# ---------- VY: Investeringsförslag ----------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict):
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)
    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    ps_filter = st.selectbox("Filtrera på P/S vs P/S-snitt", ["Alla","P/S under snitt","P/S över snitt"], index=0)

    base = df.copy() if subset == "Alla bolag" else df[df["Antal aktier"] > 0].copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()

    if ps_filter == "P/S under snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] < base["P/S-snitt"])]
    elif ps_filter == "P/S över snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] > base["P/S-snitt"])]

    if base.empty:
        st.info("Inga bolag matchar just nu."); return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    if "forslags_index" not in st.session_state: st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("⬅️ Föregående förslag"): st.session_state.forslags_index = max(0, st.session_state.forslags_index-1)
    with c2:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with c3:
        if st.button("➡️ Nästa förslag"): st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index+1)

    rad = base.iloc[st.session_state.forslags_index]

    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
f"""- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Nuvarande P/S (TTM):** {round(rad.get('P/S',0.0), 2)}
- **P/S-snitt (Q1–Q4):** {round(rad.get('P/S-snitt',0.0), 2)}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 3 år" else ""}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# ---------- MAIN ----------
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # SIDOMENY: valutakurser (endast här – inga inputs i huvudytan)
    st.sidebar.header("💱 Valutakurser → SEK")
    saved = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", STANDARD_RATES["USD"])), step=0.0001, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", STANDARD_RATES["NOK"])), step=0.0001, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", STANDARD_RATES["CAD"])), step=0.0001, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", STANDARD_RATES["EUR"])), step=0.0001, format="%.4f")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara valutakurser"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with c2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear(); st.rerun()

    if st.sidebar.button("🌐 Hämta valutakurser (Yahoo)"):
        live = hamta_valutakurser_automatiskt()
        if any(k in live for k in ("USD","NOK","CAD","EUR")):
            merged = las_sparade_valutakurser(); merged.update(live); spara_valutakurser(merged)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser uppdaterade."); st.rerun()
        else:
            st.sidebar.error("Kunde inte hämta kurser just nu.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear(); st.rerun()

    # Läs & förbered data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS}); spara_data(df)

    # Global Yahoo-uppdatering i sidomeny
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])
    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates); visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates); visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
