from __future__ import annotations
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import gspread
import yfinance as yf
from google.oauth2.service_account import Credentials

# ---- Lokal Stockholm-tid (fallback systemtid) ----
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ======================
# Svensk tal-hantering
# ======================
NBSP = "\u00A0"

def sv_to_float(x, default=0.0) -> float:
    """Accepterar '10,61', '10.61', '1 234,56', '1 234,56' m.m."""
    if x is None:
        return float(default)
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(NBSP, " ").replace(" ", "")
    if s == "" or s.lower() in {"nan", "none"}:
        return float(default)
    if "," in s and "." in s:
        # Bestäm decimaltecknet via sista förekomsten
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float(default)

def fmt_sv(val: float, decimals: int = 4) -> str:
    """Format för visning med svensk decimal."""
    try:
        return f"{float(val):.{decimals}f}".replace(".", ",")
    except Exception:
        return str(val)

def sv_number_input(label: str, value: float = 0.0, step: float = 0.01, key: str | None = None, help: str | None = None):
    """
    Text-input som visar svensk formatering och accepterar både komma/punkt.
    Returnerar float.
    """
    txt = st.text_input(label, fmt_sv(value), key=key, help=help)
    return sv_to_float(txt, default=value)

# ======================
# Google Sheets
# ======================
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    err = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = e
    raise err

def get_ss():
    return client.open_by_url(SHEET_URL)

def ws_main():
    return get_ss().worksheet(SHEET_NAME)

def ws_rates():
    ss = get_ss()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(RATES_SHEET_NAME, rows=20, cols=3)
        ws = ss.worksheet(RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta", "Kurs"]])
        return ws

STANDARD_VALUTAKURSER = {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def load_saved_rates(nonce: int) -> dict:
    rows = _with_backoff(ws_rates().get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = sv_to_float(r.get("Kurs", 0.0), default=0.0)
        if cur:
            out[cur] = float(val)
    # fyll upp saknade med standard
    for k,v in STANDARD_VALUTAKURSER.items():
        out.setdefault(k, v)
    return out

def get_saved_rates() -> dict:
    return load_saved_rates(st.session_state.get("rates_nonce", 0))

def save_rates(rates: dict):
    body = [["Valuta", "Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = float(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))
        body.append([k, f"{v:.6f}"])  # punkt-decimal till Sheets
    w = ws_rates()
    _with_backoff(w.clear)
    _with_backoff(w.update, body)

# Live-kurser (valuta) via Yahoo
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_fx() -> dict:
    pairs = {"USD": "USDSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X", "EUR": "EURSEK=X"}
    out = {"SEK": 1.0}
    for code, sym in pairs.items():
        try:
            h = yf.Ticker(sym).history(period="1d")
            if not h.empty and "Close" in h:
                out[code] = float(h["Close"].iloc[-1])
        except Exception:
            pass
    return out

def auto_update_rates_if_needed():
    try:
        cur = get_saved_rates()
        live = fetch_live_fx()
        if not live: 
            return
        changed = any(abs(float(cur.get(k,0))-float(live.get(k,0)))>1e-6 for k in ("USD","NOK","CAD","EUR"))
        if changed:
            cur.update(live)
            save_rates(cur)
            st.session_state["rates_nonce"] = st.session_state.get("rates_nonce",0)+1
    except Exception:
        pass

# ======================
# Kolumnschema
# ======================
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt", "Senast manuellt uppdaterad"
]

NUM_COLS = [
    "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","GAV (SEK)","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
]

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = "" if c not in NUM_COLS else 0.0
    # typfix: parse alla NUM_COLS via sv_to_float
    for c in NUM_COLS:
        df[c] = df[c].apply(lambda x: sv_to_float(x, 0.0))
    # str-kolumner
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        df[c] = df[c].astype(str)
    return df[FINAL_COLS].copy()

def read_db() -> pd.DataFrame:
    w = ws_main()
    rows = _with_backoff(w.get_all_values)
    if not rows:
        return pd.DataFrame({c: [] for c in FINAL_COLS})
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    df = ensure_cols(df)
    return df

def write_db(df: pd.DataFrame):
    # Skriv numeriska som punkt-decimal; övrigt som strängar.
    out = df.copy()
    for c in NUM_COLS:
        out[c] = out[c].apply(lambda v: f"{float(v):.6f}")
    body = [out.columns.tolist()] + out.astype(str).values.tolist()
    w = ws_main()
    _with_backoff(w.clear)
    _with_backoff(w.update, body)

# ========== Yahoo-hjälp ==========
def cagr_from_financials(tkr: yf.Ticker) -> float:
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
        if s.empty or len(s) < 2: return 0.0
        s = s.sort_index()
        start, end = float(s.iloc[0]), float(s.iloc[-1])
        years = max(1, len(s)-1)
        if start <= 0: return 0.0
        return round(((end/start)**(1/years) - 1.0) * 100.0, 2)
    except Exception:
        return 0.0

def yahoo_fields(ticker: str) -> dict:
    out = {"Bolagsnamn":"", "Aktuell kurs":0.0, "Valuta":"USD", "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        price = info.get("regularMarketPrice")
        if price is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        if price is not None: out["Aktuell kurs"] = float(price)
        cur = info.get("currency")
        if cur: out["Valuta"] = str(cur).upper()
        name = info.get("shortName") or info.get("longName") or ""
        if name: out["Bolagsnamn"] = str(name)
        div = info.get("dividendRate")
        if div is not None: out["Årlig utdelning"] = float(div)
        out["CAGR 5 år (%)"] = cagr_from_financials(t)
    except Exception:
        pass
    return out

# ======================
# Beräkningar
# ======================
def recalc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, r in df.iterrows():
        ps_vals = [r.get("P/S Q1",0), r.get("P/S Q2",0), r.get("P/S Q3",0), r.get("P/S Q4",0)]
        ps_clean = [sv_to_float(x,0.0) for x in ps_vals if sv_to_float(x,0.0) > 0]
        ps_avg = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_avg

        cagr = float(r.get("CAGR 5 år (%)", 0.0))
        cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = cagr/100.0
        next_rev = sv_to_float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            df.at[i, "Omsättning om 2 år"] = round(next_rev*(1+g), 2)
            df.at[i, "Omsättning om 3 år"] = round(next_rev*((1+g)**2), 2)

        shares_m = sv_to_float(r.get("Utestående aktier", 0.0))
        if shares_m > 0 and ps_avg > 0:
            df.at[i, "Riktkurs idag"]    = round(sv_to_float(r.get("Omsättning idag",0))*ps_avg/shares_m, 2)
            df.at[i, "Riktkurs om 1 år"] = round(next_rev*ps_avg/shares_m, 2)
            df.at[i, "Riktkurs om 2 år"] = round(sv_to_float(df.at[i,"Omsättning om 2 år"])*ps_avg/shares_m, 2)
            df.at[i, "Riktkurs om 3 år"] = round(sv_to_float(df.at[i,"Omsättning om 3 år"])*ps_avg/shares_m, 2)
        else:
            for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i, c] = 0.0
    return df

# ======================
# Massuppdatering
# ======================
def massupdate_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        total = len(df)
        fails = []
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = yahoo_fields(tkr)

            if data.get("Bolagsnamn"): df.at[i,"Bolagsnamn"] = data["Bolagsnamn"]
            if data.get("Valuta"): df.at[i,"Valuta"] = data["Valuta"]
            if data.get("Aktuell kurs",0)>0: df.at[i,"Aktuell kurs"] = data["Aktuell kurs"]
            if "Årlig utdelning" in data: df.at[i,"Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            if "CAGR 5 år (%)" in data: df.at[i,"CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

            bar.progress((i+1)/max(1,total))
            time.sleep(0.4)

        df = recalc(df)
        write_db(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
    return df

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# ======================
# Lägg till / Uppdatera
# ======================
def view_edit(df: pd.DataFrame) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0

    valt = st.selectbox("Välj bolag (tomt = nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_l, col_c, col_r = st.columns([1,2,1])
    with col_l:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index-1)
    with col_c:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_r:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index+1)

    if valt and valt in namn_map:
        bef = df[df["Ticker"]==namn_map[valt]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("bolag_form"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo)", value=(bef.get("Ticker","") if not bef.empty else "")).upper().strip()
            utest = sv_number_input("Utestående aktier (miljoner)", value=sv_to_float(bef.get("Utestående aktier",0.0)))
            antal = sv_number_input("Antal aktier du äger", value=sv_to_float(bef.get("Antal aktier",0.0)), step=1)
            gav   = sv_number_input("GAV (SEK)", value=sv_to_float(bef.get("GAV (SEK)",0.0)))
            ps    = sv_number_input("P/S", value=sv_to_float(bef.get("P/S",0.0)))
            ps1   = sv_number_input("P/S Q1", value=sv_to_float(bef.get("P/S Q1",0.0)))
            ps2   = sv_number_input("P/S Q2", value=sv_to_float(bef.get("P/S Q2",0.0)))
            ps3   = sv_number_input("P/S Q3", value=sv_to_float(bef.get("P/S Q3",0.0)))
            ps4   = sv_number_input("P/S Q4", value=sv_to_float(bef.get("P/S Q4",0.0)))
        with c2:
            oms_idag = sv_number_input("Omsättning idag (miljoner)", value=sv_to_float(bef.get("Omsättning idag",0.0)))
            oms_next = sv_number_input("Omsättning nästa år (miljoner)", value=sv_to_float(bef.get("Omsättning nästa år",0.0)))

            st.caption("Uppdateras automatiskt vid spara: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%).")
        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # --- Dubblett-kontroll ---
        new_tkr = ticker
        cur_tkr = (bef.get("Ticker","") if not bef.empty else "")
        tnorm = df["Ticker"].astype(str).str.strip().str.upper()
        if bef.empty:
            if (tnorm == new_tkr).any():
                st.error(f"Tickern {new_tkr} finns redan. Välj den i listan för att redigera.")
                st.stop()
        else:
            if new_tkr != cur_tkr and (tnorm == new_tkr).any():
                st.error(f"Kan inte byta till {new_tkr} – den finns redan.")
                st.stop()

        ny = {
            "Ticker": new_tkr, "Utestående aktier": utest, "Antal aktier": antal,
            "GAV (SEK)": gav, "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        changed_manual = False
        if not bef.empty:
            before = {f: sv_to_float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: sv_to_float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            changed_manual = any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM)
        else:
            changed_manual = any(sv_to_float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM)

        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==cur_tkr, k] = v
            if new_tkr != cur_tkr:
                df.loc[df["Ticker"]==cur_tkr, "Ticker"] = new_tkr
        else:
            blank = {c: (0.0 if c in NUM_COLS else "") for c in FINAL_COLS}
            blank.update(ny)
            df = pd.concat([df, pd.DataFrame([blank])], ignore_index=True)

        if changed_manual:
            df.loc[df["Ticker"]==new_tkr, "Senast manuellt uppdaterad"] = now_stamp()

        data = yahoo_fields(new_tkr)
        if data.get("Bolagsnamn"): df.loc[df["Ticker"]==new_tkr, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[df["Ticker"]==new_tkr, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[df["Ticker"]==new_tkr, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data: df.loc[df["Ticker"]==new_tkr, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:   df.loc[df["Ticker"]==new_tkr, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        df = recalc(df)
        write_db(df)
        st.success("Sparat & uppdaterat.")
    return df

def view_analysis(df: pd.DataFrame):
    st.header("📈 Analys")
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", 0, max(0, len(labels)-1), st.session_state.analys_idx, 1)
    st.selectbox("Eller välj i lista", labels, index=st.session_state.analys_idx if labels else 0, key="analys_select")
    if len(vis)>0:
        r = vis.iloc[st.session_state.analys_idx]
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","Antal aktier","GAV (SEK)","Årlig utdelning","Senast manuellt uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

def fx_of(currency: str, user_rates: dict) -> float:
    if not currency: return 1.0
    return float(user_rates.get(str(currency).upper(), 1.0))

def view_portfolio(df: pd.DataFrame, user_rates: dict):
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: fx_of(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    port["Vinst/Förlust (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst/Förlust (%)"] = np.where(
        port["Anskaffningsvärde (SEK)"]>0,
        port["Vinst/Förlust (SEK)"]/port["Anskaffningsvärde (SEK)"]*100.0,
        0.0
    )
    total = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total>0, round(port["Värde (SEK)"]/total*100.0, 2), 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    tot_ansk = float(port["Anskaffningsvärde (SEK)"].sum())
    tot_pl = float(port["Vinst/Förlust (SEK)"].sum())
    tot_pl_pct = (tot_pl/tot_ansk*100.0) if tot_ansk>0 else 0.0

    st.markdown(f"**Totalt portföljvärde:** {round(total,2)} SEK")
    st.markdown(f"**Totalt anskaffningsvärde:** {round(tot_ansk,2)} SEK")
    st.markdown(f"**Orealiserad vinst/förlust:** {round(tot_pl,2)} SEK ({round(tot_pl_pct,2)} %)")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(port[[
        "Ticker","Bolagsnamn","Antal aktier","GAV (SEK)","Anskaffningsvärde (SEK)",
        "Aktuell kurs","Valuta","Växelkurs","Värde (SEK)",
        "Vinst/Förlust (SEK)","Vinst/Förlust (%)",
        "Årlig utdelning","Total årlig utdelning (SEK)","Andel (%)"
    ]], use_container_width=True)

def view_suggestions(df: pd.DataFrame, user_rates: dict):
    st.header("💡 Investeringsförslag")
    kapital = sv_number_input("Tillgängligt kapital (SEK)", 500.0, step=100.0, key="cap")
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)
    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    sort_läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    ps_filter = st.selectbox("Filtrera på P/S relativt P/S-snitt",
                             ["Alla", "P/S under snitt", "P/S över snitt"], index=0)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()

    if ps_filter == "P/S under snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] < base["P/S-snitt"])].copy()
    elif ps_filter == "P/S över snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] > base["P/S-snitt"])].copy()

    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if sort_läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    if "forslags_index" not in st.session_state: st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"): st.session_state.forslags_index = max(0, st.session_state.forslags_index-1)
    with col_next:
        if st.button("➡️ Nästa förslag"): st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index+1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")

    rad = base.iloc[st.session_state.forslags_index]

    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: fx_of(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    tot_port = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = fx_of(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty: nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / tot_port) * 100.0, 2) if tot_port > 0 else 0.0
    ny_andel  = round((ny_total   / tot_port) * 100.0, 2) if tot_port > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Nuvarande P/S (TTM):** {round(rad.get('P/S', 0.0), 2)}
- **P/S-snitt (Q1–Q4):** {round(rad.get('P/S-snitt', 0.0), 2)}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 3 år" else ""}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
""")

def sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # tyst auto-uppdatering vid start
    auto_update_rates_if_needed()

    saved = get_saved_rates()
    usd = sv_number_input("USD → SEK", value=float(saved.get("USD", 10.0)), key="fx_usd")
    nok = sv_number_input("NOK → SEK", value=float(saved.get("NOK", 1.0)),  key="fx_nok")
    cad = sv_number_input("CAD → SEK", value=float(saved.get("CAD", 7.5)),  key="fx_cad")
    eur = sv_number_input("EUR → SEK", value=float(saved.get("EUR", 11.0)), key="fx_eur")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara valutakurser"):
            save_rates(user_rates)
            st.session_state["rates_nonce"] = st.session_state.get("rates_nonce",0)+1
            st.sidebar.success("Valutakurser sparade.")
    with c2:
        if st.button("🌐 Hämta (Yahoo)"):
            live = fetch_live_fx()
            if live:
                merged = get_saved_rates()
                merged.update(live)
                save_rates(merged)
                st.session_state["rates_nonce"] = st.session_state.get("rates_nonce",0)+1
                st.sidebar.success("Valutakurser uppdaterade.")
                st.rerun()
            else:
                st.sidebar.warning("Kunde inte hämta live-kurser.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    rates = sidebar_rates()

    df = read_db()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        write_db(df)

    df = ensure_cols(df)

    # Massuppdateringsknapp i sidopanel
    df = massupdate_sidebar(df)

    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])
    if meny == "Analys":
        view_analysis(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = view_edit(df)
    elif meny == "Investeringsförslag":
        df = recalc(df)
        view_suggestions(df, rates)
    elif meny == "Portfölj":
        df = recalc(df)
        view_portfolio(df, rates)

if __name__ == "__main__":
    main()
