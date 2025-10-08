# app.py — DIN GAMLA STRUKTUR + exakt det du bett om
from __future__ import annotations
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import gspread

# ---- Google Sheets via DIN GAMLA setup ----
from google.oauth2.service_account import Credentials

# ---- yfinance (valuta + bolagsdata); tolerera att det ev saknas ----
try:
    import yfinance as yf
    _YF_OK = True
except Exception:
    _YF_OK = False

# =========================
# KONFIG
# =========================
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# =========================
# HJÄLPARE
# =========================
def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err

def now_stamp():
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def skapa_rates_sheet_if_missing():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        ws.update([["Valuta","Kurs"]])
        return ws

# =========================
# DATA I/O
# =========================
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad", "Senast auto uppdaterad"
]

def hamta_data():
    sheet = skapa_koppling()
    data = _with_backoff(sheet.get_all_records)
    df = pd.DataFrame(data)
    # säkerställ schema
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    # typer
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad"]:
        df[c] = df[c].astype(str)
    return df

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, [df.columns.values.tolist()] + df.astype(object).where(pd.notnull(df), "").values.tolist())

# =========================
# VALUTA (SPARA/LÄSA + LIVE-KNAPP)
# =========================
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = skapa_rates_sheet_if_missing()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except:
            pass
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = skapa_rates_sheet_if_missing()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def _hamta_live_fx():
    """Hämta USD/NOK/CAD/EUR → SEK via Yahoo; returnerar dict eller None."""
    if not _YF_OK:
        return None
    pairs = {"USD": "USDSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X", "EUR": "EURSEK=X"}
    out = {}
    try:
        tickers = " ".join(pairs.values())
        data = yf.download(tickers=tickers, period="1d", interval="1d", progress=False, group_by='ticker', threads=True)
        for k, ysym in pairs.items():
            try:
                # data kan vara multi-index om flera tickers
                if isinstance(data, pd.DataFrame) and ysym in data.columns.get_level_values(0):
                    px = data[ysym]["Close"].dropna()
                    if not px.empty:
                        out[k] = float(px.iloc[-1])
                else:
                    # Single ticker fallback (om bara en)
                    px = data["Close"].dropna()
                    if not px.empty:
                        out[k] = float(px.iloc[-1])
            except Exception:
                pass
        if out:
            out["SEK"] = 1.0
            return out
    except Exception:
        return None
    return None

# =========================
# YAHOO BOLAGSFÄLT (enkel, robust)
# =========================
def beräkna_cagr_från_finansiella(tkr: "yf.Ticker") -> float:
    try:
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def hamta_yahoo_fält(ticker: str) -> dict:
    out = {
        "Bolagsnamn": "", "Aktuell kurs": 0.0, "Valuta": "USD",
        "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0,
        "Utestående aktier": 0.0, "P/S": 0.0,
    }
    if not _YF_OK:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate", None)
        if isinstance(div_rate, (int,float)):
            out["Årlig utdelning"] = float(div_rate)

        shares = info.get("sharesOutstanding", None)
        if isinstance(shares, (int,float)) and shares > 0:
            out["Utestående aktier"] = float(shares) / 1e6  # → miljoner

        ps_ttm = info.get("priceToSalesTrailing12Months", None)
        if isinstance(ps_ttm, (int,float)) and ps_ttm > 0:
            out["P/S"] = float(ps_ttm)

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

    except Exception:
        pass
    return out

# =========================
# BERÄKNINGAR
# =========================
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    df = df.copy()
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        df.at[i, "P/S-snitt"] = round(np.mean(ps_clean), 2) if ps_clean else 0.0

        # CAGR clamp
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)

        # Riktkurser (oms i miljoner, utest aktier i miljoner → pris/aktie)
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        ps_use = float(df.at[i, "P/S-snitt"]) or float(rad.get("P/S", 0.0))
        if aktier_ut > 0 and ps_use > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_use) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_use) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_use) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_use) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# =========================
# UI HJÄLP-ETIKETTER
# =========================
def _lbl_with_ts(base: str, df_row: pd.Series, typ: str) -> str:
    """
    typ: 'man' (visa Senast manuellt uppdaterad)
         'auto' (visa Senast auto uppdaterad)
    """
    if df_row is None or df_row.empty:
        return base
    if typ == "man":
        ts = df_row.get("Senast manuellt uppdaterad", "") or "—"
        return f"{base}  [{ts}]"
    if typ == "auto":
        ts = df_row.get("Senast auto uppdaterad", "") or "—"
        return f"{base}  [{ts}]"
    return base

# =========================
# VYER
# =========================
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: user_rates.get(v, 1.0))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
                                index=1)

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: user_rates.get(v, 1.0))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = user_rates.get(rad["Valuta"], 1.0)
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
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
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

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b = st.columns([1,1])
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
                "Senast manuellt uppdaterad","Senast auto uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

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

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
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
            utest = st.number_input(_lbl_with_ts("Utestående aktier (miljoner)", bef, "auto"),
                                    value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input(_lbl_with_ts("P/S (TTM)", bef, "man"),
                                  value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input(_lbl_with_ts("P/S Q1", bef, "man"),
                                  value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input(_lbl_with_ts("P/S Q2", bef, "man"),
                                  value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input(_lbl_with_ts("P/S Q3", bef, "man"),
                                  value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input(_lbl_with_ts("P/S Q4", bef, "man"),
                                  value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input(_lbl_with_ts("Omsättning idag (miljoner)", bef, "man"),
                                        value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input(_lbl_with_ts("Omsättning nästa år (miljoner)", bef, "man"),
                                        value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.caption("Senast manuellt uppdaterad: " + (bef.get("Senast manuellt uppdaterad","") or "—"))
            st.caption("Senast auto uppdaterad: " + (bef.get("Senast auto uppdaterad","") or "—"))

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Automatisk hämtning från Yahoo (enkelt)
        data = hamta_yahoo_fält(ticker)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S"]:
            if k in data:
                df.loc[df["Ticker"]==ticker, k] = data[k]
        if float(data.get("Utestående aktier",0) or 0) > 0:
            df.loc[df["Ticker"]==ticker, "Utestående aktier"] = float(data["Utestående aktier"])

        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = now_stamp()

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")
        st.rerun()

    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]], use_container_width=True)

    return df

# =========================
# SIDOPANEL – VALUTAKNAPP + STATUS
# =========================
def valutakurser_sidebar() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, format="%.4f")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("💾 Spara"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
            st.session_state["recalc_after_rates"] = True
    with c2:
        if st.button("↻ Läs"):
            st.cache_data.clear()
            st.rerun()
    with c3:
        if st.button("🌐 Live"):
            live = _hamta_live_fx()
            if live:
                spara_valutakurser(live)
                st.sidebar.success("Live-kurser hämtade & sparade.")
                st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
                st.session_state["recalc_after_rates"] = True
                st.rerun()
            else:
                st.sidebar.warning("Kunde inte hämta live-kurser just nu.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

# =========================
# MAIN
# =========================
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Valutor först (använder DIN rates-sheet)
    user_rates = valutakurser_sidebar()

    # Läs data
    df = hamta_data()

    # Om vi nyss sparade/hämtade kurser → räkna om & spara
    if st.session_state.get("recalc_after_rates", False):
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.session_state["recalc_after_rates"] = False

    # Global massuppdatering från Yahoo (valfritt)
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        total = len(df)
        misslyckade = []
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            if data.get("Bolagsnamn"): df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else: failed_fields.append("Bolagsnamn")

            if data.get("Aktuell kurs", 0) > 0: df.at[i, "Aktuell kurs"] = data["Aktuell kurs"]
            else: failed_fields.append("Aktuell kurs")

            if data.get("Valuta"): df.at[i, "Valuta"] = data["Valuta"]
            else: failed_fields.append("Valuta")

            if "Årlig utdelning" in data:
                df.at[i, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            else:
                failed_fields.append("Årlig utdelning")

            if "CAGR 5 år (%)" in data:
                df.at[i, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)
            else:
                failed_fields.append("CAGR 5 år (%)")

            if float(data.get("Utestående aktier", 0) or 0) > 0:
                df.at[i, "Utestående aktier"] = float(data["Utestående aktier"])

            if "P/S" in data and float(data["P/S"])>0:
                df.at[i, "P/S"] = float(data["P/S"])

            df.at[i, "Senast auto uppdaterad"] = now_stamp()

            time.sleep(0.3)
            bar.progress((i+1)/max(1,total))

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält saknades:\n" + "\n".join(misslyckade))

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
