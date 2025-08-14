import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# === Google Sheets-konfiguration ===
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# 💱 Standardväxel (till SEK) – du kan ändra dessa i sidomenyn
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

# ---------- Google helpers ----------
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---------- Lokal cache i session_state ----------
def load_df(force: bool = False) -> pd.DataFrame:
    """Hämta df från cache, eller Google om ingen cache/force."""
    if not force and "df_cache" in st.session_state:
        return st.session_state["df_cache"].copy()
    try:
        df = hamta_data()
    except Exception:
        st.warning("Kunde inte läsa Google Sheet. Visar lokalt cachelagrat data om det finns.")
        df = st.session_state.get("df_cache", pd.DataFrame()).copy()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    st.session_state["df_cache"] = df.copy()
    return df

def persist_df(df: pd.DataFrame):
    """Spara df både lokalt och till Google (om möjligt)."""
    st.session_state["df_cache"] = df.copy()
    try:
        spara_data(df)
        st.success("✅ Sparat till Google Sheets.")
    except Exception:
        st.warning("⚠️ Kunde inte spara till Google Sheets. Ändringarna finns kvar lokalt tills nästa lyckade sparning.")

# ---------- Kolumnstöd ----------
KOLUMNER = [
    "Ticker","Bolagsnamn","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Valuta","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=KOLUMNER)
    for k in KOLUMNER:
        if k not in df.columns:
            # numeriska kolumner
            if any(x in k.lower() for x in ["p/s","omsättning","riktkurs","aktier","utdelning","kurs","cagr"]):
                df[k] = 0.0
            else:
                df[k] = ""
    # Ta bort gamla 2026/2027/2028-kolumner om de skulle råka finnas
    for old in ["Riktkurs 2026","Riktkurs 2027","Riktkurs 2028","Riktkurs om idag"]:
        if old in df.columns:
            df.drop(columns=[old], inplace=True)
    # Ordna kolumnordning
    df = df[[c for c in KOLUMNER]]
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in KOLUMNER if c not in ["Ticker","Bolagsnamn","Valuta"]]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # text
    for c in ["Ticker","Bolagsnamn","Valuta"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df

# ---------- Yahoo helpers ----------
def hamta_kurs_valuta_namn_utdelning(ticker: str):
    """
    Hämtar: pris, valuta, bolagsnamn, årlig utdelning (per aktie).
    Faller tillbaka till None/0.0 vid fel.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        namn = info.get("shortName") or info.get("longName") or ""
        utd = info.get("dividendRate", 0.0)  # årsutdelning per aktie
        return pris, valuta, namn, (udt := 0.0 if utd is None else float(udt) if isinstance((udt := utd), (int,float)) else 0.0)
    except Exception:
        return None, "USD", "", 0.0

def hamta_cagr_5ar(ticker: str):
    """
    Försök räkna CAGR 5 år på intäkt (Revenue) från Yahoo.
    yfinance ger ofta 4–5 års 'earnings' (Revenue). Blir det <2 datapunkter returneras None.
    CAGR = (Rev_slut / Rev_start)**(1/n) - 1
    """
    try:
        t = yf.Ticker(ticker)
        # Äldre yfinance: t.earnings (års-DF med Revenue/Earnings)
        df_earn = getattr(t, "earnings", None)
        if df_earn is None or df_earn.empty or "Revenue" not in df_earn.columns:
            # Nyare yfinance kan ha t.get_earnings()
            try:
                df_earn = t.get_earnings(freq="yearly")
            except Exception:
                df_earn = None

        if df_earn is None or len(df_earn) < 2:
            return None

        # Sortera efter år (index) om möjligt
        try:
            df_e = df_earn.copy()
            df_e = df_e.sort_index()
            revenues = df_e["Revenue"].dropna().astype(float)
        except Exception:
            return None

        if len(revenues) < 2:
            return None

        rev_start = float(revenues.iloc[0])
        rev_slut = float(revenues.iloc[-1])
        n_years = max(1, len(revenues) - 1)  # t.ex. 4 punkter => 3 års intervall
        if rev_start <= 0:
            return None

        cagr = (rev_slut / rev_start) ** (1.0 / n_years) - 1.0
        return cagr * 100.0  # i %
    except Exception:
        return None

# ---------- Beräkningar ----------
def clampad_tillväxt(cagr_procent: float) -> float:
    """
    Regler:
      - om CAGR > 100% ⇒ använd 50% (0.50)
      - om CAGR < 0%   ⇒ använd 2%  (0.02)
      - annars CAGR/100
    """
    if cagr_procent is None:
        return None
    if cagr_procent > 100:
        return 0.50
    if cagr_procent < 0:
        return 0.02
    return float(cagr_procent) / 100.0

def räkna_omsättning_framåt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar 'Omsättning om 2 år' och 'Omsättning om 3 år' från 'Omsättning nästa år' och 'CAGR 5 år (%)'.
    (Miljoner in, miljoner ut.)
    """
    for i, rad in df.iterrows():
        nxt = float(rad.get("Omsättning nästa år", 0.0))
        cagr = rad.get("CAGR 5 år (%)", 0.0)
        g = clampad_tillväxt(cagr)
        if nxt > 0 and g is not None:
            oms2 = nxt * (1.0 + g)
            oms3 = oms2 * (1.0 + g)
            df.at[i, "Omsättning om 2 år"] = round(oms2, 2)
            df.at[i, "Omsättning om 3 år"] = round(oms3, 2)
    return df

def uppdatera_ps_snitt_och_riktkurser(df: pd.DataFrame) -> pd.DataFrame:
    """
    P/S-snitt på Q1–Q4 där värde > 0.
    Riktkurser i egen valuta: (Omsättning * P/S-snitt) / Utestående aktier
    """
    for i, rad in df.iterrows():
        ps_vals = [rad.get("P/S Q1",0), rad.get("P/S Q2",0), rad.get("P/S Q3",0), rad.get("P/S Q4",0)]
        ps_vals = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_vals), 2) if ps_vals else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        uts = float(rad.get("Utestående aktier", 0.0))
        if uts > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]  = round((float(rad.get("Omsättning idag",0.0))  * ps_snitt) / uts, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år",0.0)) * ps_snitt) / uts, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(rad.get("Omsättning om 2 år",0.0)) * ps_snitt) / uts, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(rad.get("Omsättning om 3 år",0.0)) * ps_snitt) / uts, 2)
        else:
            # om uts=0 eller ps_snitt=0 -> nollställ riktkurser
            for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i, k] = 0.0
    return df

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    df = räkna_omsättning_framåt(df)
    df = uppdatera_ps_snitt_och_riktkurser(df)
    return df

# ---------- Form: Lägg till / uppdatera bolag ----------
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Ordna lista (Bolagsnamn (Ticker)), bläddringsindex i session
    vis_lista = []
    for _, r in df.iterrows():
        t = (r.get("Ticker","") or "").strip()
        b = (r.get("Bolagsnamn","") or "").strip()
        if t:
            vis_lista.append(f"{b} ({t})" if b else t)
    vis_lista = sorted(vis_lista)

    if "form_idx" not in st.session_state:
        st.session_state.form_idx = 0

    # Rullista + bläddring
    colA,colB,colC = st.columns([4,1,1])
    with colA:
        valt = st.selectbox("Välj bolag (eller lämna tom för nytt)", [""] + vis_lista, index=0)
        if valt and "(" in valt and ")" in valt:
            ticker_vald = valt.split("(")[-1].split(")")[0]
            # uppdatera index för bläddring
            try:
                st.session_state.form_idx = vis_lista.index(valt)
            except Exception:
                pass
        else:
            ticker_vald = ""

    with colB:
        if st.button("⬅️ Föregående"):
            st.session_state.form_idx = max(0, st.session_state.form_idx - 1)
            if vis_lista:
                st.experimental_rerun()
    with colC:
        if st.button("➡️ Nästa"):
            st.session_state.form_idx = min(max(0,len(vis_lista)-1), st.session_state.form_idx + 1)
            if vis_lista:
                st.experimental_rerun()

    if (not valt) and vis_lista and st.session_state.form_idx < len(vis_lista):
        valt = vis_lista[st.session_state.form_idx]
        ticker_vald = valt.split("(")[-1].split(")")[0]

    if ticker_vald:
        bef = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        bef = pd.Series(dtype=object)

    with st.form("form_bolag"):
        st.caption(f"Post {st.session_state.form_idx+1 if vis_lista else 1} / {len(vis_lista) if vis_lista else 1}")
        # Manuella fält
        ticker = st.text_input("Ticker", value=bef.get("Ticker","") if not bef.empty else "").upper()
        utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)

        ps    = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
        ps1   = st.number_input("P/S Q1",value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
        ps2   = st.number_input("P/S Q2",value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
        ps3   = st.number_input("P/S Q3",value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
        ps4   = st.number_input("P/S Q4",value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
        oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

        antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara & hämta fakta")

    if sparaknapp:
        if not ticker:
            st.error("Ticker krävs.")
            return df

        # skapa/uppdatera rad
        ny = {
            "Ticker": ticker,
            "Bolagsnamn": bef.get("Bolagsnamn","") if not bef.empty else "",
            "Utestående aktier": utest,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
            "Omsättning om 2 år": bef.get("Omsättning om 2 år",0.0) if not bef.empty else 0.0,
            "Omsättning om 3 år": bef.get("Omsättning om 3 år",0.0) if not bef.empty else 0.0,
            "Riktkurs idag": bef.get("Riktkurs idag",0.0) if not bef.empty else 0.0,
            "Riktkurs om 1 år": bef.get("Riktkurs om 1 år",0.0) if not bef.empty else 0.0,
            "Riktkurs om 2 år": bef.get("Riktkurs om 2 år",0.0) if not bef.empty else 0.0,
            "Riktkurs om 3 år": bef.get("Riktkurs om 3 år",0.0) if not bef.empty else 0.0,
            "Antal aktier": antal,
            "Valuta": bef.get("Valuta","USD") if not bef.empty else "USD",
            "Årlig utdelning": bef.get("Årlig utdelning",0.0) if not bef.empty else 0.0,
            "Aktuell kurs": bef.get("Aktuell kurs",0.0) if not bef.empty else 0.0,
            "CAGR 5 år (%)": bef.get("CAGR 5 år (%)",0.0) if not bef.empty else 0.0,
            "P/S-snitt": bef.get("P/S-snitt",0.0) if not bef.empty else 0.0,
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny.keys()] = ny.values()
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)

        # Hämta Yahoo-fakta för denna ticker
        pris, valuta, namn, utd = hamta_kurs_valuta_namn_utdelning(ticker)
        if pris is not None:
            df.loc[df["Ticker"] == ticker, "Aktuell kurs"] = round(pris, 2)
        if valuta:
            df.loc[df["Ticker"] == ticker, "Valuta"] = valuta
        if namn:
            df.loc[df["Ticker"] == ticker, "Bolagsnamn"] = namn
        if utd is not None:
            df.loc[df["Ticker"] == ticker, "Årlig utdelning"] = float(udt := (udt if (udt:=utd) is not None else 0.0))

        # CAGR 5 år
        cagr = hamta_cagr_5ar(ticker)
        if cagr is not None:
            df.loc[df["Ticker"] == ticker, "CAGR 5 år (%)"] = round(cagr, 2)

        # Räkna fram omsättning år 2 & 3 + riktkurser
        df = uppdatera_berakningar(df)

        persist_df(df)  # spara cache + försök till Google
        st.success(f"{ticker} sparat & uppdaterat.")

    return df

# ---------- Portfölj ----------
def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")
    d = df[df["Antal aktier"] > 0].copy()
    if d.empty:
        st.info("Du äger inga aktier.")
        return
    d["Växelkurs"] = d["Valuta"].map(valutakurser).fillna(1.0)
    d["Värde (SEK)"] = d["Antal aktier"] * d["Aktuell kurs"] * d["Växelkurs"]
    d["Andel (%)"] = round(d["Värde (SEK)"] / d["Värde (SEK)"].sum() * 100, 2)
    d["Total årlig utdelning"] = d["Antal aktier"] * d["Årlig utdelning"] * d["Växelkurs"]

    total_värde = d["Värde (SEK)"].sum()
    total_utd = d["Total årlig utdelning"].sum()

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utd, 2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(total_utd/12,2)} SEK")

    st.dataframe(d[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning"]],
                 use_container_width=True)

# ---------- Analys (visar vald rad + hela tabellen) ----------
def analysvy(df: pd.DataFrame):
    st.subheader("📈 Analys")

    # lista för bläddring
    lista = sorted([(r.get("Bolagsnamn",""), r.get("Ticker","")) for _, r in df.iterrows() if r.get("Ticker","")])
    vis = [f"{n} ({t})" if n else t for (n,t) in lista]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    colA,colB,colC = st.columns([4,1,1])
    with colA:
        valt = st.selectbox("Välj bolag för fokusvisning", vis if vis else [""], index=st.session_state.analys_idx if vis else 0)
        if vis:
            st.session_state.analys_idx = vis.index(valt)
    with colB:
        if st.button("⬅️ Föregående (Analys)"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
            st.experimental_rerun()
    with colC:
        if st.button("➡️ Nästa (Analys)"):
            st.session_state.analys_idx = min(len(vis)-1, st.session_state.analys_idx + 1)
            st.experimental_rerun()

    if vis:
        st.caption(f"Post {st.session_state.analys_idx+1}/{len(vis)}")
        tick = vis[st.session_state.analys_idx].split("(")[-1].split(")")[0]
        rad = df[df["Ticker"] == tick]
        st.write("**Vald post:**")
        st.dataframe(rad, use_container_width=True)

    st.write("**Hela databasen:**")
    st.dataframe(df, use_container_width=True)

    # Massuppdatera från Yahoo (pris/valuta/namn/utd + CAGR) – 1s delay
    if st.button("🔄 Uppdatera ALLA från Yahoo"):
        miss = []
        ok = 0
        total = len(df)
        prog = st.progress(0)
        msg = st.empty()
        for i, (idx, r) in enumerate(df.iterrows(), start=1):
            t = (r.get("Ticker","") or "").strip()
            msg.text(f"Uppdaterar {i}/{total}: {t}")
            if not t:
                miss.append("Tom Ticker")
                prog.progress(i/total); time.sleep(1); continue

            pris, valuta, namn, utd = hamta_kurs_valuta_namn_utdelning(t)
            if pris is not None:
                df.at[idx, "Aktuell kurs"] = round(float(pris),2)
            if valuta:
                df.at[idx, "Valuta"] = valuta
            if namn:
                df.at[idx, "Bolagsnamn"] = namn
            if utd is not None:
                df.at[idx, "Årlig utdelning"] = float(udt := (udt if (udt:=utd) is not None else 0.0))

            cagr = hamta_cagr_5ar(t)
            if cagr is not None:
                df.at[idx, "CAGR 5 år (%)"] = round(float(cagr),2)

            ok += 1
            prog.progress(i/total)
            time.sleep(1)

        # Räkna om och spara
        df = uppdatera_berakningar(df)
        persist_df(df)
        st.success(f"Uppdaterade {ok}/{total} poster.")

def visa_investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)

    horisonter = ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]
    riktkurs_val = st.selectbox("Använd riktkurs:", horisonter, index=1)

    visa_endast_port = st.checkbox("Endast portföljens innehav", value=False)

    läge = st.radio("Sorteringsläge", ["Störst uppsida", "Närmast riktkurs"], horizontal=True)
    riktning = st.radio("Visa", ["Båda","Under riktkurs","Över riktkurs"], index=0, horizontal=True)

    d = df.copy()
    # Filtrera portfölj om valt
    if visa_endast_port:
        tickers_port = set(d[d["Antal aktier"] > 0]["Ticker"].astype(str))
        d = d[d["Ticker"].astype(str).isin(tickers_port)]

    # Kräver positiv aktuell kurs och riktkurs
    d = d[(d["Aktuell kurs"] > 0) & (d[riktkurs_val] > 0)].copy()
    if d.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkna uppsida och avstånd
    d["Uppsida (%)"] = ((d[riktkurs_val] - d["Aktuell kurs"]) / d["Aktuell kurs"]) * 100.0
    d["Dist till riktkurs (%)"] = (abs(d[riktkurs_val] - d["Aktuell kurs"]) / d[riktkurs_val]) * 100.0

    # Riktning
    if riktning == "Under riktkurs":
        d = d[d["Aktuell kurs"] < d[riktkurs_val]]
    elif riktning == "Över riktkurs":
        d = d[d["Aktuell kurs"] >= d[riktkurs_val]]

    if d.empty:
        st.info("Inga bolag kvar efter filter.")
        return

    # Sortering
    if läge == "Störst uppsida":
        d = d.sort_values("Uppsida (%)", ascending=False).reset_index(drop=True)
    else:
        d = d.sort_values("Dist till riktkurs (%)", ascending=True).reset_index(drop=True)

    # Indexhantering för bläddring
    filt_sig = f"{riktkurs_val}|{läge}|{riktning}|{visa_endast_port}"
    if "forslag_sig" not in st.session_state or st.session_state.forslag_sig != filt_sig:
        st.session_state.forslag_sig = filt_sig
        st.session_state.forslags_index = 0

    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0

    idx = max(0, min(st.session_state.forslags_index, len(d)-1))
    st.session_state.forslags_index = idx
    rad = d.iloc[idx]

    # Antal att köpa (kapital i SEK / kurs i SEK)
    vx = valutakurser.get(str(rad["Valuta"]).upper(), 1.0)
    kurs_sek = rad["Aktuell kurs"] * vx
    köp_antal = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0

    # Portföljandel
    port = df[df["Antal aktier"] > 0].copy()
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Valuta"].map(valutakurser).fillna(1.0)
    portvärde = port["Värde (SEK)"].sum()
    nu_innehav = port[port["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    efter = nu_innehav + (köp_antal * kurs_sek)
    nu_andel = round((nu_innehav/portvärde)*100,2) if portvärde>0 else 0.0
    efter_andel = round((efter/portvärde)*100,2) if portvärde>0 else 0.0

    # Presentera — visa alla fyra riktkurser, fetmarkera vald rad
    def bold_if(label):
        return f"**{label}**" if label == riktkurs_val else label

    st.markdown(f"#### {rad['Bolagsnamn']} ({rad['Ticker']}) — {idx+1}/{len(d)}")
    st.markdown(f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- {bold_if('Riktkurs idag')}: {round(rad['Riktkurs idag'],2)} {rad['Valuta']}
- {bold_if('Riktkurs om 1 år')}: {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}
- {bold_if('Riktkurs om 2 år')}: {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}
- {bold_if('Riktkurs om 3 år')}: {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}
- **Uppsida enligt val ({riktkurs_val}):** {round(rad['Uppsida (%)'],2)}%
- **Köpförslag (kapital {int(kapital_sek)} SEK):** {köp_antal} st
- **Nuvarande andel av portfölj:** {nu_andel}%
- **Andel efter köp:** {efter_andel}%
    """)

    c1,c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with c2:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(d)-1, st.session_state.forslags_index + 1)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidomeny: valutakurser & Google-reload
    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=STANDARD_VALUTAKURSER["USD"], step=0.01, key="vx_usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=STANDARD_VALUTAKURSER["NOK"], step=0.01, key="vx_nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=STANDARD_VALUTAKURSER["CAD"], step=0.01, key="vx_cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=STANDARD_VALUTAKURSER["EUR"], step=0.01, key="vx_eur"),
    }

    if st.sidebar.button("🔁 Ladda om från Google"):
        load_df(force=True)
        st.experimental_rerun()

    df = load_df()

    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        # redan sparat i persist_df i funktionen
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df.copy())
        visa_investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df.copy())
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
