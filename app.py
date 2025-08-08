import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# --- Google Sheets ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- Valutakurser (SEK per 1 enhet) – standardvärden kan ändras i sidomenyn ---
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# --- Kolumnuppsättning (EXAKT enligt din lista) ---
KOLUMNER = [
    "Ticker",
    "Bolagsnamn",
    "Utestående aktier",
    "P/S",
    "P/S Q1",
    "P/S Q2",
    "P/S Q3",
    "P/S Q4",
    "Omsättning idag",
    "Omsättning nästa år",
    "Omsättning om 2 år",
    "Omsättning om 3 år",
    "Riktkurs idag",
    "Riktkurs om 1 år",
    "Riktkurs om 2 år",
    "Riktkurs om 3 år",
    "Antal aktier",
    "Valuta",
    "Årlig utdelning",
    "Aktuell kurs",
    "CAGR 5 år (%)",
    "P/S-snitt",
]

NUMERISKA = [
    "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KOLUMNER)
    for kol in KOLUMNER:
        if kol not in df.columns:
            df[kol] = "" if kol not in NUMERISKA else 0.0
    df = df[KOLUMNER]
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    for kol in NUMERISKA:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    for kol in df.columns:
        if kol not in NUMERISKA:
            df[kol] = df[kol].astype(str)
    return df

def hamta_vaxelkurs_for_rad(row, valutakurser: dict) -> float:
    val = str(row.get("Valuta", "") or "").upper()
    return float(valutakurser.get(val, 1.0))

# --- Yahoo helpers ---

def hamta_kurs_valuta_namn_utdelning(ticker: str):
    """
    Hämtar aktuell kurs, valuta, bolagsnamn och årlig utdelning (om tillgänglig).
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", None)
        namn = info.get("shortName", None) or info.get("longName", None) or ""
        utd = info.get("dividendRate", None)
        if utd is None:
            utd = info.get("trailingAnnualDividendRate", 0.0)
        try:
            utd = float(utd or 0.0)
        except Exception:
            utd = 0.0
        return pris, valuta, namn, utd
    except Exception:
        return None, None, "", 0.0

def hamta_cagr_5ar(ticker: str) -> float:
    """
    Beräknar CAGR för intäkter över ~5 år från Yahoo (income statement).
    Använder 'Total Revenue' om möjligt.
    """
    try:
        t = yf.Ticker(ticker)
        df_inc = None
        try:
            df_inc = t.income_stmt
        except Exception:
            pass
        if df_inc is None or df_inc.empty:
            try:
                df_inc = t.financials
            except Exception:
                df_inc = None

        if df_inc is None or df_inc.empty:
            return 0.0

        row_candidates = ["Total Revenue", "TotalRevenue", "Total revenue", "Revenue"]
        rev_series = None
        for cand in row_candidates:
            if cand in df_inc.index:
                rev_series = df_inc.loc[cand]
                break

        if rev_series is None or rev_series.empty:
            return 0.0

        vals = rev_series.dropna().astype(float)
        if len(vals) < 2:
            return 0.0

        earliest = float(vals.iloc[-1])
        latest   = float(vals.iloc[0])
        periods  = len(vals) - 1
        if earliest <= 0 or latest <= 0 or periods <= 0:
            return 0.0

        cagr = (latest / earliest) ** (1.0 / periods) - 1.0
        return float(cagr * 100.0)
    except Exception:
        return 0.0

# --- Beräkningar ---

def justera_cagr(cagr_procent: float) -> float:
    """
    Begränsning: >100% -> 50% ; <0% -> 2%
    Returnerar decimal (0.50 etc).
    """
    if cagr_procent > 100.0:
        return 0.50
    if cagr_procent < 0.0:
        return 0.02
    return float(cagr_procent) / 100.0

def räkna_omsättning_framåt(oms_next_year: float, cagr_pct: float):
    """
    Från 'Omsättning nästa år' -> räkna fram 'om 2 år' och 'om 3 år' med justerad CAGR.
    """
    g = justera_cagr(cagr_pct)
    if oms_next_year <= 0:
        return 0.0, 0.0
    oms2 = oms_next_year * (1.0 + g)
    oms3 = oms2 * (1.0 + g)
    return round(oms2, 2), round(oms3, 2)

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    """
    - P/S-snitt = snitt av P/S Q1-4 > 0
    - Riktkurser (idag/1/2/3 år) = Oms * P/S-snitt / Utestående aktier
    """
    df = df.copy()
    for i, rad in df.iterrows():
        q = [rad.get("P/S Q1", 0.0), rad.get("P/S Q2", 0.0),
             rad.get("P/S Q3", 0.0), rad.get("P/S Q4", 0.0)]
        psvals = []
        for x in q:
            try:
                xv = float(x)
                if xv > 0:
                    psvals.append(xv)
            except Exception:
                pass
        ps_snitt = round(float(np.mean(psvals)), 2) if psvals else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        utest = float(rad.get("Utestående aktier", 0.0))
        if utest > 0 and ps_snitt > 0:
            oms_idag = float(rad.get("Omsättning idag", 0.0))
            oms_1    = float(rad.get("Omsättning nästa år", 0.0))
            oms_2    = float(rad.get("Omsättning om 2 år", 0.0))
            oms_3    = float(rad.get("Omsättning om 3 år", 0.0))

            df.at[i, "Riktkurs idag"]    = round((oms_idag * ps_snitt) / utest, 2) if oms_idag > 0 else 0.0
            df.at[i, "Riktkurs om 1 år"] = round((oms_1   * ps_snitt) / utest, 2) if oms_1   > 0 else 0.0
            df.at[i, "Riktkurs om 2 år"] = round((oms_2   * ps_snitt) / utest, 2) if oms_2   > 0 else 0.0
            df.at[i, "Riktkurs om 3 år"] = round((oms_3   * ps_snitt) / utest, 2) if oms_3   > 0 else 0.0
        else:
            for kol in ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]:
                df.at[i, kol] = 0.0
    return df

def massuppdatera_yahoo(df: pd.DataFrame, paus_s: float = 1.0) -> pd.DataFrame:
    """
    För varje ticker: hämta kurs/valuta/namn/utdelning + CAGR, räkna om omsättning 2/3 år och riktkurser.
    """
    df = df.copy()
    misslyckade = []
    status = st.empty()
    bar = st.progress(0.0)
    total = len(df) if len(df) > 0 else 1

    for i, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        status.text(f"Uppdaterar {i+1}/{total}: {ticker}")
        try:
            pris, valuta, namn, utd = hamta_kurs_valuta_namn_utdelning(ticker)
            if pris is not None:
                df.at[i, "Aktuell kurs"] = round(float(pris), 2)
            if valuta:
                df.at[i, "Valuta"] = str(valuta).upper()
            if namn:
                df.at[i, "Bolagsnamn"] = namn
            df.at[i, "Årlig utdelning"] = float(utd or 0.0)

            cagr = hamta_cagr_5ar(ticker)
            df.at[i, "CAGR 5 år (%)"] = round(float(cagr), 2)

            oms_1 = float(df.at[i, "Omsättning nästa år"])
            oms2, oms3 = räkna_omsättning_framåt(oms_1, cagr)
            df.at[i, "Omsättning om 2 år"] = oms2
            df.at[i, "Omsättning om 3 år"] = oms3

        except Exception:
            misslyckade.append(ticker)

        bar.progress((i+1) / total)
        time.sleep(paus_s)

    df = konvertera_typer(df)
    df = uppdatera_berakningar(df)

    if misslyckade:
        st.warning("Kunde inte uppdatera följande tickers:\n" + ", ".join(misslyckade))
    else:
        st.success("Massuppdatering klar.")

    return df

# --- Lägg till / uppdatera bolag ---

def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # 🔁 Nollställ bläddring
    if st.button("🔁 Nollställ bläddring", key="reset_edit"):
        st.session_state.edit_idx = 0
        st.experimental_rerun()

    if "edit_idx" not in st.session_state:
        st.session_state.edit_idx = 0

    options = df["Ticker"].astype(str).tolist() if not df.empty else []
    valt_ticker = st.selectbox(
        "Välj bolag (eller lämna tomt för nytt)",
        [""] + options,
        index=0 if not options else st.session_state.edit_idx + 1 if st.session_state.edit_idx < len(options) else 0
    )

    cnav = st.columns([1,1,2])
    with cnav[0]:
        if st.button("⬅️ Föregående", disabled=df.empty or st.session_state.edit_idx<=0):
            st.session_state.edit_idx = max(0, st.session_state.edit_idx-1)
            st.experimental_rerun()
    with cnav[1]:
        if st.button("Nästa ➡️", disabled=df.empty or st.session_state.edit_idx>=max(0,len(options)-1)):
            st.session_state.edit_idx = min(max(0,len(options)-1), st.session_state.edit_idx+1)
            st.experimental_rerun()
    with cnav[2]:
        if options:
            st.caption(f"Post **{st.session_state.edit_idx+1} / {len(options)}**")

    if valt_ticker:
        st.session_state.edit_idx = options.index(valt_ticker)

    bef = df.iloc[st.session_state.edit_idx] if (valt_ticker and not df.empty and st.session_state.edit_idx < len(df)) else None

    with st.form("form_bolag"):
        ticker = st.text_input("Ticker", value=(bef["Ticker"] if bef is not None else "")).upper()

        # Manuella fält
        utest = st.number_input("Utestående aktier (miljoner)", value=float(bef["Utestående aktier"]) if bef is not None else 0.0, step=0.01)
        ps   = st.number_input("P/S", value=float(bef["P/S"]) if bef is not None else 0.0, step=0.01)
        ps1  = st.number_input("P/S Q1", value=float(bef["P/S Q1"]) if bef is not None else 0.0, step=0.01)
        ps2  = st.number_input("P/S Q2", value=float(bef["P/S Q2"]) if bef is not None else 0.0, step=0.01)
        ps3  = st.number_input("P/S Q3", value=float(bef["P/S Q3"]) if bef is not None else 0.0, step=0.01)
        ps4  = st.number_input("P/S Q4", value=float(bef["P/S Q4"]) if bef is not None else 0.0, step=0.01)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef["Omsättning idag"]) if bef is not None else 0.0, step=0.01)
        oms_1    = st.number_input("Omsättning nästa år (miljoner)", value=float(bef["Omsättning nästa år"]) if bef is not None else 0.0, step=0.01)

        antal_aktier = st.number_input("Antal aktier du äger", value=float(bef["Antal aktier"]) if bef is not None else 0.0, step=1.0)

        # Visning (uppdateras efter Spara)
        col_auto = st.columns(4)
        col_auto[0].metric("Aktuell kurs", f'{(bef["Aktuell kurs"] if bef is not None else 0.0):.2f}')
        col_auto[1].metric("Valuta", f'{(bef["Valuta"] if bef is not None else "")}')
        col_auto[2].metric("Årlig utdelning", f'{(bef["Årlig utdelning"] if bef is not None else 0.0):.2f}')
        col_auto[3].metric("CAGR 5 år (%)", f'{(bef["CAGR 5 år (%)"] if bef is not None else 0.0):.2f}')

        sparaknapp = st.form_submit_button("💾 Spara (hämta Yahoo & beräkna)")

    if sparaknapp:
        if not ticker:
            st.error("Ticker måste anges.")
            return df

        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1,
            "Antal aktier": antal_aktier,
        }

        # Hämta från Yahoo
        pris, valuta, namn, utd = hamta_kurs_valuta_namn_utdelning(ticker)
        cagr = hamta_cagr_5ar(ticker)

        ny["Bolagsnamn"] = namn or ""
        ny["Valuta"] = (valuta or "USD").upper()
        ny["Årlig utdelning"] = float(utd or 0.0)
        ny["Aktuell kurs"] = round(float(pris), 2) if pris else 0.0
        ny["CAGR 5 år (%)"] = round(float(cagr), 2)

        # Räkna fram omsättning om 2 & 3 år
        oms2, oms3 = räkna_omsättning_framåt(ny["Omsättning nästa år"], ny["CAGR 5 år (%)"])
        ny["Omsättning om 2 år"] = oms2
        ny["Omsättning om 3 år"] = oms3

        # Fyll eventuella saknade kolumner
        for kol in KOLUMNER:
            if kol not in ny:
                ny[kol] = df.iloc[0][kol] if (not df.empty and kol in df.columns) else (0.0 if kol in NUMERISKA else "")

        if ticker in df["Ticker"].astype(str).values:
            df.loc[df["Ticker"].astype(str) == ticker, ny.keys()] = ny.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"{ticker} tillagt.")

        df = konvertera_typer(df)
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.experimental_rerun()

    return df

# --- Analysvy med portföljfilter & bläddring ---

def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.header("📈 Analys")

    # 🔁 Nollställ bläddringsindex
    if st.button("🔁 Nollställ bläddring", key="reset_analys"):
        st.session_state.analys_idx = 0
        st.experimental_rerun()

    dfall = df.copy()
    dfall["Antal aktier"] = pd.to_numeric(dfall["Antal aktier"], errors="coerce").fillna(0.0)

    filterval = st.radio("Visa", ["Alla bolag", "Endast portföljen"], horizontal=True)

    if "analys_filter" not in st.session_state:
        st.session_state.analys_filter = filterval
    elif st.session_state.analys_filter != filterval:
        st.session_state.analys_filter = filterval
        st.session_state.analys_idx = 0

    if filterval == "Endast portföljen":
        dfview = dfall[dfall["Antal aktier"] > 0].copy()
    else:
        dfview = dfall.copy()

    if dfview.empty:
        st.info("Inga rader att visa för valt filter.")
        return

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = max(0, min(st.session_state.analys_idx, len(dfview)-1))

    options = dfview["Ticker"].astype(str).tolist()
    valt = st.selectbox(
        "Välj bolag",
        options,
        index=st.session_state.analys_idx if st.session_state.analys_idx < len(options) else 0
    )

    cnav = st.columns([1,1,2,2])
    with cnav[0]:
        if st.button("⬅️ Föregående", use_container_width=True) and st.session_state.analys_idx > 0:
            st.session_state.analys_idx -= 1
            st.experimental_rerun()
    with cnav[1]:
        if st.button("Nästa ➡️", use_container_width=True) and st.session_state.analys_idx < len(options) - 1:
            st.session_state.analys_idx += 1
            st.experimental_rerun()
    with cnav[2]:
        st.caption(f"Post **{st.session_state.analys_idx + 1} / {len(options)}**")

    rad = dfview[dfview["Ticker"].astype(str) == valt]
    st.subheader(f"Detaljer: {valt}")
    st.dataframe(rad, use_container_width=True)

    st.markdown("---")
    st.subheader("Hela databasen")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Massuppdatera från Yahoo (1 s mellan)", type="primary"):
        nytt = massuppdatera_yahoo(df, paus_s=1.0)
        spara_data(nytt)
        st.success("Uppdaterat och sparat.")
        st.experimental_rerun()

# --- Investeringsförslag (med portföljfilter & bläddring) ---

def investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.header("💡 Investeringsförslag")

    # 🔁 Nollställ bläddringsindex
    if st.button("🔁 Nollställ bläddring", key="reset_forslag"):
        st.session_state.forslag_idx = 0
        st.experimental_rerun()

    # Filter: alla vs endast innehav i portföljen
    filterval = st.radio("Visa", ["Alla bolag", "Endast portföljen"], horizontal=True)

    if "forslag_filter" not in st.session_state:
        st.session_state.forslag_filter = filterval
    elif st.session_state.forslag_filter != filterval:
        st.session_state.forslag_filter = filterval
        st.session_state.forslag_idx = 0

    # Välj riktkurs som styr sortering/uppsida
    val = st.selectbox(
        "Sortera & beräkna uppsida utifrån:",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    d = df.copy()
    for col in ["Aktuell kurs", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    if filterval == "Endast portföljen":
        d = d[d["Antal aktier"] > 0]

    d["Uppside (%)"] = ((d[val] - d["Aktuell kurs"]) / d["Aktuell kurs"]) * 100.0
    d = d.replace([pd.NA, np.inf, -np.inf], np.nan).dropna(subset=["Aktuell kurs", val, "Uppside (%)"])
    d = d.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    if d.empty:
        st.info("Inga förslag – saknar värden för vald vy.")
        return

    if "forslag_idx" not in st.session_state:
        st.session_state.forslag_idx = 0

    cnav = st.columns([1,1,2,2])
    with cnav[0]:
        if st.button("⬅️ Föregående", use_container_width=True) and st.session_state.forslag_idx > 0:
            st.session_state.forslag_idx -= 1
    with cnav[1]:
        if st.button("Nästa ➡️", use_container_width=True) and st.session_state.forslag_idx < len(d) - 1:
            st.session_state.forslag_idx += 1
    with cnav[2]:
        st.caption(f"Förslag **{st.session_state.forslag_idx + 1} / {len(d)}**")

    rad = d.iloc[st.session_state.forslag_idx]
    akt_val = str(rad.get("Valuta", "") or "").upper()
    curr = float(rad["Aktuell kurs"] or 0.0)

    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad['Ticker']})")
    st.write(f"Aktuell kurs: **{curr:.2f} {akt_val}**")

    # Lista alla riktkurser och fetmarkera vald
    label_order = ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]
    vals = {lbl: float(rad.get(lbl, 0.0) or 0.0) for lbl in label_order}
    for lbl in label_order:
        v = vals[lbl]
        if lbl == val:
            st.markdown(f"- **{lbl}: {v:.2f} {akt_val}**")
        else:
            st.markdown(f"- {lbl}: {v:.2f} {akt_val}")

    st.write(f"Uppsida (baserat på *{val}*): **{float(rad['Uppside (%)']):.2f}%**")

    # Köpförslag (SEK in → antal i aktiens valuta)
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", min_value=0.0, value=0.0, step=100.0)
    vx = float(valutakurser.get(akt_val, 1.0))
    kapital_i_aktiens_valuta = (kapital_sek / vx) if vx > 0 else 0.0
    antal = int(kapital_i_aktiens_valuta // curr) if curr > 0 else 0
    investering_sek = antal * curr * vx
    st.write(f"Förslag: **{antal} st** (≈ {investering_sek:.2f} SEK)")

    # Andelar i portföljen (SEK-baserat)
    d_port = df.copy()
    d_port["Antal aktier"] = pd.to_numeric(d_port["Antal aktier"], errors="coerce").fillna(0.0)
    d_port["Aktuell kurs"] = pd.to_numeric(d_port["Aktuell kurs"], errors="coerce").fillna(0.0)
    if (d_port["Antal aktier"] > 0).any():
        d_port["Växelkurs"] = d_port.apply(lambda r: float(valutakurser.get(str(r.get("Valuta","")).upper(), 1.0)), axis=1)
        d_port["Värde (SEK)"] = (d_port["Antal aktier"] * d_port["Aktuell kurs"] * d_port["Växelkurs"]).astype(float)
        portfoljvarde = float(d_port["Värde (SEK)"].sum())
        nuvarande_innehav = d_port.loc[d_port["Ticker"].astype(str).str.upper() == str(rad["Ticker"]).upper(), "Värde (SEK)"].sum()

        nuvarande_andel = (nuvarande_innehav / portfoljvarde * 100.0) if portfoljvarde > 0 else 0.0
        ny_andel = ((nuvarande_innehav + investering_sek) / portfoljvarde * 100.0) if portfoljvarde > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Portföljvärde (SEK)", f"{portfoljvarde:,.0f}")
        c2.metric("Nuvarande andel", f"{nuvarande_andel:.2f}%")
        c3.metric("Andel efter köp", f"{ny_andel:.2f}%")
    else:
        st.info("Ingen registrerad portfölj (Antal aktier = 0 på alla rader).")

# --- Portföljvy (SEK-summering) ---

def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.header("📦 Min portfölj")

    # 🔁 Nollställ (ingen bläddring här – men för konsekvens)
    if st.button("🔁 Nollställ bläddring", key="reset_port"):
        st.experimental_rerun()

    d = df.copy()
    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)

    d = d[d["Antal aktier"] > 0]
    if d.empty:
        st.info("Du äger inga aktier.")
        return

    d["Växelkurs"] = d.apply(lambda r: float(valutakurser.get(str(r.get("Valuta","")).upper(), 1.0)), axis=1)
    d["Värde (SEK)"] = (d["Antal aktier"] * d["Aktuell kurs"] * d["Växelkurs"]).astype(float)

    d["Årlig utdelning"] = pd.to_numeric(d["Årlig utdelning"], errors="coerce").fillna(0.0)
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Årlig utdelning"] * d["Växelkurs"]).astype(float)

    total_varde = float(d["Värde (SEK)"].sum())
    total_utd = float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Totalt portföljvärde (SEK)", f"{total_varde:,.0f}")
    c2.metric("Total kommande utdelning (SEK/år)", f"{total_utd:,.0f}")
    c3.metric("Utdelning per månad (SEK)", f"{(total_utd/12):,.0f}")

    d["Andel (%)"] = (d["Värde (SEK)"] / total_varde * 100.0).round(2)
    st.dataframe(
        d[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
           "Värde (SEK)","Andel (%)","Årlig utdelning","Årlig utdelning (SEK)"]],
        use_container_width=True
    )

# --- Main ---

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(STANDARD_VALUTAKURSER["USD"]), step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(STANDARD_VALUTAKURSER["NOK"]), step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(STANDARD_VALUTAKURSER["CAD"]), step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(STANDARD_VALUTAKURSER["EUR"]), step=0.01),
        "SEK": 1.0,
    }

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df, valutakurser)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df)
        df2 = konvertera_typer(df2)
        df2 = uppdatera_berakningar(df2)
        spara_data(df2)
        st.experimental_rerun()
    elif meny == "Investeringsförslag":
        df = konvertera_typer(df)
        df = uppdatera_berakningar(df)
        investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
