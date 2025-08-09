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

# --- Standard växelkurser -> SEK (går att justera i sidopanelen) ---
STANDARD_VALUTAKURSER = {
    "USD": 9.50,
    "NOK": 0.93,
    "CAD": 7.00,
    "EUR": 11.10
}

# --- Kolumnuppsättning (måste matcha bladet) ---
KOL = [
    "Ticker","Bolagsnamn","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Valuta","Årlig utdelning","Aktuell kurs",
    "CAGR 5 år (%)","P/S-snitt","Omsättningsvaluta"
]

# ---------- Helpers ----------
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    rows = sheet.get_all_records()
    df = pd.DataFrame(rows)
    # Säkerställ alla kolumner finns
    for c in KOL:
        if c not in df.columns:
            df[c] = np.nan
    # Ordna kolumnordning
    df = df[KOL]
    return df

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    # Konvertera NaN -> ""
    out = df.copy()
    out = out.fillna("")
    sheet.clear()
    sheet.update([out.columns.tolist()] + out.astype(str).values.tolist())

def _num(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _to_upper_str(s, default=""):
    if s is None:
        return default
    try:
        return str(s).strip().upper()
    except Exception:
        return default

def _soft_clip_ps_for_avg(values, cap=100.0):
    # Används endast för snittet; originalvärden ändras inte
    xs = [v for v in values if v is not None and v > 0]
    if not xs:
        return 0.0
    clipped = [min(v, cap) for v in xs]
    return round(float(np.mean(clipped)), 2)

def applicera_cagr_regler(cagr_value):
    """>100% => 50%; <0 => 2%; annars så som det är."""
    if cagr_value is None or np.isnan(cagr_value):
        return None
    try:
        c = float(cagr_value)
    except Exception:
        return None
    if c > 100.0:
        return 50.0
    if c < 0.0:
        return 2.0
    return c

def beräkna_oms_framåt(oms_next_year, cagr_proc):
    """Från 'Omsättning nästa år' och CAGR (%) -> år2, år3"""
    base = _num(oms_next_year, 0.0)
    c = _num(cagr_proc, 0.0) / 100.0
    if base <= 0.0:
        return 0.0, 0.0
    year2 = round(base * (1.0 + c), 2)
    year3 = round(year2 * (1.0 + c), 2)
    return year2, year3

# ---------- Yahoo helpers ----------
def hamta_yahoo_basics(ticker: str):
    """Returnerar (bolagsnamn, aktuell_kurs, valuta) eller (None,None,'USD') vid fel."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        namn = info.get("shortName") or info.get("longName")
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        return namn, pris, valuta
    except Exception:
        return None, None, "USD"

def hamta_oms_hist_5y_cagr(ticker: str):
    """
    Försöker ta fram CAGR 5 år ur annual income statement (Total Revenue).
    yfinance ger ofta 3–4 år; om <3 punkter -> returnera None.
    CAGR beräknas som (last/first)^(1/n)-1 över n år -> *100.
    """
    try:
        t = yf.Ticker(ticker)
        # income_stmt (annual) – i vissa versioner 'income_stmt', ibland 'financials'
        df_is = None
        try:
            df_is = t.income_stmt  # nyare
        except Exception:
            pass
        if df_is is None or df_is.empty:
            try:
                df_is = t.financials  # fallback
            except Exception:
                df_is = None

        if df_is is None or df_is.empty:
            return None

        # Försök hitta "Total Revenue" rad (index kan vara sträng/med case)
        # Normalisera index till lower
        idx_lower = [str(i).strip().lower() for i in df_is.index]
        if "total revenue" in idx_lower:
            row_idx = idx_lower.index("total revenue")
        elif "totalrevenue" in idx_lower:
            row_idx = idx_lower.index("totalrevenue")
        else:
            return None

        rev_series = df_is.iloc[row_idx]
        # Kolumner är år (Timestamp) i fallande ordning – vi sorterar stigande på datum
        # Konvertera till list [ (year, value) ... ]
        points = []
        for col in rev_series.index:
            val = rev_series[col]
            if pd.isna(val) or val is None:
                continue
            # col kan vara Timestamp eller str (t.ex. '2024-12-31')
            try:
                year = pd.to_datetime(col).year
            except Exception:
                # Om inte datum, hoppa över
                continue
            points.append((year, float(val)))

        if len(points) < 3:
            return None

        points = sorted(points, key=lambda x: x[0])  # äldst -> senaste
        first_val = points[0][1]
        last_val = points[-1][1]
        years = points[-1][0] - points[0][0]
        if years <= 0 or first_val <= 0:
            return None

        cagr = (last_val / first_val) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return None

# ---------- Beräkningar ----------
def uppdatera_berakningar(df: pd.DataFrame, valutakurser: dict):
    """
    - Beräknar P/S-snitt (mjuk cap 100 bara i snittet)
    - Riktkurser (idag, om 1/2/3 år)
    OBS: Riktkurser använder aktiekursens valuta och 'Utestående aktier' i miljoner.
    Omsättningstalen förväntas vara i 'Omsättningsvaluta'. Vi räknar riktkurs i aktiens valuta,
    så vi konverterar omsättning -> aktievaluta via manuell växelkursinmatning.
    """
    df = df.copy()

    # Karta valutor
    aktie_val = df["Valuta"].fillna("USD").astype(str).str.upper()
    oms_val = df["Omsättningsvaluta"].fillna("USD").astype(str).str.upper()

    # Hämtar växelkurs (-> SEK) från sidopanelvärden; för omräkning omsättning -> aktiens valuta
    def vx_to_sek(cur):
        return float(valutakurser.get(cur, 1.0))

    # Omräkning av omsättningar: från omsättningsvaluta -> SEK -> aktievaluta
    def oms_to_aktieval(oms, ov, av):
        # oms * (OV->SEK) / (AV->SEK)
        try:
            ov_sek = vx_to_sek(ov)
            av_sek = vx_to_sek(av)
            if ov_sek <= 0 or av_sek <= 0:
                return float(oms)
            return float(oms) * (ov_sek / av_sek)
        except Exception:
            return float(oms)

    # P/S-snitt (mjuk cap = 100)
    ps_cols = ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
    ps_vals = df[ps_cols].applymap(lambda x: _num(x, 0.0))
    df["P/S-snitt"] = ps_vals.apply(lambda r: _soft_clip_ps_for_avg([r[c] for c in ps_cols]), axis=1)

    # Konverterade omsättningar till aktievaluta
    df["_Oms_idag_AV"] = [
        oms_to_aktieval(_num(o,0.0), ov, av) for o,ov,av in zip(df["Omsättning idag"], oms_val, aktie_val)
    ]
    df["_Oms_1_AV"] = [
        oms_to_aktieval(_num(o,0.0), ov, av) for o,ov,av in zip(df["Omsättning nästa år"], oms_val, aktie_val)
    ]
    df["_Oms_2_AV"] = [
        oms_to_aktieval(_num(o,0.0), ov, av) for o,ov,av in zip(df["Omsättning om 2 år"], oms_val, aktie_val)
    ]
    df["_Oms_3_AV"] = [
        oms_to_aktieval(_num(o,0.0), ov, av) for o,ov,av in zip(df["Omsättning om 3 år"], oms_val, aktie_val)
    ]

    # Riktkurser = (Omsättning * P/S-snitt) / Utestående aktier
    uos = df["Utestående aktier"].apply(lambda x: _num(x, 0.0))
    psn = df["P/S-snitt"].apply(lambda x: _num(x, 0.0))

    def rikt(oms):
        return round((oms * psn / uos).replace([np.inf, -np.inf], 0.0).fillna(0.0), 2)

    df["Riktkurs idag"]   = rikt(pd.Series(df["_Oms_idag_AV"]))
    df["Riktkurs om 1 år"] = rikt(pd.Series(df["_Oms_1_AV"]))
    df["Riktkurs om 2 år"] = rikt(pd.Series(df["_Oms_2_AV"]))
    df["Riktkurs om 3 år"] = rikt(pd.Series(df["_Oms_3_AV"]))

    # Städa temp
    df = df.drop(columns=[c for c in df.columns if c.startswith("_Oms_")], errors="ignore")
    return df

# ---------- UI helpers ----------
def positionstext(idx, total):
    if total <= 0:
        return "0/0"
    return f"{idx+1}/{total}"

def lagg_till_eller_uppdatera(df: pd.DataFrame, valutakurser: dict):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Navigering/bläddring i listan
    tickers = df["Ticker"].fillna("").astype(str).tolist()
    visningslista = [f"{r['Bolagsnamn']} ({r['Ticker']})" if pd.notna(r["Bolagsnamn"]) and str(r["Bolagsnamn"]).strip()
                     else r["Ticker"] for _, r in df.iterrows()]
    visningslista = [v if v else "(tom)" for v in visningslista]

    if "form_index" not in st.session_state:
        st.session_state.form_index = 0

    # Dropdown + föregående/nästa
    colA, colB, colC = st.columns([4,1,1])
    with colA:
        valt = st.selectbox("Välj bolag (förifyll fält)", options=["(nytt bolag)"] + visningslista, index=0)
    with colB:
        if st.button("⬅️ Föregående"):
            st.session_state.form_index = max(0, st.session_state.form_index - 1)
    with colC:
        if st.button("Nästa ➡️"):
            st.session_state.form_index = min(max(0, len(tickers)-1), st.session_state.form_index + 1)

    st.caption(f"Post: {positionstext(st.session_state.form_index, len(tickers))}")

    # Hämta befintlig rad
    bef = pd.Series(dtype=object)
    if valt != "(nytt bolag)" and len(df) > 0:
        bef = df.iloc[st.session_state.form_index]

    # --- Formulär ---
    with st.form("bolagsform"):
        st.markdown("#### Obligatoriskt (manuellt)")
        c1,c2,c3 = st.columns(3)
        with c1:
            ticker = st.text_input("Ticker", value=str(bef.get("Ticker","")) if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=_num(bef.get("Utestående aktier"),0.0), step=0.1, format="%.2f")
            antal = st.number_input("Antal aktier", value=_num(bef.get("Antal aktier"),0.0), step=1.0, format="%.0f")
        with c2:
            ps = st.number_input("P/S", value=_num(bef.get("P/S"),0.0), step=0.01)
            ps1 = st.number_input("P/S Q1", value=_num(bef.get("P/S Q1"),0.0), step=0.01)
            ps2 = st.number_input("P/S Q2", value=_num(bef.get("P/S Q2"),0.0), step=0.01)
        with c3:
            ps3 = st.number_input("P/S Q3", value=_num(bef.get("P/S Q3"),0.0), step=0.01)
            ps4 = st.number_input("P/S Q4", value=_num(bef.get("P/S Q4"),0.0), step=0.01)

        st.markdown("#### Omsättning (manuellt)")
        c4,c5,c6 = st.columns(3)
        with c4:
            oms_val = st.selectbox("Omsättningsvaluta", ["USD","NOK","CAD","EUR"], index= ["USD","NOK","CAD","EUR"].index(_to_upper_str(bef.get("Omsättningsvaluta","USD"),"USD")))
        with c5:
            oms_idag = st.number_input("Omsättning idag", value=_num(bef.get("Omsättning idag"),0.0), step=100.0)
        with c6:
            oms_1 = st.number_input("Omsättning nästa år", value=_num(bef.get("Omsättning nästa år"),0.0), step=100.0)

        st.divider()
        colL, colR = st.columns([1,1])
        spar = colL.form_submit_button("💾 Spara & hämta från Yahoo")
        avbryt = colR.form_submit_button("Avbryt ändringar")

    if not spar:
        return df  # inget att göra

    # --- SPARA & HÄMTA ---
    if not ticker:
        st.error("Ange en ticker.")
        return df

    uppd_fields = []  # för infobox

    # Bygg ny rad med manuella fält
    ny = {
        "Ticker": ticker,
        "Utestående aktier": utest,
        "Antal aktier": antal,
        "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
        "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1,
        "Omsättningsvaluta": oms_val,
    }

    # Yahoo basics
    namn, pris, valuta = hamta_yahoo_basics(ticker)
    if pris is not None:
        ny["Aktuell kurs"] = round(float(pris), 2)
        uppd_fields.append(f"Aktuell kurs → {ny['Aktuell kurs']}")
    if namn:
        ny["Bolagsnamn"] = namn
        uppd_fields.append(f"Bolagsnamn → {namn}")
    if valuta:
        ny["Valuta"] = _to_upper_str(valuta,"USD")
        uppd_fields.append(f"Valuta → {ny['Valuta']}")

    # CAGR 5 år
    cagr = hamta_oms_hist_5y_cagr(ticker)
    if cagr is not None:
        ny["CAGR 5 år (%)"] = round(float(cagr), 2)
        uppd_fields.append(f"CAGR 5 år (%) → {ny['CAGR 5 år (%)']}")

    just_cagr = applicera_cagr_regler(ny.get("CAGR 5 år (%)"))
    if just_cagr is not None and _num(ny.get("Omsättning nästa år"),0.0) > 0:
        oms2, oms3 = beräkna_oms_framåt(ny.get("Omsättning nästa år"), just_cagr)
        ny["Omsättning om 2 år"] = oms2
        ny["Omsättning om 3 år"] = oms3
        uppd_fields.append(f"Omsättning om 2 år → {oms2}")
        uppd_fields.append(f"Omsättning om 3 år → {oms3}")

    # Årlig utdelning (om den finns i info)
    try:
        div = yf.Ticker(ticker).info.get("dividendRate", None)
        if div is not None:
            ny["Årlig utdelning"] = float(div)
            uppd_fields.append(f"Årlig utdelning → {float(div)}")
    except Exception:
        pass

    # Uppdatera/infoga i df
    if ticker.upper() in df["Ticker"].astype(str).str.upper().values:
        mask = df["Ticker"].astype(str).str.upper() == ticker.upper()
        for k,v in ny.items():
            df.loc[mask, k] = v
        st.success(f"{ticker} uppdaterat.")
    else:
        df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
        st.success(f"{ticker} tillagt.")

    # Räkna och spara
    df = uppdatera_berakningar(df, valutakurser)
    spara_data(df)

    # Infobox
    if uppd_fields:
        st.info("**Följande fält uppdaterades:**\n- " + "\n- ".join(uppd_fields))
    else:
        st.info("Inga nya fält kunde uppdateras från Yahoo (manuella fält sparades).")

    return df


def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📈 Analys")
    if "analys_index" not in st.session_state:
        st.session_state.analys_index = 0

    # Förhandsvisa ett bolag (dropdown + bläddra)
    vis = [f"{r['Bolagsnamn']} ({r['Ticker']})" if pd.notna(r["Bolagsnamn"]) and str(r["Bolagsnamn"]).strip()
           else r["Ticker"] for _, r in df.iterrows()]
    vis = [v if v else "(tom)" for v in vis]

    col1, col2, col3 = st.columns([4,1,1])
    with col1:
        if len(vis) > 0:
            sel = st.selectbox("Förhandsvisa bolag", options=vis, index=min(st.session_state.analys_index, max(0,len(vis)-1)))
            st.session_state.analys_index = vis.index(sel)
    with col2:
        if st.button("⬅️ Föregående", key="anal_prev"):
            st.session_state.analys_index = max(0, st.session_state.analys_index - 1)
    with col3:
        if st.button("Nästa ➡️", key="anal_next"):
            st.session_state.analys_index = min(max(0,len(vis)-1), st.session_state.analys_index + 1)

    st.caption(f"Post: {positionstext(st.session_state.analys_index, len(vis))}")

    if len(df) > 0:
        st.write(df.iloc[[st.session_state.analys_index]])

    st.divider()

    # Uppdatera alla från Yahoo
    miss = []
    total = len(df)
    if st.button("🔄 Uppdatera ALLA från Yahoo (1 s/bolag)"):
        bar = st.progress(0.0)
        updated = 0
        for i, row in df.iterrows():
            t = str(row["Ticker"]).strip().upper()
            try:
                namn, pris, valuta = hamta_yahoo_basics(t)
                if pris is not None:
                    df.at[i, "Aktuell kurs"] = round(float(pris), 2)
                if namn:
                    df.at[i, "Bolagsnamn"] = namn
                if valuta:
                    df.at[i, "Valuta"] = _to_upper_str(valuta,"USD")

                cagr = hamta_oms_hist_5y_cagr(t)
                if cagr is not None:
                    df.at[i, "CAGR 5 år (%)"] = round(float(cagr), 2)

                # Oms 2/3 år om vi har Omsättning nästa år
                jc = applicera_cagr_regler(df.at[i, "CAGR 5 år (%)"] if "CAGR 5 år (%)" in df.columns else None)
                if jc is not None and _num(df.at[i,"Omsättning nästa år"],0.0) > 0:
                    o2, o3 = beräkna_oms_framåt(df.at[i,"Omsättning nästa år"], jc)
                    df.at[i, "Omsättning om 2 år"] = o2
                    df.at[i, "Omsättning om 3 år"] = o3

                updated += 1
            except Exception as e:
                miss.append(f"{t}: {e}")
            bar.progress((i+1)/max(1,total))
            time.sleep(1)  # 1 sekund mellan anrop

        df = uppdatera_berakningar(df, valutakurser)
        spara_data(df)
        st.success(f"Uppdaterade {updated} av {total} bolag.")
        if miss:
            st.warning("Kunde inte uppdatera:\n" + "\n".join(miss))

    st.write("### Databas")
    st.dataframe(df, use_container_width=True)


def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")
    d = df.copy()
    d = d[pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0) > 0]

    if d.empty:
        st.info("Du äger inga aktier.")
        return

    # växelkurs -> SEK
    def vx(cur): return float(valutakurser.get(_to_upper_str(cur,"USD"), 1.0))
    d["Växelkurs"] = d["Valuta"].apply(lambda x: vx(x))
    d["Värde (SEK)"] = d["Antal aktier"].apply(_num) * d["Aktuell kurs"].apply(_num) * d["Växelkurs"]
    total_varde = round(d["Värde (SEK)"].sum(), 2)

    # utdelning
    d["Total årlig utdelning (SEK)"] = d["Antal aktier"].apply(_num) * d["Årlig utdelning"].apply(_num) * d["Växelkurs"]
    total_utd = round(d["Total årlig utdelning (SEK)"].sum(), 2)
    månads_utd = round(total_utd / 12.0, 2)

    d["Andel (%)"] = (d["Värde (SEK)"] / d["Värde (SEK)"].sum() * 100.0).round(2)

    st.markdown(f"**Totalt portföljvärde:** {total_varde} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {total_utd} SEK")
    st.markdown(f"**Genomsnittlig månadsutdelning:** {månads_utd} SEK")

    st.dataframe(
        d[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

def visa_investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktvals = st.selectbox(
        "Använd riktkurs:",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    filter_portf = st.checkbox("Visa endast portföljens innehav", value=False)

    d = df.copy()
    # potential = (rikt - aktuell) / aktuell
    d["Potential (%)"] = ((pd.to_numeric(d[riktvals], errors="coerce") - pd.to_numeric(d["Aktuell kurs"], errors="coerce"))
                          / pd.to_numeric(d["Aktuell kurs"], errors="coerce")) * 100.0
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["Potential (%)"])

    if filter_portf:
        d = d[pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0) > 0]

    d = d.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if d.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    # Paginering/bläddra
    if "forslag_idx" not in st.session_state:
        st.session_state.forslag_idx = 0

    colA, colB, colC = st.columns([1,1,6])
    with colA:
        if st.button("⬅️ Föregående", key="fs_prev"):
            st.session_state.forslag_idx = max(0, st.session_state.forslag_idx - 1)
    with colB:
        if st.button("Nästa ➡️", key="fs_next"):
            st.session_state.forslag_idx = min(len(d)-1, st.session_state.forslag_idx + 1)
    with colC:
        st.caption(f"Förslag: {positionstext(st.session_state.forslag_idx, len(d))}")

    rad = d.iloc[st.session_state.forslag_idx]

    # Antal att köpa (SEK / (kurs * växelkurs))
    vx = float(valutakurser.get(_to_upper_str(rad["Valuta"],"USD"), 1.0))
    kurs_sek = _num(rad["Aktuell kurs"]) * vx
    antal = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0

    # Portföljandel före/efter (beräknas på SEK)
    port = df.copy()
    port = port[pd.to_numeric(port["Antal aktier"], errors="coerce").fillna(0) > 0]
    if port.empty:
        nuv_andel = 0.0
        ny_andel = 0.0
        portvarde = 0.0
    else:
        port["Växelkurs"] = port["Valuta"].apply(lambda x: float(valutakurser.get(_to_upper_str(x,"USD"),1.0)))
        port["Värde (SEK)"] = port["Antal aktier"].apply(_num) * port["Aktuell kurs"].apply(_num) * port["Växelkurs"]
        portvarde = port["Värde (SEK)"].sum()
        nuv = port[port["Ticker"].astype(str).str.upper() == str(rad["Ticker"]).upper()]["Värde (SEK)"].sum()
        ny_total = nuv + antal * kurs_sek
        nuv_andel = round((nuv / portvarde * 100.0), 2) if portvarde > 0 else 0.0
        ny_andel = round((ny_total / portvarde * 100.0), 2) if portvarde > 0 else 0.0

    # Presentera – i aktiens valuta
    def bold_if(label, is_sel):  # enkel fetmarkering i markdown
        return f"**{label}**" if is_sel else label

    st.markdown(f"### {rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})")
    st.markdown(f"- **Aktuell kurs:** {round(_num(rad['Aktuell kurs']),2)} {rad.get('Valuta','')}")
    st.markdown(
        "- " +
        bold_if(f"Riktkurs idag: {round(_num(rad['Riktkurs idag']),2)} {rad.get('Valuta','')}", riktvals=="Riktkurs idag")
    )
    st.markdown(
        "- " +
        bold_if(f"Riktkurs om 1 år: {round(_num(rad['Riktkurs om 1 år']),2)} {rad.get('Valuta','')}", riktvals=="Riktkurs om 1 år")
    )
    st.markdown(
        "- " +
        bold_if(f"Riktkurs om 2 år: {round(_num(rad['Riktkurs om 2 år']),2)} {rad.get('Valuta','')}", riktvals=="Riktkurs om 2 år")
    )
    st.markdown(
        "- " +
        bold_if(f"Riktkurs om 3 år: {round(_num(rad['Riktkurs om 3 år']),2)} {rad.get('Valuta','')}", riktvals=="Riktkurs om 3 år")
    )

    st.markdown(f"- **Potential (mot vald riktkurs):** {round(_num(rad['Potential (%)']),2)}%")
    st.markdown(f"- **Antal att köpa (för {int(kapital_sek)} SEK):** {antal} st")
    st.markdown(f"- **Nuvarande portföljandel:** {nuv_andel}%  •  **Efter köp:** {ny_andel}%")

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs in data
    df = hamta_data()

    # Sidopanel – valutakurser
    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=STANDARD_VALUTAKURSER["USD"], step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=STANDARD_VALUTAKURSER["NOK"], step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=STANDARD_VALUTAKURSER["CAD"], step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=STANDARD_VALUTAKURSER["EUR"], step=0.01),
    }

    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Analys":
        df = uppdatera_berakningar(df, valutakurser)
        analysvy(df, valutakurser)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, valutakurser)
        # df redan sparad i funktionen

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, valutakurser)
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, valutakurser)
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
