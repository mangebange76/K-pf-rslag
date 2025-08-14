import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

# ---------- Grundinställningar ----------
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# Standard-valutakurser till SEK (kan ändras i sidomenyn)
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18}

# ---------- Google Sheets I/O ----------
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

# ---------- Kolumnhantering ----------
def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Exakt de kolumner du listat
    kolonner = [
        "Ticker","Bolagsnamn","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","Valuta","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
    ]
    for c in kolonner:
        if c not in df.columns:
            # numeriska fält default 0.0, annars tom sträng
            if c in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                     "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                     "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                     "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"]:
                df[c] = 0.0
            else:
                df[c] = ""
    # Ta bort ev. överblivna kolumner som kan spöka i beräkningar (icke-destruktivt för de du listat)
    return df[kolonner]

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "Valuta" in df.columns:
        df["Valuta"] = df["Valuta"].astype(str).str.upper().replace("", "USD")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
    return df

# ---------- Inputhjälp: tillåt tomma fält ----------
def _parse_float(txt: str) -> float:
    if txt is None: return 0.0
    txt = str(txt).strip().replace(" ", "").replace(",", ".")
    if txt == "": return 0.0
    try:
        return float(txt)
    except:
        return 0.0

def float_text_input(label: str, default_val: float = 0.0, key: str = None) -> float:
    default_str = "" if (default_val == 0.0 or pd.isna(default_val)) else str(default_val)
    s = st.text_input(label, value=default_str, key=key)
    return _parse_float(s)

# ---------- Yahoo! Finance hämtning ----------
def yahoo_hämta_bas(ticker: str):
    """Hämta bolagsnamn, aktuell kurs, valuta, årlig utdelning (per aktie) om möjligt."""
    namn = ""
    kurs = 0.0
    valuta = "USD"
    utd = 0.0
    try:
        y = yf.Ticker(ticker)
        info = y.info
        namn = info.get("shortName") or info.get("longName") or ""
        kurs = float(info.get("regularMarketPrice") or 0.0)
        valuta = (info.get("currency") or "USD").upper()
        # Utdelning per aktie (trailingAnnualDividendRate) om finns
        utd = float(info.get("trailingAnnualDividendRate") or 0.0)
    except Exception:
        pass
    return namn, kurs, valuta, utd

def yahoo_hämta_årsomsättningar(ticker: str):
    """Försök läsa årliga intäkter (Total Revenue) och beräkna CAGR över de senaste 5 åren (om möjligt)."""
    try:
        y = yf.Ticker(ticker)
        # Nyare yfinance kan ha .income_stmt eller .financials; båda kan testas
        df = None
        try:
            df = y.income_stmt  # årsdata
        except Exception:
            pass
        if df is None or df.empty:
            try:
                df = y.financials  # fallback
            except Exception:
                df = None

        if df is None or df.empty:
            return None  # kan ej beräkna

        # Försök hitta "Total Revenue"
        row_names = [s.lower() for s in df.index.astype(str)]
        if "total revenue" in row_names:
            idx = row_names.index("total revenue")
            serie = df.iloc[idx]
        else:
            # ibland heter den "TotalRevenue" eller liknande
            mask = df.index.astype(str).str.replace(" ", "").str.lower().str.contains("totalrevenue")
            if mask.any():
                serie = df[mask].iloc[0]
            else:
                return None

        # serie: kolumner = år (Timestamp), värden = int/float
        # Plocka ut senaste upp till 6 punkter om finns
        s = serie.dropna()
        if s.empty: 
            return None

        # Sortera efter år (kolumn-namn är ofta Timestamps)
        try:
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
        except Exception:
            pass

        # Behöver minst två punkter. För “5 år CAGR” försök ta 6 år om finns, annars ta största spannet.
        if len(s) < 2:
            return None

        first_val = float(s.iloc[0])
        last_val  = float(s.iloc[-1])

        if first_val <= 0 or last_val <= 0:
            return None

        n_years = len(s) - 1  # antal intervall
        cagr = (last_val / first_val) ** (1.0 / n_years) - 1.0
        return cagr * 100.0  # i %
    except Exception:
        return None

# ---------- Beräkningar ----------
def cagr_till_tillväxt_for_framtiden(cagr_procent: float) -> float:
    """Affärsregel: >100% => 50%, <0% => 2%, annars lämna oförändrat."""
    if cagr_procent is None:
        return None
    if cagr_procent > 100.0:
        return 50.0
    if cagr_procent < 0.0:
        return 2.0
    return cagr_procent

def beräkna_omsättning_framåt(oms_year1: float, tillväxt_procent: float, år: int) -> float:
    """Beräkna år 2/3 från år1 med given årlig tillväxt (procent)."""
    if oms_year1 <= 0 or tillväxt_procent is None:
        return 0.0
    g = 1.0 + (tillväxt_procent / 100.0)
    # år=2 ⇒ 1 steg; år=3 ⇒ 2 steg från år1
    steg = max(0, år - 1)
    return float(oms_year1 * (g ** steg))

def beräkna_ps_snitt(rad) -> float:
    vals = [rad.get("P/S Q1", 0.0), rad.get("P/S Q2", 0.0), rad.get("P/S Q3", 0.0), rad.get("P/S Q4", 0.0)]
    vals = [float(v) for v in vals if pd.to_numeric(v, errors="coerce") and float(v) > 0.0]
    if not vals:
        return 0.0
    return round(float(np.mean(vals)), 2)

def beräkna_riktkurs(oms: float, ps_snitt: float, utest: float) -> float:
    if utest > 0 and ps_snitt > 0 and oms > 0:
        return round((oms * ps_snitt) / utest, 2)
    return 0.0

# ---------- Uppdatera en rad från Yahoo + räkna om ----------
def uppdatera_från_yahoo_för_rad(df: pd.DataFrame, idx: int):
    t = str(df.at[idx, "Ticker"]).strip().upper()
    if not t:
        return

    namn, kurs, valuta, utd = yahoo_hämta_bas(t)
    if namn:   df.at[idx, "Bolagsnamn"] = namn
    if kurs>0: df.at[idx, "Aktuell kurs"] = round(float(kurs), 2)
    if valuta: df.at[idx, "Valuta"] = valuta
    if utd>=0: df.at[idx, "Årlig utdelning"] = round(float(utd), 6)

    cagr = yahoo_hämta_årsomsättningar(t)
    if cagr is not None:
        df.at[idx, "CAGR 5 år (%)"] = round(float(cagr), 4)
        # räkna fram om 2/3 år från "Omsättning nästa år"
        oms1 = float(df.at[idx, "Omsättning nästa år"])
        g = cagr_till_tillväxt_for_framtiden(float(cagr))
        if g is not None and oms1 > 0:
            df.at[idx, "Omsättning om 2 år"] = round(beräkna_omsättning_framåt(oms1, g, 2), 2)
            df.at[idx, "Omsättning om 3 år"] = round(beräkna_omsättning_framtåt(oms1, g, 3), 2)

    # Räkna om P/S-snitt + riktkurser
    ps_snitt = beräkna_ps_snitt(df.loc[idx])
    df.at[idx, "P/S-snitt"] = ps_snitt

    utest = float(df.at[idx, "Utestående aktier"])
    df.at[idx, "Riktkurs idag"]    = beräkna_riktkurs(float(df.at[idx, "Omsättning idag"]),     ps_snitt, utest)
    df.at[idx, "Riktkurs om 1 år"] = beräkna_riktkurs(float(df.at[idx, "Omsättning nästa år"]), ps_snitt, utest)
    df.at[idx, "Riktkurs om 2 år"] = beräkna_riktkurs(float(df.at[idx, "Omsättning om 2 år"]),  ps_snitt, utest)
    df.at[idx, "Riktkurs om 3 år"] = beräkna_riktkurs(float(df.at[idx, "Omsättning om 3 år"]),  ps_snitt, utest)

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    # Beräkna P/S-snitt + riktkurser för alla rader (utan Yahoo-hämtning)
    for i, rad in df.iterrows():
        ps_snitt = beräkna_ps_snitt(rad)
        df.at[i, "P/S-snitt"] = ps_snitt
        u = float(rad["Utestående aktier"])
        df.at[i, "Riktkurs idag"]    = beräkna_riktkurs(float(rad["Omsättning idag"]),     ps_snitt, u)
        df.at[i, "Riktkurs om 1 år"] = beräkna_riktkurs(float(rad["Omsättning nästa år"]), ps_snitt, u)
        df.at[i, "Riktkurs om 2 år"] = beräkna_riktkurs(float(rad["Omsättning om 2 år"]),  ps_snitt, u)
        df.at[i, "Riktkurs om 3 år"] = beräkna_riktkurs(float(rad["Omsättning om 3 år"]),  ps_snitt, u)
    return df

# ---------- VY: Analys ----------
def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📈 Analys")

    # --- Bläddra/filtrera enskilt bolag ---
    tickers = df["Ticker"].astype(str).tolist()
    if "analys_idx" not in st.session_state: st.session_state["analys_idx"] = 0

    colA, colB, colC = st.columns([1,1,4])
    with colA:
        if st.button("⬅️ Föregående", key="an_prev"):
            st.session_state["analys_idx"] = (st.session_state["analys_idx"] - 1) % max(len(tickers), 1)
    with colB:
        if st.button("Nästa ➡️", key="an_next"):
            st.session_state["analys_idx"] = (st.session_state["analys_idx"] + 1) % max(len(tickers), 1)
    with colC:
        st.caption(f"Post {st.session_state['analys_idx']+1}/{len(tickers) if tickers else 0}")

    valt_ticker = st.selectbox("Välj bolag", tickers, index=st.session_state["analys_idx"])
    # synka index om användaren byter i selectbox
    st.session_state["analys_idx"] = tickers.index(valt_ticker) if valt_ticker in tickers else 0

    df_one = df[df["Ticker"] == valt_ticker]
    st.write("**Valt bolag (ur databasen):**")
    st.dataframe(df_one, use_container_width=True)

    st.write("—")
    if st.button("🔄 Uppdatera valt bolag från Yahoo", key="an_upd_one"):
        i = df.index[df["Ticker"] == valt_ticker]
        if len(i):
            uppdatera_från_yahoo_för_rad(df, int(i[0]))
            df = uppdatera_berakningar(df)
            spara_data(df)
            st.success(f"{valt_ticker} uppdaterat och beräknat.")
        else:
            st.warning("Hittade inte raden i tabellen.")

    st.write("—")
    if st.button("🌐 Uppdatera ALLA från Yahoo (1s paus)", key="an_upd_all"):
        miss = []
        for i in range(len(df)):
            try:
                uppdatera_från_yahoo_för_rad(df, i)
            except Exception as e:
                miss.append(df.at[i, "Ticker"])
            time.sleep(1.0)
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success("Massuppdatering klar.")
        if miss:
            st.warning("Kunde inte uppdatera:\n" + ", ".join([m for m in miss if m]))

    st.write("**Hela databasen:**")
    st.dataframe(df, use_container_width=True)

# ---------- VY: Lägg till / uppdatera bolag ----------
def lagg_till_eller_uppdatera(df: pd.DataFrame):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # lista för rullgardin + bläddring
    poster = [f"{r['Bolagsnamn'] or ''} ({r['Ticker']})".strip() for _, r in df.iterrows()]
    # Fallback om tomma namn
    poster = [p if p != " ()" else r for p, r in zip(poster, df["Ticker"])]

    if "form_idx" not in st.session_state: st.session_state["form_idx"] = 0
    colA, colB, colC = st.columns([1,1,4])
    with colA:
        if st.button("⬅️ Föregående", key="form_prev"):
            st.session_state["form_idx"] = (st.session_state["form_idx"] - 1) % max(len(poster), 1)
    with colB:
        if st.button("Nästa ➡️", key="form_next"):
            st.session_state["form_idx"] = (st.session_state["form_idx"] + 1) % max(len(poster), 1)
    with colC:
        st.caption(f"Post {st.session_state['form_idx']+1}/{len(poster) if poster else 0}")

    valt_label = st.selectbox("Välj befintligt (eller lämna tom för nytt)", [""] + poster,
                              index=(st.session_state["form_idx"] + 1 if poster else 0))
    # Synka index
    if valt_label and poster:
        st.session_state["form_idx"] = poster.index(valt_label)

    # Hämta befintlig rad eller tom
    if valt_label:
        curr_idx = st.session_state["form_idx"]
        bef = df.iloc[curr_idx].copy()
    else:
        bef = pd.Series(dtype=object)

    with st.form("form_bolag"):
        # Manuella fält som du vill ange själv (textinput för att kunna vara helt tomma)
        ticker = st.text_input("Ticker", value=str(bef.get("Ticker","")) if not bef.empty else "").upper()
        utest  = float_text_input("Utestående aktier (miljoner)", default_val=float(bef.get("Utestående aktier",0.0)))
        antal  = float_text_input("Antal aktier du äger",        default_val=float(bef.get("Antal aktier",0.0)))

        ps    = float_text_input("P/S",     default_val=float(bef.get("P/S",0.0)))
        ps1   = float_text_input("P/S Q1",  default_val=float(bef.get("P/S Q1",0.0)))
        ps2   = float_text_input("P/S Q2",  default_val=float(bef.get("P/S Q2",0.0)))
        ps3   = float_text_input("P/S Q3",  default_val=float(bef.get("P/S Q3",0.0)))
        ps4   = float_text_input("P/S Q4",  default_val=float(bef.get("P/S Q4",0.0)))

        oms_idag = float_text_input("Omsättning idag (miljoner)",     default_val=float(bef.get("Omsättning idag",0.0)))
        oms_1    = float_text_input("Omsättning nästa år (miljoner)", default_val=float(bef.get("Omsättning nästa år",0.0)))

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar:
        if not ticker:
            st.error("Ticker krävs.")
            return df

        # skapa/uppdatera rad
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1
        }

        if valt_label:
            # uppdatera befintlig (via index)
            curr_idx = st.session_state["form_idx"]
            for k,v in ny.items():
                df.iat[curr_idx, df.columns.get_loc(k)] = v
            # Hämta Yahoo för raden + räkna om
            uppdatera_från_yahoo_för_rad(df, curr_idx)
        else:
            # lägg till ny och hämta Yahoo
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            uppdatera_från_yahoo_för_rad(df, len(df)-1)

        # Slutlig omräkning för allt och spara
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success(f"{ticker} sparat och uppdaterat från Yahoo.")

    # Visa ett snapshot på den aktuella posten (om någon)
    if "form_idx" in st.session_state and len(df)>0:
        st.write("**Aktuell post:**")
        st.dataframe(df.iloc[[st.session_state["form_idx"]]], use_container_width=True)

# ---------- VY: Portfölj ----------
def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].map(valutakurser).fillna(1.0)
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port["Andel (%)"] = (port["Värde (SEK)"] / port["Värde (SEK)"].sum() * 100).round(2)

    port["Årsutd. SEK"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_v = float(port["Värde (SEK)"].sum())
    tot_u = float(port["Årsutd. SEK"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(tot_v,2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(tot_u,2)} SEK")
    st.markdown(f"**Månadsutdelning (snitt):** {round(tot_u/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Årsutd. SEK"]],
        use_container_width=True
    )

# ---------- VY: Investeringsförslag (med nytt filter) ----------
def visa_investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0, key="inv_k")
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
                                index=1, key="inv_rk")
    endast_pf = st.checkbox("Visa endast innehav i portföljen", value=False, key="inv_pf")
    sortläge = st.radio("Sorteringsläge", ["Max potential","Närmast riktkurs"], index=0, key="inv_mode")

    rk_pos = "Alla"
    if sortläge == "Närmast riktkurs":
        rk_pos = st.radio("Visa", ["Alla","Under riktkurs","Över riktkurs"], index=0, horizontal=True, key="inv_pos")

    d = df.copy()
    d["Växelkurs"] = d["Valuta"].map(valutakurser).fillna(1.0)
    d = d[pd.to_numeric(d["Aktuell kurs"], errors="coerce") > 0]
    d = d[pd.to_numeric(d[riktkurs_val], errors="coerce") > 0]

    if endast_pf:
        d = d[d["Antal aktier"] > 0]

    if d.empty:
        st.info("Inga bolag att visa.")
        return

    d["Potential (%)"] = ((d[riktkurs_val] - d["Aktuell kurs"]) / d["Aktuell kurs"]) * 100.0
    d["Avvikelse mot riktkurs (%)"] = ((d["Aktuell kurs"] - d[riktkurs_val]) / d[riktkurs_val]) * 100.0
    d["|Avvikelse|"] = d["Avvikelse mot riktkurs (%)"].abs()

    if sortläge == "Max potential":
        kandidater = d[d["Potential (%)"] > 0].sort_values("Potential (%)", ascending=False, ignore_index=True)
        sort_text = "max potential"
    else:
        tmp = d
        if rk_pos == "Under riktkurs":
            tmp = tmp[tmp["Avvikelse mot riktkurs (%)"] < 0]
        elif rk_pos == "Över riktkurs":
            tmp = tmp[tmp["Avvikelse mot riktkurs (%)"] > 0]
        kandidater = tmp.sort_values("|Avvikelse|", ascending=True, ignore_index=True)
        sort_text = f"närmast riktkurs • {rk_pos.lower()}"

    if kandidater.empty:
        st.info("Inga kandidater matchar valt läge/filtrering.")
        return

    # Portföljvärde för andelsberäkning
    pf = df[df["Antal aktier"] > 0].copy()
    pf["Växelkurs"] = pf["Valuta"].map(valutakurser).fillna(1.0)
    pf["Värde (SEK)"] = pf["Antal aktier"] * pf["Aktuell kurs"] * pf["Växelkurs"]
    portf_v = float(pf["Värde (SEK)"].sum())

    # Stabil bläddring
    sig = (riktkurs_val, endast_pf, sortläge, rk_pos, tuple(kandidater["Ticker"].astype(str).tolist()))
    if st.session_state.get("inv_sig") != sig:
        st.session_state["inv_sig"] = sig
        st.session_state["inv_i"] = 0
    if "inv_i" not in st.session_state: st.session_state["inv_i"] = 0
    n = len(kandidater)
    st.session_state["inv_i"] %= n

    cA,cB,cC = st.columns([1,1,4])
    with cA:
        if st.button("⬅️ Föregående", key="inv_prev"):
            st.session_state["inv_i"] = (st.session_state["inv_i"] - 1) % n
    with cB:
        if st.button("Nästa ➡️", key="inv_next"):
            st.session_state["inv_i"] = (st.session_state["inv_i"] + 1) % n
    with cC:
        st.caption(f"Förslag {st.session_state['inv_i']+1}/{n} • Sortering: {sort_text}")

    rad = kandidater.iloc[st.session_state["inv_i"]]
    kurs_sek = float(rad["Aktuell kurs"]) * float(rad["Växelkurs"])
    antal = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0
    investering_sek = antal * kurs_sek

    nuv_innehav = pf[pf["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum() if portf_v > 0 else 0.0
    ny_total = nuv_innehav + investering_sek
    nuv_andel = round((nuv_innehav/portf_v)*100.0, 2) if portf_v>0 else 0.0
    ny_andel  = round((ny_total/portf_v)*100.0, 2)   if portf_v>0 else 0.0

    def rn(title, val, active):
        return f"- **{title}:** {'**' if active else ''}{round(float(val),2)} {rad['Valuta']}{'**' if active else ''}"

    avv = float(rad["Avvikelse mot riktkurs (%)"])
    if avv < 0: avv_text = f"{abs(round(avv,2))}% under riktkurs"
    elif avv > 0: avv_text = f"{round(avv,2)}% över riktkurs"
    else: avv_text = "exakt på riktkurs"

    st.markdown(f"""
**{rad['Bolagsnamn']}** ({rad['Ticker']})

- **Aktuell kurs:** {round(float(rad['Aktuell kurs']),2)} {rad['Valuta']}

{rn("Riktkurs idag", float(rad.get("Riktkurs idag", 0.0)), riktkurs_val=="Riktkurs idag")}
{rn("Riktkurs om 1 år", float(rad.get("Riktkurs om 1 år", 0.0)), riktkurs_val=="Riktkurs om 1 år")}
{rn("Riktkurs om 2 år", float(rad.get("Riktkurs om 2 år", 0.0)), riktkurs_val=="Riktkurs om 2 år")}
{rn("Riktkurs om 3 år", float(rad.get("Riktkurs om 3 år", 0.0)), riktkurs_val=="Riktkurs om 3 år")}
""")

    if sortläge == "Max potential":
        st.markdown(f"- **Potential (utifrån valet ovan):** {round(float(rad['Potential (%)']),2)}%")
    else:
        st.markdown(f"- **Avstånd till vald riktkurs:** {avv_text}")

    st.markdown(f"""
- **Antal att köpa:** {antal} st
- **Beräknad investering:** {round(investering_sek,2)} SEK
- **Nuvarande andel i portföljen:** {nuv_andel}%
- **Andel efter köp:** {ny_andel}%
""")

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Sidomeny: manuella valutakurser (används endast i portföljvärden & SEK-kalkyl för köp)
    st.sidebar.header("💱 Valutakurser → SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(STANDARD_VALUTAKURSER["USD"]), step=0.01, key="fx_usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(STANDARD_VALUTAKURSER["NOK"]), step=0.01, key="fx_nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(STANDARD_VALUTAKURSER["CAD"]), step=0.01, key="fx_cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(STANDARD_VALUTAKURSER["EUR"]), step=0.01, key="fx_eur"),
    }

    meny = st.sidebar.radio("📌 Välj vy",
                            ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"],
                            index=0)

    if meny == "Analys":
        analysvy(df, valutakurser)
    elif meny == "Lägg till / uppdatera bolag":
        lagg_till_eller_uppdatera(df)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
