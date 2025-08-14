import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# 🗄️ Google Sheets
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# 💱 Standard-växelkurser (SEK per 1 enhet av respektive valuta)
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# Utils: GSheet IO
# ─────────────────────────────────────────────────────────────────────────────
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data() -> pd.DataFrame:
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# Kolumner & typkonvertering
# ─────────────────────────────────────────────────────────────────────────────
NUKOL = [
    "Ticker","Bolagsnamn","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Valuta","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt",
    "Omsättningsvaluta",              # NY
    "Senast manuell uppdatering"      # NY
]

NUMCOLS = [
    "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUKOL:
        if c not in df.columns:
            if c in NUMCOLS:
                df[c] = 0.0
            elif c in ["Valuta","Omsättningsvaluta","Ticker","Bolagsnamn","Senast manuell uppdatering"]:
                df[c] = ""
            else:
                df[c] = ""
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMCOLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # trim str
    for c in ["Ticker","Bolagsnamn","Valuta","Omsättningsvaluta","Senast manuell uppdatering"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()
    return df

def parse_float_blank(txt: str) -> float:
    """Accepterar tom str -> 0.0, ersätter komma med punkt, strippar mellanslag."""
    if txt is None:
        return 0.0
    s = str(txt).replace(" ", "").replace(",", ".").strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Chip/etikett (återanvänds i flera vyer)
# ─────────────────────────────────────────────────────────────────────────────
def visa_chip_sortlage():
    if "inv_sort_mode" not in st.session_state:
        st.session_state.inv_sort_mode = "Störst uppsida"
    aktivt_text = "Uppsida" if st.session_state.inv_sort_mode.startswith("Störst") else "Närmast riktkurs"
    chip_color = "#e6f4ea" if aktivt_text == "Uppsida" else "#e8f0fe"
    chip_border = "#34a853" if aktivt_text == "Uppsida" else "#1a73e8"
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:4px 10px;
            border-radius:999px;
            background:{chip_color};
            border:1px solid {chip_border};
            font-weight:600;
            font-size:0.9rem;
            margin:2px 0 10px 0;">
            Läge: {aktivt_text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Yahoo: kurs/valuta/bolagsnamn/utdelning + CAGR (intäkt) 5 år
# ─────────────────────────────────────────────────────────────────────────────
def hamta_kurs_valuta_info(ticker: str):
    """Returnerar (pris, valuta, bolagsnamn, årlig utdelning per aktie) eller (None, 'USD', '', 0.0)."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        namn = info.get("shortName") or info.get("longName") or ""
        utd = info.get("dividendRate", 0.0) or 0.0  # per aktie, per år (om finns)
        return pris, valuta, namn, float(utd) if utd is not None else 0.0
    except Exception:
        return None, "USD", "", 0.0

def hamta_cagr_5ar_revenue(ticker: str) -> float:
    """
    Försöker beräkna CAGR på intäkter ~5 år bakåt:
    CAGR = (Rev_slut / Rev_start) ** (1/years) - 1
    Om färre än 5 år, använder tillgängliga år.
    Returnerar i procent (t.ex. 12.3), 0.0 om ej möjligt.
    """
    try:
        t = yf.Ticker(ticker)
        fin = t.financials  # årliga
        if fin is None or fin.empty:
            return 0.0
        row_candidates = ["Total Revenue", "TotalRevenue", "Revenue"]
        rev_row = None
        for rc in row_candidates:
            if rc in fin.index:
                rev_row = fin.loc[rc]
                break
        if rev_row is None or rev_row.empty:
            return 0.0
        # Kolumner är år (senaste först). Ta min 2 värden.
        vals = rev_row.dropna().values.astype(float)
        if len(vals) < 2:
            return 0.0
        # ta längst möjliga spann upp till ~5 år
        start = vals[-1]
        end = vals[0]
        years = len(vals) - 1  # intervall
        if start <= 0 or end <= 0 or years <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return float(cagr * 100.0)
    except Exception:
        return 0.0

def effekt_tillvaxt(cagr_pct: float) -> float:
    """
    Regler för framåträkning (används för omsättning om 2 & 3 år):
    - Om CAGR > 100% → använd 50% (0.50)
    - Om CAGR < 0%   → använd 2% (0.02)
    - Annars använd cagr_pct/100
    Returnerar faktor (t.ex. 0.12 för 12%).
    """
    if cagr_pct > 100.0:
        return 0.50
    if cagr_pct < 0.0:
        return 0.02
    return cagr_pct / 100.0

def valutakurs_sek(valuta: str, user_rates: dict) -> float:
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

# ─────────────────────────────────────────────────────────────────────────────
# Beräkningar: P/S-snitt, riktkurser & omsättningar (med valutajustering)
# ─────────────────────────────────────────────────────────────────────────────
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    - P/S-snitt = medel av P/S Q1..Q4 > 0
    - Beräknar Omsättning om 2 & 3 år via CAGR-regler (om CAGR 5 år (%) finns)
    - Riktkurser: använder omsättning * (växlad till aktiens valuta) och P/S-snitt
      dividerat på Utestående aktier (miljoner → per-aktie i aktiens valuta).
    """
    df = df.copy()

    # P/S-snitt
    for i, rad in df.iterrows():
        ps_list = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_pos = [float(x) for x in ps_list if pd.to_numeric(x, errors="coerce") and float(x) > 0]
        ps_snitt = round(float(np.mean(ps_pos)), 2) if ps_pos else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

    # Omsättning om 2 & 3 år (om man inte matat in – eller så skriver vi om när som helst)
    for i, rad in df.iterrows():
        cagr_pct = float(rad.get("CAGR 5 år (%)", 0.0))
        g = effekt_tillvaxt(cagr_pct)
        oms_next = float(rad.get("Omsättning nästa år", 0.0))  # år 1
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * (1.0 + g) * (1.0 + g), 2)

    # Riktkurser (räknas med omsättning konverterad till aktiens valuta innan P/S/aktier)
    for i, rad in df.iterrows():
        ps_snitt = float(rad.get("P/S-snitt", 0.0))
        shares_m = float(rad.get("Utestående aktier", 0.0))
        aktie_val = (rad.get("Valuta") or "USD").strip().upper()
        oms_val = (rad.get("Omsättningsvaluta") or aktie_val).strip().upper()

        if shares_m <= 0 or ps_snitt <= 0:
            continue

        # konverteringsfaktor från oms_valuta till aktie_valuta
        sek_per_oms = valutakurs_sek(oms_val, user_rates)
        sek_per_aktie = valutakurs_sek(aktie_val, user_rates)
        if sek_per_oms <= 0 or sek_per_aktie <= 0:
            continue
        fx = sek_per_oms / sek_per_aktie  # multiplicera omsättning (miljoner) med fx

        def rikt(oms_m):
            if oms_m <= 0:
                return 0.0
            return round((oms_m * fx * ps_snitt) / shares_m, 2)

        df.at[i, "Riktkurs idag"]   = rikt(float(rad.get("Omsättning idag", 0.0)))
        df.at[i, "Riktkurs om 1 år"] = rikt(float(rad.get("Omsättning nästa år", 0.0)))
        df.at[i, "Riktkurs om 2 år"] = rikt(float(rad.get("Omsättning om 2 år", 0.0)))
        df.at[i, "Riktkurs om 3 år"] = rikt(float(rad.get("Omsättning om 3 år", 0.0)))

    return df

# ─────────────────────────────────────────────────────────────────────────────
# Lägg till / uppdatera bolag (med rullista + bläddring). Blank-friendly inputs.
# ─────────────────────────────────────────────────────────────────────────────
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")
    # rullista namn (alfabetiskt) + bläddring
    namn_list = [f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})".strip() for _, row in df.iterrows()]
    namn_map = {f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})".strip(): row.get('Ticker','') for _, row in df.iterrows()}
    namn_list_sorted = sorted(namn_list)

    if "edit_list" not in st.session_state:
        st.session_state.edit_list = namn_list_sorted
    if "edit_idx" not in st.session_state:
        st.session_state.edit_idx = 0

    valt = st.selectbox("Välj bolag", [""] + namn_list_sorted, index=0, key="edit_select")

    # Om användaren väljer i rullistan – hoppa till det indexet
    if valt and valt in st.session_state.edit_list:
        st.session_state.edit_idx = st.session_state.edit_list.index(valt)

    # Bläddringsknappar + indikator
    total = len(st.session_state.edit_list)
    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("⬅️ Föregående", key="edit_prev") and total > 0:
            st.session_state.edit_idx = (st.session_state.edit_idx - 1) % total
    with c2:
        st.markdown(f"<div style='text-align:center'>Post <b>{(st.session_state.edit_idx+1) if total else 0}</b> / <b>{total}</b></div>", unsafe_allow_html=True)
    with c3:
        if st.button("Nästa ➡️", key="edit_next") and total > 0:
            st.session_state.edit_idx = (st.session_state.edit_idx + 1) % total

    # Hämta befintlig rad
    if total > 0 and st.session_state.edit_list:
        try:
            tick = namn_map.get(st.session_state.edit_list[st.session_state.edit_idx], "")
            befintlig = df[df["Ticker"] == tick].iloc[0] if tick else pd.Series(dtype=object)
        except Exception:
            befintlig = pd.Series(dtype=object)
    else:
        befintlig = pd.Series(dtype=object)

    # Formulär (text_input för blankvänligt)
    with st.form("form_bolag"):
        ticker = st.text_input("Ticker", value=str(befintlig.get("Ticker",""))).upper()
        namn = st.text_input("Bolagsnamn (hämtas från Yahoo om tomt)", value=str(befintlig.get("Bolagsnamn","")))
        akt_val = st.selectbox("Valuta (aktiekursens valuta)", ["USD","NOK","CAD","SEK","EUR"],
                               index=(["USD","NOK","CAD","SEK","EUR"].index(befintlig.get("Valuta","USD")) if not pd.isna(befintlig.get("Valuta","USD")) and befintlig.get("Valuta","USD") in ["USD","NOK","CAD","SEK","EUR"] else 0))

        oms_val = st.selectbox("Omsättningsvaluta", ["USD","NOK","CAD","SEK","EUR"],
                               index=(["USD","NOK","CAD","SEK","EUR"].index(befintlig.get("Omsättningsvaluta", akt_val)) if not pd.isna(befintlig.get("Omsättningsvaluta", akt_val)) and befintlig.get("Omsättningsvaluta", akt_val) in ["USD","NOK","CAD","SEK","EUR"] else 0))

        uo = st.text_input("Utestående aktier (miljoner)", value=str(befintlig.get("Utestående aktier","")))
        antal_eget = st.text_input("Antal aktier du äger", value=str(befintlig.get("Antal aktier","")))

        # P/S-fält (manuella)
        ps_idag = st.text_input("P/S (nuvarande)", value=str(befintlig.get("P/S","")))
        ps1 = st.text_input("P/S Q1", value=str(befintlig.get("P/S Q1","")))
        ps2 = st.text_input("P/S Q2", value=str(befintlig.get("P/S Q2","")))
        ps3 = st.text_input("P/S Q3", value=str(befintlig.get("P/S Q3","")))
        ps4 = st.text_input("P/S Q4", value=str(befintlig.get("P/S Q4","")))

        # Omsättningar (manuella för idag & nästa år)
        oms_idag = st.text_input("Omsättning idag (miljoner)", value=str(befintlig.get("Omsättning idag","")))
        oms_next = st.text_input("Omsättning nästa år (miljoner)", value=str(befintlig.get("Omsättning nästa år","")))

        # Visning (icke redigerbara här – uppdateras via Yahoo om går)
        st.caption("Följande fält hämtas automatiskt vid spar (om möjligt): Bolagsnamn, Aktuell kurs, Årlig utdelning, Valuta (om tom), CAGR 5 år (%).")

        spar = st.form_submit_button("💾 Spara bolag")

    if spar:
        if not ticker:
            st.warning("Ange Ticker.")
            return df

        # Convertera manuella strängar → float
        new_row = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Valuta": akt_val,
            "Omsättningsvaluta": oms_val,
            "Utestående aktier": parse_float_blank(uo),
            "Antal aktier": parse_float_blank(antal_eget),
            "P/S": parse_float_blank(ps_idag),
            "P/S Q1": parse_float_blank(ps1),
            "P/S Q2": parse_float_blank(ps2),
            "P/S Q3": parse_float_blank(ps3),
            "P/S Q4": parse_float_blank(ps4),
            "Omsättning idag": parse_float_blank(oms_idag),
            "Omsättning nästa år": parse_float_blank(oms_next),
            # följande uppdateras strax
            "Aktuell kurs": float(befintlig.get("Aktuell kurs", 0.0)),
            "Årlig utdelning": float(befintlig.get("Årlig utdelning", 0.0)),
            "CAGR 5 år (%)": float(befintlig.get("CAGR 5 år (%)", 0.0)),
            "Senast manuell uppdatering": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Hämta från Yahoo (namn/kurs/valuta/utdelning + CAGR)
        pris, valuta, namn_auto, utd = hamta_kurs_valuta_info(ticker)
        cagr_auto = hamta_cagr_5ar_revenue(ticker)
        if pris is not None and pris > 0:
            new_row["Aktuell kurs"] = float(pris)
        if valuta and valuta.strip():
            new_row["Valuta"] = valuta.strip().upper()
        if namn_auto and not new_row["Bolagsnamn"]:
            new_row["Bolagsnamn"] = namn_auto
        if utd is not None and utd >= 0:
            new_row["Årlig utdelning"] = float(utd)
        if cagr_auto is not None and cagr_auto >= 0:
            new_row["CAGR 5 år (%)"] = float(cagr_auto)

        # Slutligen – beräkna resten (inkl. riktkurser & oms. 2/3 år)
        tmp_df = df.copy()
        if ticker in tmp_df["Ticker"].values:
            tmp_df.loc[tmp_df["Ticker"] == ticker, list(new_row.keys())] = list(new_row.values())
        else:
            tmp_df = pd.concat([tmp_df, pd.DataFrame([new_row])], ignore_index=True)

        tmp_df = uppdatera_berakningar(tmp_df, user_rates)
        spara_data(tmp_df)
        st.success(f"{ticker} sparat och beräkningar uppdaterade.")
        return tmp_df

    return df

# ─────────────────────────────────────────────────────────────────────────────
# Portfölj
# ─────────────────────────────────────────────────────────────────────────────
def visa_portfolj(df: pd.DataFrame, user_rates: dict):
    st.subheader("📦 Min portfölj")
    visa_chip_sortlage()

    df = df.copy()
    if df.empty or "Antal aktier" not in df.columns:
        st.info("Ingen data att visa.")
        return

    df["Växelkurs"] = df["Valuta"].apply(lambda v: valutakurs_sek(v, user_rates))
    df["Värde (SEK)"] = pd.to_numeric(df["Antal aktier"], errors="coerce").fillna(0.0) * \
                        pd.to_numeric(df["Aktuell kurs"], errors="coerce").fillna(0.0) * \
                        df["Växelkurs"]
    df["Andel (%)"] = (df["Värde (SEK)"] / df["Värde (SEK)"].sum()).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 100

    df["Total årlig utdelning (SEK)"] = pd.to_numeric(df["Antal aktier"], errors="coerce").fillna(0.0) * \
                                        pd.to_numeric(df["Årlig utdelning"], errors="coerce").fillna(0.0) * \
                                        df["Växelkurs"]

    total_utdelning = df["Total årlig utdelning (SEK)"].sum()
    total_varde = df["Värde (SEK)"].sum()

    st.markdown(f"**Totalt portföljvärde:** {total_varde:,.2f} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {total_utdelning:,.2f} SEK")
    st.markdown(f"**Genomsnittlig månadsutdelning:** {total_utdelning/12:,.2f} SEK")

    # Uppdatera alla från Yahoo (1s delay) – lagd här enligt önskemål
    if st.button("🔄 Uppdatera alla från Yahoo"):
        misslyckade = []
        uppdaterade = 0
        total = len(df)
        bar = st.progress(0)
        status = st.empty()

        for i, row in df.iterrows():
            ticker = str(row.get("Ticker","")).strip().upper()
            status.text(f"Uppdaterar {i+1}/{total} – {ticker}")
            try:
                p, cur, namn_auto, utd = hamta_kurs_valuta_info(ticker)
                if p is None or p <= 0:
                    misslyckade.append(ticker)
                else:
                    df.at[i, "Aktuell kurs"] = float(p)
                    if cur: df.at[i, "Valuta"] = cur.strip().upper()
                    if namn_auto and not str(df.at[i,"Bolagsnamn"]).strip():
                        df.at[i, "Bolagsnamn"] = namn_auto
                    if utd is not None and utd >= 0:
                        df.at[i, "Årlig utdelning"] = float(udt)
                    # hämta/uppdatera CAGR 5 år (%)
                    cagr_auto = hamta_cagr_5ar_revenue(ticker)
                    if cagr_auto is not None and cagr_auto >= 0:
                        df.at[i, "CAGR 5 år (%)"] = float(cagr_auto)
                    uppdaterade += 1
            except Exception:
                misslyckade.append(ticker)
            bar.progress((i+1)/total)
            time.sleep(1)

        # räkna om beräkningar efter uppdatering
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        status.text("✅ Klar.")
        st.success(f"Uppdaterade {uppdaterade} av {total} bolag.")
        if misslyckade:
            st.warning("Misslyckades med: " + ", ".join(misslyckade))

    st.dataframe(
        df[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Analys (databasvy + filtrering + bläddring)
# ─────────────────────────────────────────────────────────────────────────────
def analysvy(df: pd.DataFrame, user_rates: dict):
    st.subheader("📈 Analysläge")
    visa_chip_sortlage()

    if df.empty:
        st.info("Ingen data ännu.")
        return

    # Filtrera ett bolag (rullista) + bläddra
    namn_list = [f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})".strip() for _, row in df.iterrows()]
    namn_map = {f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})".strip(): row.get('Ticker','') for _, row in df.iterrows()}
    namn_list_sorted = sorted(namn_list)

    if "an_list" not in st.session_state:
        st.session_state.an_list = namn_list_sorted
    if "an_idx" not in st.session_state:
        st.session_state.an_idx = 0

    valt = st.selectbox("Visa bolag", [""] + namn_list_sorted, index=0, key="an_select")
    if valt and valt in st.session_state.an_list:
        st.session_state.an_idx = st.session_state.an_list.index(valt)

    total = len(st.session_state.an_list)
    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("⬅️ Föregående", key="an_prev") and total > 0:
            st.session_state.an_idx = (st.session_state.an_idx - 1) % total
    with c2:
        st.markdown(f"<div style='text-align:center'>Post <b>{(st.session_state.an_idx+1) if total else 0}</b> / <b>{total}</b></div>", unsafe_allow_html=True)
    with c3:
        if st.button("Nästa ➡️", key="an_next") and total > 0:
            st.session_state.an_idx = (st.session_state.an_idx + 1) % total

    # Visa endast valt bolag överst
    if total > 0:
        try:
            tick = namn_map.get(st.session_state.an_list[st.session_state.an_idx], "")
            one = df[df["Ticker"] == tick].copy() if tick else pd.DataFrame()
        except Exception:
            one = pd.DataFrame()
        if not one.empty:
            st.dataframe(one, use_container_width=True, height=200)

    # Hela tabellen under
    st.markdown("—")
    st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Investeringsförslag
# ─────────────────────────────────────────────────────────────────────────────
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict):
    st.subheader("💡 Investeringsförslag")

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1,
        key="inv_riktkurs_select"
    )

    # Sorteringsläge (Uppsida vs Närmast riktkurs)
    if "inv_sort_mode" not in st.session_state:
        st.session_state.inv_sort_mode = "Störst uppsida"

    csm1, csm2, _ = st.columns([1,1,6])
    with csm1:
        if st.button("⬆ Uppsida", key="btn_sort_upside"):
            st.session_state.inv_sort_mode = "Störst uppsida"
            st.experimental_rerun()
    with csm2:
        if st.button("↔ Riktkurs", key="btn_sort_nearest"):
            st.session_state.inv_sort_mode = "Närmast riktkurs (±%)"
            st.experimental_rerun()

    # Chip
    visa_chip_sortlage()

    filterval = st.radio("Visa förslag för:", ["Alla bolag","Endast portföljen"], horizontal=True, key="inv_filter_radio")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0, key="inv_kapital_sek")

    df = df.copy()
    df["Växelkurs"] = df["Valuta"].apply(lambda v: valutakurs_sek(v, user_rates))

    bas = df.copy()
    if filterval == "Endast portföljen":
        bas = bas[pd.to_numeric(bas["Antal aktier"], errors="coerce").fillna(0.0) > 0]

    bas["Aktuell kurs"] = pd.to_numeric(bas["Aktuell kurs"], errors="coerce")
    bas[riktkurs_val] = pd.to_numeric(bas[riktkurs_val], errors="coerce")
    bas = bas[(bas["Aktuell kurs"] > 0) & (bas[riktkurs_val] > 0)].copy()
    if bas.empty:
        st.info("Inga bolag matchar kriterierna.")
        return

    # Uppsida & avvikelse
    bas["Potential (%)"] = ((bas[riktkurs_val] - bas["Aktuell kurs"]) / bas["Aktuell kurs"]) * 100
    bas["Avvikelse (%)"] = ((bas["Aktuell kurs"] - bas[riktkurs_val]) / bas[riktkurs_val]) * 100
    bas["|Avvikelse| (%)"] = bas["Avvikelse (%)"].abs()

    if st.session_state.inv_sort_mode == "Störst uppsida":
        bas = bas.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        bas = bas.sort_values(by="|Avvikelse| (%)", ascending=True).reset_index(drop=True)

    # Bläddring
    if "inv_list_tickers" not in st.session_state:
        st.session_state.inv_list_tickers = bas["Ticker"].tolist()
    if "inv_idx" not in st.session_state:
        st.session_state.inv_idx = 0

    curr_list = bas["Ticker"].tolist()
    if st.session_state.inv_list_tickers != curr_list:
        st.session_state.inv_list_tickers = curr_list
        st.session_state.inv_idx = 0

    total = len(st.session_state.inv_list_tickers)
    idx = st.session_state.inv_idx
    if idx >= total:
        idx = 0
        st.session_state.inv_idx = 0

    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("⬅️ Föregående", key="inv_prev") and total:
            st.session_state.inv_idx = (st.session_state.inv_idx - 1) % total
            st.experimental_rerun()
    with c2:
        st.markdown(f"<div style='text-align:center'>Förslag <b>{idx+1}</b> / <b>{total}</b></div>", unsafe_allow_html=True)
    with c3:
        if st.button("Nästa ➡️", key="inv_next") and total:
            st.session_state.inv_idx = (st.session_state.inv_idx + 1) % total
            st.experimental_rerun()

    vald_ticker = st.session_state.inv_list_tickers[idx]
    rad = bas[bas["Ticker"] == vald_ticker].iloc[0]

    # Antal att köpa för X SEK
    vx = float(rad["Växelkurs"]) if pd.notna(rad["Växelkurs"]) and float(rad["Växelkurs"]) > 0 else 1.0
    pris_i_sek = float(rad["Aktuell kurs"]) * vx
    antal = int(kapital_sek // max(pris_i_sek, 1e-9)) if kapital_sek > 0 and pris_i_sek > 0 else 0
    investering_sek = antal * pris_i_sek

    # Portföljandel före/efter
    df_port = df[pd.to_numeric(df["Antal aktier"], errors="coerce").fillna(0.0) > 0].copy()
    df_port["Värde (SEK)"] = pd.to_numeric(df_port["Antal aktier"], errors="coerce").fillna(0.0) * \
                             pd.to_numeric(df_port["Aktuell kurs"], errors="coerce").fillna(0.0) * \
                             df_port["Växelkurs"].astype(float)
    portfoljvarde = df_port["Värde (SEK)"].sum()
    nuvarande_innehav_sek = df_port.loc[df_port["Ticker"] == rad["Ticker"], "Värde (SEK)"].sum() if not df_port.empty else 0.0
    nuvarande_andel = round((nuvarande_innehav_sek / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0.0
    ny_andel = round(((nuvarande_innehav_sek + investering_sek) / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0.0

    def fmt(x):
        try: return f"{float(x):.2f}"
        except: return "-"

    riktkurser = {
        "Riktkurs idag": rad.get("Riktkurs idag", 0.0),
        "Riktkurs om 1 år": rad.get("Riktkurs om 1 år", 0.0),
        "Riktkurs om 2 år": rad.get("Riktkurs om 2 år", 0.0),
        "Riktkurs om 3 år": rad.get("Riktkurs om 3 år", 0.0),
    }
    lines = []
    for rub, val in riktkurser.items():
        text = f"**{rub}: {fmt(val)} {rad['Valuta']}**" if rub == riktkurs_val else f"{rub}: {fmt(val)} {rad['Valuta']}"
        lines.append(f"- {text}")

    extra_rad = (
        f"- Uppsida (mot *{riktkurs_val}*): **{rad['Potential (%)']:.2f}%**"
        if st.session_state.inv_sort_mode == "Störst uppsida"
        else f"- Avvikelse mot *{riktkurs_val}*: **{rad['Avvikelse (%)']:.2f}%** (positiv = över riktkurs)"
    )

    st.markdown(
        f"""
### 💰 {rad['Bolagsnamn']} ({rad['Ticker']})
- Aktuell kurs: **{fmt(rad['Aktuell kurs'])} {rad['Valuta']}**
{chr(10).join(lines)}
{extra_rad}
- Antal att köpa för {int(kapital_sek)} SEK: **{antal} st**
- Beräknad investering: **{investering_sek:,.2f} SEK**
- Nuvarande andel i portföljen: **{nuvarande_andel:.2f}%**
- Andel efter köp: **{ny_andel:.2f}%**
""".strip()
    )

# ─────────────────────────────────────────────────────────────────────────────
# Huvudprogram
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Ladda data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Sidebar: valutakurser → SEK
    st.sidebar.header("💱 Valutakurser → SEK")
    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=STANDARD_VALUTAKURSER["USD"], step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=STANDARD_VALUTAKURSER["NOK"], step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=STANDARD_VALUTAKURSER["CAD"], step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=STANDARD_VALUTAKURSER["EUR"], step=0.01),
        "SEK": 1.00
    }

    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Analys":
        # uppdatera beräkningar för visningen (sparas ej förrän man explicit sparar)
        df_view = uppdatera_berakningar(df, user_rates)
        analysvy(df_view, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df_new = lagg_till_eller_uppdatera(df, user_rates)
        # om ändrat, df_new returneras. Visa efter spar?
        if not df_new.equals(df):
            df = df_new

    elif meny == "Investeringsförslag":
        df_view = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df_view, user_rates)

    elif meny == "Portfölj":
        df_view = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df_view, user_rates)


if __name__ == "__main__":
    main()
