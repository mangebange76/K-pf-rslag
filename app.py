import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys & Investeringsförslag", layout="wide")

# ---- Google Sheets ----
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Kolumner (exakt enligt din lista) ----
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

def sakerstall_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Lägg till saknade kolumner
    for c in KOLUMNER:
        if c not in df.columns:
            df[c] = ""
    # Behåll endast definierade kolumner i rätt ordning
    df = df[KOLUMNER]
    return df

def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---- CAGR hjälpare ----
def justera_tillvaxt(cagr_pct):
    """
    >100% => 50%
    <0%   => 2%
    annars rå
    """
    if cagr_pct is None or pd.isna(cagr_pct):
        return None
    try:
        x = float(cagr_pct)
    except:
        return None
    if x > 100:
        return 50.0
    if x < 0:
        return 2.0
    return x

# ---- Valutakurs-hjälpare (rad → SEK) ----
def hamta_vaxelkurs_for_rad(rad, valutakurser: dict):
    val = str(rad.get("Valuta", "") or "").upper().strip()
    return float(valutakurser.get(val, 1.0))

# ---- Rå CAGR från historisk omsättning (Yahoo) ----
def hamta_hist_oms_ra_cagr(ticker: str):
    """
    Hämtar senaste och äldsta 'Total Revenue' från income statement och beräknar CAGR.
    Faller tillbaka på 'financials' om 'income_stmt' saknas (olika yfinance-versioner).
    """
    try:
        t = yf.Ticker(ticker)
        df_is = None
        for attr in ("income_stmt", "financials"):
            try:
                cand = getattr(t, attr)
                if isinstance(cand, pd.DataFrame) and "Total Revenue" in cand.index and cand.shape[1] >= 2:
                    df_is = cand
                    break
            except:
                pass
        if df_is is None:
            return None

        # Ordna kolumner i stigande tid (äldre -> nyare)
        cols_sorted = sorted(df_is.columns)
        start_col = cols_sorted[0]
        end_col   = cols_sorted[-1]
        oms_start = df_is.loc["Total Revenue", start_col]
        oms_slut  = df_is.loc["Total Revenue", end_col]

        # Antal år mellan punkterna (fallback om timestamp saknar år)
        try:
            years = max(1, (end_col.year - start_col.year))
        except:
            years = max(1, len(cols_sorted) - 1)

        if pd.isna(oms_start) or pd.isna(oms_slut) or float(oms_start) <= 0:
            return None

        cagr_dec = (float(oms_slut) / float(oms_start)) ** (1.0 / years) - 1.0
        return round(cagr_dec * 100.0, 2)
    except:
        return None

# ---- Hämta auto-fält för EN ticker och skriv in i df ----
def uppdatera_en_ticker_i_df(df: pd.DataFrame, ticker: str, fel_lista: list):
    t = (ticker or "").strip().upper()
    if not t:
        return df

    try:
        aktie = yf.Ticker(t)
        info = getattr(aktie, "info", {}) or {}

        bolagsnamn = info.get("longName") or info.get("shortName") or ""
        kurs       = info.get("currentPrice")
        valuta     = info.get("currency")
        utdelning  = info.get("dividendRate")

        cagr_ra = hamta_hist_oms_ra_cagr(t)

        # hitta eller skapa rad
        mask = df["Ticker"].astype(str).str.upper() == t
        if not mask.any():
            tom = {k: "" for k in KOLUMNER}
            tom["Ticker"] = t
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            mask = df["Ticker"].astype(str).str.upper() == t
        i = df[mask].index[0]

        if bolagsnamn:
            df.at[i, "Bolagsnamn"] = bolagsnamn
        if kurs is not None:
            df.at[i, "Aktuell kurs"] = kurs
        if valuta:
            df.at[i, "Valuta"] = valuta
        if utdelning is not None:
            df.at[i, "Årlig utdelning"] = utdelning
        if cagr_ra is not None:
            df.at[i, "CAGR 5 år (%)"] = cagr_ra

        miss = []
        if not bolagsnamn: miss.append("bolagsnamn")
        if kurs is None: miss.append("kurs")
        if not valuta: miss.append("valuta")
        if utdelning is None: miss.append("utdelning")
        if cagr_ra is None: miss.append("CAGR")
        if miss:
            fel_lista.append(f"{t}: saknar {', '.join(miss)}")

    except Exception as e:
        fel_lista.append(f"{t}: fel {e}")

    return df

def berakna_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # numeriska
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Antal aktier", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    df = _to_num(df, num_cols)

    # P/S-snitt av Q1..Q4 > 0
    qs = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
    ps_only = df[qs].where(df[qs] > 0)
    df["P/S-snitt"] = ps_only.mean(axis=1, skipna=True).fillna(0.0).round(2)

    # Proj omsättning (Omsättning nästa år * (1+adjCAGR)^exponent)
    def proj(bas, cagr_pct, exponent):
        if pd.isna(bas) or pd.isna(cagr_pct):
            return np.nan
        try:
            cagr_adj = justera_tillvaxt(float(cagr_pct))
            if cagr_adj is None:
                return np.nan
            return round(float(bas) * ((1.0 + cagr_adj / 100.0) ** exponent), 2)
        except:
            return np.nan

    df["Omsättning om 2 år"] = df.apply(lambda r: proj(r["Omsättning nästa år"], r["CAGR 5 år (%)"], 1), axis=1)
    df["Omsättning om 3 år"] = df.apply(lambda r: proj(r["Omsättning nästa år"], r["CAGR 5 år (%)"], 2), axis=1)

    # Riktkurser
    def rk(oms, ps, aktier):
        try:
            if pd.isna(oms) or pd.isna(ps) or pd.isna(aktier): return np.nan
            oms = float(oms); ps = float(ps); aktier = float(aktier)
            if aktier <= 0 or ps <= 0: return np.nan
            return round((oms * ps) / aktier, 2)
        except:
            return np.nan

    df["Riktkurs idag"]    = df.apply(lambda r: rk(r["Omsättning idag"],     r["P/S-snitt"], r["Utestående aktier"]), axis=1)
    df["Riktkurs om 1 år"] = df.apply(lambda r: rk(r["Omsättning nästa år"], r["P/S-snitt"], r["Utestående aktier"]), axis=1)
    df["Riktkurs om 2 år"] = df.apply(lambda r: rk(r["Omsättning om 2 år"],  r["P/S-snitt"], r["Utestående aktier"]), axis=1)
    df["Riktkurs om 3 år"] = df.apply(lambda r: rk(r["Omsättning om 3 år"],  r["P/S-snitt"], r["Utestående aktier"]), axis=1)

    return df

# ---- Formulär: Lägg till / uppdatera ----
def formular(df: pd.DataFrame):
    st.header("➕ Lägg till / uppdatera bolag")

    options = ["(nytt bolag)"] + sorted([t for t in df["Ticker"].astype(str).unique() if t.strip()])
    total_existing = len(options) - 1  # exkl. "(nytt bolag)"

    if "form_idx" not in st.session_state:
        st.session_state.form_idx = 0  # 0 = nytt bolag

    # Navigering
    cnav = st.columns([1,1,2,2])
    with cnav[0]:
        if st.button("⬅️ Föregående", use_container_width=True) and st.session_state.form_idx > 0:
            st.session_state.form_idx -= 1
    with cnav[1]:
        if st.button("Nästa ➡️", use_container_width=True) and st.session_state.form_idx < len(options) - 1:
            st.session_state.form_idx += 1
    with cnav[2]:
        pos_txt = "ny" if st.session_state.form_idx == 0 else f"{st.session_state.form_idx}/{total_existing}"
        st.caption(f"Post **{pos_txt}**")

    # Rullista
    vald = st.selectbox(
        "Välj i rullistan (eller bläddra ovan):",
        options,
        index=st.session_state.form_idx,
        key="form_selectbox",
    )
    st.session_state.form_idx = options.index(vald)

    # Hitta befintlig rad
    if st.session_state.form_idx == 0:
        bef = pd.Series(dtype="object")
    else:
        tick = options[st.session_state.form_idx]
        bef = df[df["Ticker"].astype(str) == tick].iloc[0] if not df[df["Ticker"].astype(str) == tick].empty else pd.Series(dtype="object")

    # Form
    with st.form("bolagsform"):
        ticker = st.text_input("Ticker", value=bef.get("Ticker", "") if not bef.empty else "").strip().upper()

        utest = st.number_input("Utestående aktier", min_value=0.0, value=float(bef.get("Utestående aktier", 0) or 0.0))
        antal = st.number_input("Antal aktier (ägda)", min_value=0.0, value=float(bef.get("Antal aktier", 0) or 0.0))

        ps  = st.number_input("P/S", value=float(bef.get("P/S", 0) or 0.0))
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0) or 0.0))
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0) or 0.0))
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0) or 0.0))
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0) or 0.0))

        oms_idag = st.number_input("Omsättning idag", value=float(bef.get("Omsättning idag", 0) or 0.0))
        oms_next = st.number_input("Omsättning nästa år", value=float(bef.get("Omsättning nästa år", 0) or 0.0))

        c1, c2, c3 = st.columns(3)
        with c1:
            btn_hamta = st.form_submit_button("🔎 Hämta från Yahoo (denna ticker)")
        with c2:
            btn_spara = st.form_submit_button("💾 Spara")
        with c3:
            btn_avbryt = st.form_submit_button("Avbryt")

    if btn_avbryt:
        st.stop()

    # Hämta auto-fält från Yahoo och uppdatera/skriv in i df
    if btn_hamta:
        if not ticker:
            st.warning("Ange en ticker först.")
            return df
        fel = []
        df = uppdatera_en_ticker_i_df(df, ticker, fel)
        df = berakna_kolumner(df)
        spara_data(df)
        st.success("Data hämtad och beräkningar uppdaterade.")
        if fel:
            st.info("Misslyckade/partiella fält (kopiera):")
            st.code("\n".join(fel))
        return df

    # Spara manuella fält, behåll auto-fält om de fanns
    if btn_spara:
        if not ticker:
            st.warning("Ange en ticker.")
            return df

        ny = {k: "" for k in KOLUMNER}
        ny.update({
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        })
        if not bef.empty:
            for k in ["Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"]:
                ny[k] = bef.get(k, ny[k])

        # Ersätt/append
        df = df[df["Ticker"].astype(str).str.upper() != ticker.upper()]
        df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)

        df = berakna_kolumner(df)
        spara_data(df)
        st.success("Bolag sparat och beräkningar uppdaterade.")
        # Håll fokus kvar
        new_opts = ["(nytt bolag)"] + sorted([t for t in df["Ticker"].astype(str).unique() if t.strip()])
        st.session_state.form_idx = new_opts.index(ticker) if ticker in new_opts else 0
        return df

    return df

# ---- Analys: bläddring + rullista + massuppdatering ----
def analysvy(df: pd.DataFrame):
    st.header("📈 Analys")

    options = sorted([t for t in df["Ticker"].astype(str).unique() if t.strip()])
    total = len(options)

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    if total == 0:
        st.info("Inga tickers i databasen ännu.")
        st.dataframe(df, use_container_width=True)
        return

    cols = st.columns([1,1,2,2])
    with cols[0]:
        if st.button("⬅️ Föregående", use_container_width=True) and st.session_state.analys_idx > 0:
            st.session_state.analys_idx -= 1
    with cols[1]:
        if st.button("Nästa ➡️", use_container_width=True) and st.session_state.analys_idx < total - 1:
            st.session_state.analys_idx += 1
    with cols[2]:
        st.caption(f"Post **{st.session_state.analys_idx + 1} / {total}**")

    val_via_select = st.selectbox(
        "Välj bolag (rullista):",
        options,
        index=st.session_state.analys_idx,
        key="analys_selectbox"
    )
    st.session_state.analys_idx = options.index(val_via_select)

    # Visa vald rad
    ticker = options[st.session_state.analys_idx]
    st.subheader(f"Detaljer: {ticker}")
    st.dataframe(df[df["Ticker"].astype(str) == ticker], use_container_width=True)

    st.markdown("---")
    st.subheader("Hela databasen")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Uppdatera ALLA från Yahoo (1 s mellan anrop)"):
        fel = []
        tot = len(df)
        bar = st.progress(0)
        status = st.empty()

        for i, t in enumerate(df["Ticker"].astype(str).tolist(), start=1):
            t_clean = (t or "").strip()
            if not t_clean:
                continue
            status.text(f"Hämtar {i}/{tot} — {t_clean}")
            df = uppdatera_en_ticker_i_df(df, t_clean, fel)
            time.sleep(1)
            bar.progress(i / tot)

        df = berakna_kolumner(df)
        spara_data(df)
        status.text("✅ Klar")
        st.success("Alla bolag uppdaterade.")
        if fel:
            st.info("Misslyckade/partiella fält (kopiera):")
            st.code("\n".join(fel))

# ---- Investeringsförslag: bläddring, alla riktkurser, andelar ----
def investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.header("💡 Investeringsförslag")

    val = st.selectbox(
        "Sortera och räkna uppsida baserat på:",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    d = df.copy()
    # säkra numerik
    for col in ["Aktuell kurs", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier", "Årlig utdelning"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["Uppside (%)"] = ((d[val] - d["Aktuell kurs"]) / d["Aktuell kurs"]) * 100.0
    d = d.replace([pd.NA, np.inf, -np.inf], np.nan).dropna(subset=["Aktuell kurs", val, "Uppside (%)"])
    d = d.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    if d.empty:
        st.info("Inga förslag – saknar värden för vald riktkurs.")
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
    vx = hamta_vaxelkurs_for_rad(rad, valutakurser)

    aktuell_kurs_sek = (rad["Aktuell kurs"] or 0.0) * vx
    rk_map = {
        "Riktkurs idag":  (rad["Riktkurs idag"] or 0.0) * vx,
        "Riktkurs om 1 år": (rad["Riktkurs om 1 år"] or 0.0) * vx,
        "Riktkurs om 2 år": (rad["Riktkurs om 2 år"] or 0.0) * vx,
        "Riktkurs om 3 år": (rad["Riktkurs om 3 år"] or 0.0) * vx,
    }

    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad['Ticker']})")
    st.write(f"Aktuell kurs: **{aktuell_kurs_sek:.2f} SEK**  (valuta: {rad.get('Valuta','')})")

    # Lista alla riktkurser – fetmarkera den valda
    for label, sek_val in rk_map.items():
        if label == val:
            st.markdown(f"- **{label}: {sek_val:.2f} SEK**")
        else:
            st.markdown(f"- {label}: {sek_val:.2f} SEK")

    # Uppsida i % baserat på VALD riktkurs
    st.write(f"Uppsida (baserat på *{val}*): **{rad['Uppside (%)']:.2f}%**")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", min_value=0.0, value=0.0, step=100.0)
    antal = int(kapital_sek // aktuell_kurs_sek) if aktuell_kurs_sek > 0 else 0
    investering_sek = antal * aktuell_kurs_sek
    st.write(f"Förslag: **{antal} st** (≈ {investering_sek:.2f} SEK)")

    # Andelar i portfölj före/efter
    d_port = df.copy()
    d_port["Antal aktier"] = pd.to_numeric(d_port["Antal aktier"], errors="coerce").fillna(0.0)
    d_port["Aktuell kurs"] = pd.to_numeric(d_port["Aktuell kurs"], errors="coerce").fillna(0.0)

    if (d_port["Antal aktier"] > 0).any():
        d_port["Växelkurs"] = d_port.apply(lambda r: hamta_vaxelkurs_for_rad(r, valutakurser), axis=1)
        d_port["Värde (SEK)"] = (d_port["Antal aktier"] * d_port["Aktuell kurs"] * d_port["Växelkurs"]).astype(float)

        portfoljvarde = float(d_port["Värde (SEK)"].sum())
        nuvarande_innehav = d_port.loc[
            d_port["Ticker"].astype(str).str.upper() == str(rad["Ticker"]).upper(),
            "Värde (SEK)"
        ].sum()

        nuvarande_andel = (nuvarande_innehav / portfoljvarde * 100.0) if portfoljvarde > 0 else 0.0
        ny_andel = ((nuvarande_innehav + investering_sek) / portfoljvarde * 100.0) if portfoljvarde > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Portföljvärde (SEK)", f"{portfoljvarde:,.0f}")
        c2.metric("Nuvarande andel", f"{nuvarande_andel:.2f}%")
        c3.metric("Andel efter köp", f"{ny_andel:.2f}%")
    else:
        st.info("Ingen registrerad portfölj (Antal aktier = 0 på alla rader).")

# ---- Portfölj ----
def portfolj(df: pd.DataFrame, valutakurser: dict):
    st.header("📦 Portfölj")
    d = df.copy()
    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)
    d["Årlig utdelning"] = pd.to_numeric(d["Årlig utdelning"], errors="coerce").fillna(0.0)

    agda = d[d["Antal aktier"] > 0].copy()
    if agda.empty:
        st.info("Inga innehav registrerade ännu.")
        return

    agda["Växelkurs"] = agda.apply(lambda r: hamta_vaxelkurs_for_rad(r, valutakurser), axis=1)
    agda["Värde (SEK)"] = (agda["Antal aktier"] * agda["Aktuell kurs"] * agda["Växelkurs"]).round(2)
    agda["Utdelning/år (SEK)"] = (agda["Antal aktier"] * agda["Årlig utdelning"] * agda["Växelkurs"]).round(2)

    tot_varde = float(agda["Värde (SEK)"].sum())
    tot_utd = float(agda["Utdelning/år (SEK)"].sum())
    man = tot_utd / 12.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Totalt portföljvärde", f"{tot_varde:,.0f} SEK")
    c2.metric("Total årlig utdelning", f"{tot_utd:,.0f} SEK")
    c3.metric("Utdelning per månad (snitt)", f"{man:,.0f} SEK")

    st.dataframe(
        agda[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Årlig utdelning","Utdelning/år (SEK)"]],
        use_container_width=True
    )

# ---- MAIN ----
def main():
    st.title("📊 Aktieanalys & Investeringsförslag")

    # Valutakurser till SEK (manuellt med hårdkodade startvärden enligt ditt senaste beslut)
    st.sidebar.header("💱 Valutakurser → SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "SEK": 1.0,
    }

    df = hamta_data()
    df = sakerstall_kolumner(df)
    df = berakna_kolumner(df)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera":
        df = formular(df)
    elif meny == "Investeringsförslag":
        investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
