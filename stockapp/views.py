# stockapp/views.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np

from .utils import (
    TS_FIELDS,
    add_oldest_ts_col,
    ensure_schema,
    konvertera_typer,
    uppdatera_berakningar,
)

# (valfritt) single-ticker uppdateringar – används här i formulär-vyn
try:
    from .sources import run_update_full, run_update_price_only
    _HAS_SOURCES = True
except Exception:
    _HAS_SOURCES = False


# =========================
# Hjälpare
# =========================
def _fx(user_rates: Dict[str, float], valuta: str) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(str(valuta).upper(), 1.0))

def _fmt_money(n: float, ccy: str = "") -> str:
    try:
        v = float(n)
    except Exception:
        return "-"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        s = f"{v/1e12:.2f} T"
    elif v >= 1e9:
        s = f"{v/1e9:.2f} B"
    elif v >= 1e6:
        s = f"{v/1e6:.2f} M"
    elif v >= 1e3:
        s = f"{v/1e3:.2f} k"
    else:
        s = f"{v:.2f}"
    return f"{sign}{s}{(' ' + ccy) if ccy else ''}"

def _implied_mcap(row: pd.Series) -> float:
    """Marketcap = pris * (utestående aktier i miljoner * 1e6). Returnerar i samma valuta som priset."""
    try:
        px = float(row.get("Aktuell kurs", 0.0))
        shares_mil = float(row.get("Utestående aktier", 0.0))
    except Exception:
        return 0.0
    if px <= 0 or shares_mil <= 0:
        return 0.0
    return px * shares_mil * 1_000_000.0

def _manuell_prognos_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lista bolag där 'Omsättning idag' och/eller 'Omsättning nästa år' saknar TS
    eller där TS är äldst bland spårade fält – för att stötta manuell uppdatering.
    """
    need_cols = ["Omsättning idag", "Omsättning nästa år"]
    ts_cols = [TS_FIELDS[c] for c in need_cols if c in TS_FIELDS]
    work = add_oldest_ts_col(df.copy())

    rows: List[dict] = []
    for _, r in work.iterrows():
        miss_val = any((float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
        miss_ts = any((not str(r.get(ts, "")).strip()) for ts in ts_cols if ts in r)
        oldest = r.get("_oldest_any_ts")
        if miss_val or miss_ts:
            rows.append({
                "Ticker": r.get("Ticker", ""),
                "Bolagsnamn": r.get("Bolagsnamn", ""),
                "TS_Omsättning idag": r.get("TS_Omsättning idag", ""),
                "TS_Omsättning nästa år": r.get("TS_Omsättning nästa år", ""),
                "Äldsta TS (alla)": "" if pd.isna(oldest) else str(pd.to_datetime(oldest).date()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # sortera på äldsta TS för de två fälten (string sort här duger; annars kan man parsea datum)
        out = out.sort_values(by=["TS_Omsättning idag", "TS_Omsättning nästa år", "Bolagsnamn"], ascending=[True, True, True])
    return out


# =========================
# Vy: Kontroll
# =========================
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade – topp 20
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"]).head(20)
    cols_show = ["Ticker", "Bolagsnamn"]
    for k in ["TS_Utestående aktier", "TS_P/S", "TS_P/S Q1", "TS_P/S Q2", "TS_P/S Q3", "TS_P/S Q4",
              "TS_Omsättning idag", "TS_Omsättning nästa år"]:
        if k in vis.columns:
            cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Senaste körlogg (om batch-panelen satt den)
    st.subheader("📒 Senaste körlogg (batch)")
    log = st.session_state.get("last_batch_log")
    if not log:
        st.info("Ingen batch-körning i denna session ännu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar** (ticker → fält)")
            st.json(log.get("changed", {}) or {})
        with col2:
            st.markdown("**Missar** (ticker → skäl)")
            st.json(log.get("misses", {}) or {})
        if "queue_info" in log:
            st.markdown("**Kö-info**")
            st.json(log["queue_info"])


# =========================
# Vy: Analys
# =========================
def analysvy(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📈 Analys")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0,
        max_value=max(0, len(etiketter) - 1),
        value=st.session_state.analys_idx,
        step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter) - 1, st.session_state.analys_idx + 1)

    st.write(f"Post {st.session_state.analys_idx + 1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    # Små etiketter för uppdateringsstatus
    chips = []
    if str(r.get("Senast manuellt uppdaterad", "")).strip():
        chips.append(f"🖐 Manuell: {r.get('Senast manuellt uppdaterad')}")
    if str(r.get("Senast auto-uppdaterad", "")).strip():
        chips.append(f"🤖 Auto: {r.get('Senast auto-uppdaterad')} ({r.get('Senast uppdaterad källa','-')})")
    if chips:
        st.caption(" | ".join(chips))

    cols = [
        "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "CAGR 5 år (%)", "Antal aktier", "Årlig utdelning",
        "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
        "TS_Utestående aktier", "TS_P/S", "TS_P/S Q1", "TS_P/S Q2", "TS_P/S Q3", "TS_P/S Q4",
        "TS_Omsättning idag", "TS_Omsättning nästa år",
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)


# =========================
# Vy: Lägg till / uppdatera bolag
# =========================
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> Optional[pd.DataFrame]:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsläge i vyn
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)", "Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])

    # Val av befintligt bolag (eller nytt)
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista) - 1))
    col_prev, col_pos, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista) - 1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista) - 1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker", "") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier", 0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier", 0.0)) if not bef.empty else 0.0)
            gav_sek = st.number_input("GAV (SEK)", value=float(bef.get("GAV (SEK)", 0.0)) if not bef.empty else 0.0)

            ps = st.number_input("P/S", value=float(bef.get("P/S", 0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag", 0.0)) if not bef.empty else 0.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- Riktkurser/beräkningar räknas om")
            st.write("- Inga automatkällor körs här (batchpanelen gör det).")

        spar = st.form_submit_button("💾 Spara")

    # Sparaformulär – validera dubbletter
    if spar and ticker:
        ticker = ticker.strip().upper()
        # Blockera dubbletter när man skapar nytt
        if bef.empty and (df["Ticker"].astype(str).str.upper() == ticker).any():
            st.error(f"Ticker {ticker} finns redan. Inga ändringar sparade.")
            return None

        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV (SEK)": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            # initiera rad med rimliga defaults
            tom = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad",
                                        "Senast auto-uppdaterad", "Senast uppdaterad källa"]
                       and not str(c).startswith("TS_") else "")
                   for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # stämpla manuell uppdatering för ändrade kärnfält
        df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        for f in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år"]:
            if f in TS_FIELDS and f in ny:
                df.loc[df["Ticker"] == ticker, TS_FIELDS[f]] = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Räkna om och returnera uppdaterad df (app.py sparar)
        df = ensure_schema(df)
        df = konvertera_typer(df)
        df = uppdatera_berakningar(df, user_rates)
        st.success("Sparat.")
        return df

    # Enskild uppdatering för valt bolag (pris / full) – om sources finns
    if not bef.empty and _HAS_SOURCES:
        st.markdown("### ⚙️ Enskild uppdatering")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📈 Uppdatera kurs (snabb)"):
                try:
                    vals, debug = run_update_price_only(bef["Ticker"])
                    st.success(f"{bef['Ticker']}: Kurs uppdaterad.")
                    st.json(debug)
                except Exception as e:
                    st.error(f"{bef['Ticker']}: Fel: {e}")
        with c2:
            if st.button("🔄 Full auto-uppdatering (alla fält)"):
                try:
                    vals, debug, source = run_update_full(bef["Ticker"])
                    st.success(f"{bef['Ticker']}: Full uppdatering körd (källa: {source}).")
                    st.json(debug)
                except Exception as e:
                    st.error(f"{bef['Ticker']}: Fel: {e}")

    st.divider()

    # Manuell prognoslista – nu i denna vy (som du önskade)
    st.markdown("### 📝 Manuell prognoslista (Omsättning idag/nästa år)")
    mp = _manuell_prognos_df(df)
    if mp.empty:
        st.success("Inga uppenbara bolag kräver manuell prognosuppdatering just nu.")
    else:
        st.warning(f"{len(mp)} bolag kan behöva manuell uppdatering:")
        st.dataframe(mp, use_container_width=True, hide_index=True)

    return None  # inget att spara


# =========================
# Vy: Investeringsförslag
# =========================
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )
    subset = st.radio("Vilka bolag?", ["Alla bolag", "Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential", "Närmast riktkurs"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential & absdiff
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    # Implied market cap (nu) + formatering
    base["_implied_mcap"] = base.apply(_implied_mcap, axis=1)

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Navigering
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base) - 1)

    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index + 1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base) - 1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljvärde för andel-beräkning
    port = df[df["Antal aktier"] > 0].copy()
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: _fx(user_rates, v))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        total_värde = float(port["Värde (SEK)"].sum())
    else:
        total_värde = 0.0

    vx = _fx(user_rates, rad["Valuta"])
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / total_värde) * 100.0, 2) if total_värde > 0 else 0.0
    ny_andel = round((ny_total / total_värde) * 100.0, 2) if total_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    mcap_str = _fmt_money(rad["_implied_mcap"], rad["Valuta"])
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}",
        f"- **Utestående aktier:** {round(float(rad.get('Utestående aktier',0.0)), 2)} M",
        f"- **Nuvarande marketcap:** {mcap_str}",
        f"- **P/S (nu):** {round(float(rad.get('P/S', 0.0)), 2)}",
        f"- **P/S-snitt (Q1–Q4):** {round(float(rad.get('P/S-snitt', 0.0)), 2)}",
        f"- **Riktkurs (vald):** {round(float(rad[riktkurs_val]), 2)} {rad['Valuta']}",
        f"- **Uppsida (valda riktkursen):** {round(float(rad['Potential (%)']), 2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ]
    st.markdown("\n".join(lines))

    with st.expander("Detaljerad info (PS Q1–Q4, riktkurser mm.)", expanded=False):
        cols = [
            "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
            "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        ]
        cols = [c for c in cols if c in base.columns]
        st.dataframe(pd.DataFrame([rad[cols].to_dict()]), use_container_width=True, hide_index=True)


# =========================
# Vy: Portfölj
# =========================
def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: _fx(user_rates, v))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0), 2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde, 2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd, 2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd / 12.0, 2)} SEK")

    show_cols = ["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)
