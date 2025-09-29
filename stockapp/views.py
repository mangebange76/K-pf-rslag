# stockapp/views.py
# -*- coding: utf-8 -*-
"""
Streamlit-vyer:
- kontrollvy(df)
- analysvy(df, user_rates)
- lagg_till_eller_uppdatera(df, user_rates)
- visa_investeringsforslag(df, user_rates)
- visa_portfolj(df, user_rates)

Designad för att vara robust även om vissa kolumner saknas.
För att skriva data används stockapp.storage.spara_data (har wipe-skydd).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st

from .rates import hamta_valutakurs
from .storage import spara_data
from .utils import now_stamp, ensure_schema


# ---------------------------------------------------------------------
# Hjälpare (lokala)
# ---------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def _get_num(row, name, default: float = 0.0) -> float:
    try:
        v = row.get(name, default)
        v = float(v) if v is not None and v != "" else default
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _fmt_money(x: float) -> str:
    # Enkel formattering i tusental/miljoner/miljarder/”triljoner”
    try:
        x = float(x)
    except Exception:
        return "-"
    sign = "-" if x < 0 else ""
    v = abs(x)
    if v >= 1_000_000_000_000:
        return f"{sign}{v/1_000_000_000_000:.2f} T"
    if v >= 1_000_000_000:
        return f"{sign}{v/1_000_000_000:.2f} B"
    if v >= 1_000_000:
        return f"{sign}{v/1_000_000:.2f} M"
    if v >= 1_000:
        return f"{sign}{v/1_000:.2f} k"
    return f"{sign}{v:.2f}"

def _calc_mcap_native(row: pd.Series) -> float:
    """Market cap i bolagets noteringsvaluta (grovt): aktier (milj) * 1e6 * kurs."""
    shares_m = _get_num(row, "Utestående aktier", 0.0)
    px = _get_num(row, "Aktuell kurs", 0.0)
    if shares_m > 0 and px > 0:
        return shares_m * 1_000_000.0 * px
    return 0.0

def _div_yield(row: pd.Series) -> float:
    div = _get_num(row, "Årlig utdelning", 0.0)
    px = _get_num(row, "Aktuell kurs", 0.0)
    if div > 0 and px > 0:
        return 100.0 * (div / px)
    return 0.0

def _potential_pct(row: pd.Series, target_col: str = "Riktkurs om 1 år") -> float:
    px = _get_num(row, "Aktuell kurs", 0.0)
    tgt = _get_num(row, target_col, 0.0)
    if px > 0 and tgt > 0:
        return 100.0 * (tgt - px) / px
    return 0.0

def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """Minsta datum bland alla TS_-kolumner, annars None."""
    ds: List[pd.Timestamp] = []
    for c in row.index:
        if str(c).startswith("TS_"):
            s = str(row.get(c, "")).strip()
            if s:
                try:
                    d = pd.to_datetime(s, errors="coerce")
                    if pd.notna(d):
                        ds.append(d)
                except Exception:
                    pass
    if not ds:
        return None
    return min(ds)

def _ensure_num_col(df: pd.DataFrame, name: str):
    if name not in df.columns:
        df[name] = 0.0
    df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------
# 1) Kontrollvy
# ---------------------------------------------------------------------
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    if df is None or df.empty:
        st.info("Inga bolag inlästa ännu.")
        return

    # Äldst uppdaterade (alla spårade TS_-fält)
    st.subheader("⏱️ Äldst uppdaterade fält")
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    # Sortera: äldst först (None sist)
    work["_oldest_any_ts_sort"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    vis_cols = [c for c in ["Ticker", "Bolagsnamn"] if c in work.columns]
    ts_cols = [c for c in work.columns if str(c).startswith("TS_")]
    vis_cols = vis_cols + ts_cols + ["_oldest_any_ts"]
    st.dataframe(
        work.sort_values(by=["_oldest_any_ts_sort"])[vis_cols].head(20),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("📋 Databas (förhandsgranskning)")
    base_cols = [c for c in ["Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier", "P/S", "P/S-snitt"] if c in df.columns]
    st.dataframe(df[base_cols].sort_values(by=["Bolagsnamn", "Ticker"]).head(50), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# 2) Analysvy
# ---------------------------------------------------------------------
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    if df is None or df.empty:
        st.info("Inga bolag i databasen.")
        return

    work = df.copy()
    work = work.sort_values(by=[c for c in ["Bolagsnamn", "Ticker"] if c in work.columns])
    labels = [f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})" for _, r in work.iterrows()]

    idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(work)-1), value=0, step=1)
    st.selectbox("Eller välj i lista", labels, index=idx if labels else 0, key="analys_select")

    st.write(f"Post {idx+1}/{len(work)}")

    row = work.iloc[idx]
    st.subheader(f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})")

    show_cols = [c for c in [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"
    ] if c in work.columns]

    st.dataframe(pd.DataFrame([row[show_cols].to_dict()]), use_container_width=True, hide_index=True)

    # Enkel nyckeltals-badge
    mcap = _calc_mcap_native(row)
    divy = _div_yield(row)
    pot  = _potential_pct(row, "Riktkurs om 1 år")
    st.caption(f"⛳ Market cap (native): **{_fmt_money(mcap)}** | Utdelningsyield: **{divy:.2f}%** | Potensial vs 1-årsriktkurs: **{pot:.1f}%**")


# ---------------------------------------------------------------------
# 3) Lägg till / uppdatera bolag
# ---------------------------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    work = df.copy()
    work = work.sort_values(by=[c for c in ["Bolagsnamn", "Ticker"] if c in work.columns])

    namn_map = {f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})": r.get("Ticker","") for _, r in work.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    sel = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=0)

    if sel and sel in namn_map and namn_map[sel]:
        bef = work[work["Ticker"] == namn_map[sel]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=str(bef.get("Ticker","")).upper())
            utest = st.number_input("Utestående aktier (miljoner)", value=float(_get_num(bef, "Utestående aktier", 0.0)), step=0.01)
            antal = st.number_input("Antal aktier du äger", value=float(_get_num(bef, "Antal aktier", 0.0)), step=1.0)
            valuta = st.text_input("Valuta", value=str(bef.get("Valuta","")).upper())
            kurs = st.number_input("Aktuell kurs", value=float(_get_num(bef, "Aktuell kurs", 0.0)), step=0.01)

        with c2:
            ps   = st.number_input("P/S", value=float(_get_num(bef, "P/S", 0.0)), step=0.01)
            ps1  = st.number_input("P/S Q1", value=float(_get_num(bef, "P/S Q1", 0.0)), step=0.01)
            ps2  = st.number_input("P/S Q2", value=float(_get_num(bef, "P/S Q2", 0.0)), step=0.01)
            ps3  = st.number_input("P/S Q3", value=float(_get_num(bef, "P/S Q3", 0.0)), step=0.01)
            ps4  = st.number_input("P/S Q4", value=float(_get_num(bef, "P/S Q4", 0.0)), step=0.01)

        c3, c4 = st.columns(2)
        with c3:
            oms_idag = st.number_input("Omsättning i år (miljoner, MANUELL)", value=float(_get_num(bef, "Omsättning idag", 0.0)), step=1.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner, MANUELL)", value=float(_get_num(bef, "Omsättning nästa år", 0.0)), step=1.0)
        with c4:
            utd = st.number_input("Årlig utdelning (per aktie)", value=float(_get_num(bef, "Årlig utdelning", 0.0)), step=0.01)
            cagr = st.number_input("CAGR 5 år (%)", value=float(_get_num(bef, "CAGR 5 år (%)", 0.0)), step=0.1)

        spar = st.form_submit_button("💾 Spara")

    if spar:
        if not ticker:
            st.error("Ticker krävs.")
            return df

        # Uppdatera/infoga
        new_vals = {
            "Ticker": ticker.upper(),
            "Utestående aktier": float(utest),
            "Antal aktier": float(antal),
            "Valuta": str(valuta).upper() if valuta else "",
            "Aktuell kurs": float(kurs),
            "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
            "Omsättning idag": float(oms_idag),
            "Omsättning nästa år": float(oms_next),
            "Årlig utdelning": float(utd),
            "CAGR 5 år (%)": float(cagr),
        }

        out = df.copy()
        if "Ticker" in out.columns and (out["Ticker"] == ticker.upper()).any():
            ridx = out.index[out["Ticker"] == ticker.upper()][0]
            for k, v in new_vals.items():
                if k in out.columns:
                    out.at[ridx, k] = v
                else:
                    out[k] = out.get(k, 0.0)
                    out.at[ridx, k] = v
            # Stämpel
            if "Senast manuellt uppdaterad" in out.columns:
                out.at[ridx, "Senast manuellt uppdaterad"] = now_stamp()
        else:
            # Lägg till ny rad (säkerställ schema först)
            out = ensure_schema(out)
            blank = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in out.columns}
            blank.update(new_vals)
            out = pd.concat([out, pd.DataFrame([blank])], ignore_index=True)
            ridx = out.index[out["Ticker"] == ticker.upper()][0]
            if "Senast manuellt uppdaterad" in out.columns:
                out.at[ridx, "Senast manuellt uppdaterad"] = now_stamp()

        out = ensure_schema(out)
        spara_data(out)
        st.success("Sparat.")
        return out

    # ”Manuell prognoslista” – visa de med äldst TS i omsättningsfälten
    st.divider()
    st.subheader("📝 Manuell prognoslista (äldsta 'Omsättning i år' / 'nästa år')")
    work = df.copy()
    ts_cols = [c for c in work.columns if c in ("TS_Omsättning idag", "TS_Omsättning nästa år")]
    if ts_cols:
        work["_oldest_prog_ts"] = work[ts_cols].apply(
            lambda r: pd.to_datetime(
                min([x for x in r.values if str(x).strip()], default=None), errors="coerce"
            ),
            axis=1
        )
        work["_oldest_prog_ts_sort"] = work["_oldest_prog_ts"].fillna(pd.Timestamp("2099-12-31"))
        show_cols = [c for c in ["Ticker","Bolagsnamn","Omsättning idag","Omsättning nästa år"] if c in work.columns] + ts_cols + ["_oldest_prog_ts"]
        st.dataframe(
            work.sort_values(by=["_oldest_prog_ts_sort"])[show_cols].head(20),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Inga TS-kolumner för omsättning hittades.")


# ---------------------------------------------------------------------
# 4) Investeringsförslag (enkel, robust)
# ---------------------------------------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen.")
        return

    df = df.copy()
    _ensure_num_col(df, "Aktuell kurs")
    _ensure_num_col(df, "P/S")
    for c in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        _ensure_num_col(df, c)

    # P/S-snitt om saknas
    if "P/S-snitt" not in df.columns:
        df["P/S-snitt"] = 0.0
    def _ps_mean_row(r):
        vals = [ _get_num(r, c, 0.0) for c in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] ]
        vals = [v for v in vals if v > 0]
        return round(float(np.mean(vals)), 2) if vals else _get_num(r, "P/S", 0.0)
    df["P/S-snitt"] = df.apply(_ps_mean_row, axis=1)

    mode = st.radio("Fokus", ["Tillväxt","Utdelning","Alla"], horizontal=True)

    base = df.copy()
    # Enkel prior: Tillväxt → sortera på Potential (om riktkurs finns) annars lägst P/S-snitt
    base["Potential (%)"] = base.apply(lambda r: _potential_pct(r, "Riktkurs om 1 år"), axis=1)
    base["Dividend yield (%)"] = base.apply(_div_yield, axis=1)
    base["Market cap (native)"] = base.apply(_calc_mcap_native, axis=1)

    if mode == "Tillväxt":
        if (base["Potential (%)"] != 0).any():
            base = base.sort_values(by=["Potential (%)"], ascending=False)
        else:
            base = base[base["P/S-snitt"] > 0].sort_values(by=["P/S-snitt"], ascending=True)
    elif mode == "Utdelning":
        base = base.sort_values(by=["Dividend yield (%)"], ascending=False)
    else:
        base = base.sort_values(by=["Bolagsnamn","Ticker"])

    # Filtrera bort rader utan pris
    base = base[base["Aktuell kurs"] > 0].copy()
    if base.empty:
        st.info("Inga kandidater med tillräcklig data.")
        return

    # Paginering/browse
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = max(0, min(st.session_state.forslags_index, len(base)-1))

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    r = base.iloc[st.session_state.forslags_index]

    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")
    lines = []
    lines.append(f"- **Aktuell kurs:** { _get_num(r, 'Aktuell kurs', 0.0):.2f} {r.get('Valuta','')}")
    if "Riktkurs idag" in r.index:
        lines.append(f"- **Riktkurs idag:** { _get_num(r, 'Riktkurs idag', 0.0):.2f} {r.get('Valuta','')}")
    if "Riktkurs om 1 år" in r.index:
        lines.append(f"- **Riktkurs om 1 år:** { _get_num(r, 'Riktkurs om 1 år', 0.0):.2f} {r.get('Valuta','')}")
    lines.append(f"- **P/S-snitt (Q1–Q4):** { _get_num(r, 'P/S-snitt', 0.0):.2f}")
    lines.append(f"- **P/S (nu):** { _get_num(r, 'P/S', 0.0):.2f}")
    lines.append(f"- **Utdelningsyield:** { _div_yield(r):.2f}%")
    lines.append(f"- **Market cap (native):** {_fmt_money(_calc_mcap_native(r))}")
    lines.append(f"- **Potential vs 1-årsriktkurs:** { _potential_pct(r, 'Riktkurs om 1 år'):.1f}%")
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------------
# 5) Portföljvy
# ---------------------------------------------------------------------
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    if df is None or df.empty or "Antal aktier" not in df.columns:
        st.info("Inga innehav registrerade.")
        return

    port = df.copy()
    _ensure_num_col(port, "Antal aktier")
    _ensure_num_col(port, "Aktuell kurs")

    port = port[port["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Valuta → SEK
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates) if isinstance(v, str) and v else 1.0)
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    if total_värde <= 0:
        st.info("Kunde inte beräkna portföljvärde (saknar kurser/valuta).")
        return

    port["Andel (%)"] = np.where(total_värde > 0, (port["Värde (SEK)"] / total_värde) * 100.0, 0.0)
    _ensure_num_col(port, "Årlig utdelning")
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = [c for c in [
        "Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
        "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"
    ] if c in port.columns]

    st.dataframe(
        port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True, hide_index=True
    )
