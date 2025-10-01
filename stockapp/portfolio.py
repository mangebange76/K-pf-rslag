# -*- coding: utf-8 -*-
"""
stockapp.portfolio
------------------
Visar portföljen: värden i SEK, andelar, utdelningar, ev. GAV/avkastning,
sektor-fördelning och enkla koncentrationsvarningar.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from .rates import hamta_valutakurs
from .utils import format_large_number, coerce_float


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return isinstance(df, pd.DataFrame) and col in df.columns


def _pick_sector_col(df: pd.DataFrame) -> str:
    """Stöd både 'Sektor' och 'Sector'."""
    if _col_exists(df, "Sektor"):
        return "Sektor"
    if _col_exists(df, "Sector"):
        return "Sector"
    return ""


def _safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if not _col_exists(df, col):
        return pd.Series([0.0] * len(df), index=df.index)
    return df[col].apply(coerce_float).fillna(0.0)


def _render_sector_breakdown(port: pd.DataFrame) -> None:
    sec_col = _pick_sector_col(port)
    if not sec_col:
        st.info("Ingen sektorkolumn hittades (lägger du till 'Sektor' i databasen visas fördelningen här).")
        return

    grp = port.groupby(sec_col, dropna=False)["Värde (SEK)"].sum().sort_values(ascending=False)
    if grp.empty:
        st.info("Ingen data för sektorfördelning ännu.")
        return

    tot = float(grp.sum())
    pct = (grp / max(tot, 1e-9) * 100.0).round(2)
    show = pd.DataFrame({"Värde (SEK)": grp.round(2), "Andel (%)": pct})
    st.subheader("Sektorfördelning")
    st.dataframe(show, use_container_width=True)
    try:
        st.bar_chart(grp)
    except Exception:
        pass

    # Varningar för övervikt
    heavy_sectors = pct[pct >= 30.0]
    if not heavy_sectors.empty:
        lines = [f"- {name}: {val:.1f} %" for name, val in heavy_sectors.items()]
        st.warning("Du har hög koncentration i följande sektorer (≥30%):\n" + "\n".join(lines))


def _render_concentration_warnings(port: pd.DataFrame, total_sek: float) -> None:
    """
    Enkla koncentrationsvarningar per position.
    """
    if total_sek <= 0 or port.empty:
        return
    heavy_pos = port[port["Andel (%)"] >= 15.0]
    if not heavy_pos.empty:
        items = [f"- {r['Bolagsnamn']} ({r['Ticker']}): {r['Andel (%)']:.1f} %" for _, r in heavy_pos.iterrows()]
        st.warning("Hög koncentration i enskilda innehav (≥15%):\n" + "\n".join(items))


def _render_sell_guard(port: pd.DataFrame) -> None:
    """
    Enkel säljvakt (beta). Ger signaler om:
      - Kurs ligger ≥20 % över vald riktkurs (om riktkurskolumner finns)
      - 'Valuation Label' indikerar stark övervärdering (om kolumnen finns)
    Tar också med GAV i SEK om det finns, för att ge ett fingerat förslag.
    """
    cols_rikt = [c for c in port.columns if c.startswith("Riktkurs")]
    has_label = _col_exists(port, "Valuation Label")
    has_gav = _col_exists(port, "GAV (SEK)")

    if not cols_rikt and not has_label:
        with st.expander("Säljvakt (beta)"):
            st.info("Lägg till riktkurs-kolumner (t.ex. 'Riktkurs om 1 år') eller 'Valuation Label' för att aktivera säljvakt.")
        return

    out_rows: List[dict] = []
    for _, r in port.iterrows():
        cur = coerce_float(r.get("Aktuell kurs"))
        if not cur or cur <= 0:
            continue

        # välj en riktningskolumn (prioritera 1 år)
        rk = None
        for pref in ["Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Riktkurs idag"]:
            if _col_exists(port, pref):
                tmp = coerce_float(r.get(pref))
                if tmp and tmp > 0:
                    rk = tmp
                    break
        # label
        label = r.get("Valuation Label") if has_label else ""

        signal = ""
        if rk and rk > 0:
            # Om pris > 1.2 * riktkurs => ev. trim/sälj-signal
            if cur >= 1.2 * rk:
                signal = "Över riktkurs +20% → överväg trim/sälj"
        if not signal and label:
            if str(label).lower() in ("overvalued", "dyr", "sälj", "trimma/sälj"):
                signal = f"Värdering: {label}"

        if signal:
            row = {
                "Ticker": r.get("Ticker", ""),
                "Bolagsnamn": r.get("Bolagsnamn", ""),
                "Aktuell kurs": cur,
                "Valuta": r.get("Valuta", ""),
                "Signal": signal,
            }
            if has_gav:
                row["GAV (SEK)"] = coerce_float(r.get("GAV (SEK)"))
            out_rows.append(row)

    with st.expander("Säljvakt (beta)"):
        if not out_rows:
            st.success("Inga starka sälj/trim-signaler just nu baserat på enkla regler.")
        else:
            df_s = pd.DataFrame(out_rows)
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            st.caption(
                "Regler i beta: pris ≥ 120% av närmast relevanta riktkurs, eller "
                "Valuation Label indikerar klar övervärdering."
            )


def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    """
    Huvudvy för portföljen.
    Räknar värde i SEK, andelar, utdelningar, P/L om GAV finns,
    samt sektorfördelning och koncentrationsvarningar.
    """
    st.header("📦 Min portfölj")

    if df is None or df.empty or "Antal aktier" not in df.columns:
        st.info("Inga bolag i databasen ännu eller kolumnen 'Antal aktier' saknas.")
        return

    # Fokusera bara på innehav > 0
    port = df.copy()
    port["Antal aktier"] = _safe_float_series(port, "Antal aktier")
    port = port[port["Antal aktier"] > 0].copy()

    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Säkerställ numerik
    port["Aktuell kurs"] = _safe_float_series(port, "Aktuell kurs")
    port["Årlig utdelning"] = _safe_float_series(port, "Årlig utdelning")

    # Växelkurs + värden i SEK
    port["Växelkurs"] = port.get("Valuta", pd.Series(["SEK"] * len(port))).apply(
        lambda v: hamta_valutakurs(v, user_rates)
    )
    port["Värde (SEK)"] = (port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]).astype(float)

    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)

    # Utdelningar i SEK
    port["Total årlig utdelning (SEK)"] = (port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]).astype(float)
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    # GAV (SEK) om finns
    if _col_exists(port, "GAV (SEK)"):
        port["GAV (SEK)"] = _safe_float_series(port, "GAV (SEK)")
        port["Anskaffningsvärde (SEK)"] = (port["GAV (SEK)"] * port["Antal aktier"]).astype(float)
        port["Orealiserad P/L (SEK)"] = (port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]).astype(float)
        port["P/L (%)"] = np.where(
            port["Anskaffningsvärde (SEK)"] > 0,
            (port["Orealiserad P/L (SEK)"] / port["Anskaffningsvärde (SEK)"]) * 100.0,
            0.0,
        ).round(2)
        tot_cost = float(port["Anskaffningsvärde (SEK)"].sum())
        tot_pl = float(port["Orealiserad P/L (SEK)"].sum())
        st.markdown(
            f"**Totalt portföljvärde:** {format_large_number(total_värde, 'SEK')}  \n"
            f"**Anskaffningsvärde:** {format_large_number(tot_cost, 'SEK')}  \n"
            f"**Orealiserad P/L:** {format_large_number(tot_pl, 'SEK')}  \n"
            f"**Total kommande utdelning/år:** {format_large_number(tot_utd, 'SEK')}  \n"
            f"**≈ Månadsutdelning:** {format_large_number(tot_utd/12.0, 'SEK')}"
        )
    else:
        st.markdown(
            f"**Totalt portföljvärde:** {format_large_number(total_värde, 'SEK')}  \n"
            f"**Total kommande utdelning/år:** {format_large_number(tot_utd, 'SEK')}  \n"
            f"**≈ Månadsutdelning:** {format_large_number(tot_utd/12.0, 'SEK')}"
        )
        st.caption("Tips: lägg till kolumnen **'GAV (SEK)'** för att se P/L per innehav och totalt.")

    # Visa tabell
    show_cols = ["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Växelkurs", "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning (SEK)"]
    if _col_exists(port, "GAV (SEK)"):
        show_cols += ["GAV (SEK)", "Anskaffningsvärde (SEK)", "Orealiserad P/L (SEK)", "P/L (%)"]

    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(
        port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Sektorfördelning & koncentration
    _render_sector_breakdown(port)
    _render_concentration_warnings(port, total_värde)

    # Enkel säljvakt (beta)
    _render_sell_guard(port)
