# stockapp/views/control.py
import streamlit as st
import pandas as pd

def _oldest_any_ts(row: pd.Series):
    dates = []
    for c, v in row.items():
        if str(c).startswith("TS_") and str(v).strip():
            try:
                d = pd.to_datetime(str(v).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else pd.NaT

def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_oldest_any_ts"] = out.apply(_oldest_any_ts, axis=1)
    out["_oldest_any_ts_fill"] = out["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return out

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Flagga bolag där något av nyckelfälten saknas eller äldsta TS är äldre än X dagar.
    """
    need_cols = [c for c in [
        "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år"
    ] if c in df.columns]

    ts_cols = [c for c in df.columns if str(c).startswith("TS_")]
    out_rows = []
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(older_than_days))

    for _, r in df.iterrows():
        missing_val = any(float(r.get(c, 0.0) or 0.0) <= 0.0 for c in need_cols)
        missing_ts  = any(not str(r.get(ts, "")).strip() for ts in ts_cols)
        oldest = _oldest_any_ts(r)
        oldest_dt = pd.to_datetime(oldest, errors="coerce")
        too_old = (pd.notna(oldest_dt) and oldest_dt < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest.strftime("%Y-%m-%d") if pd.notna(oldest) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    return pd.DataFrame(out_rows)

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")
    if df is None or df.empty:
        st.info("Inga bolag än.")
        return

    # 1) Äldst uppdaterade
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = _add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
    cols_show = [c for c in ["Ticker","Bolagsnamn"] if c in vis.columns]
    cols_show += [c for c in vis.columns if str(c).startswith("TS_")]
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Kräver manuell hantering?
    st.subheader("🛠️ Kräver manuell hantering")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Körlogg om någon satt något i session_state
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen körlogg i denna session.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar**")
            st.json(log.get("changed", {}))
        with col2:
            st.markdown("**Missar**")
            st.json(log.get("misses", {}))
