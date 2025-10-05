from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

# Egna moduler
from stockapp.sheets import ws_read_df, ws_write_df, list_worksheet_titles, delete_worksheet
from stockapp.schema import enforce_schema, CANON_COLS
from stockapp.rates import read_rates, save_rates, fetch_live_rates, DEFAULT_RATES
from stockapp.dividends import build_dividend_calendar

# Fetchers
from stockapp.fetchers.yahoo import get_quick as yf_quick, get_all as yf_all
from stockapp.fetchers.finviz import get_overview as fz_overview
from stockapp.fetchers.morningstar import get_overview as ms_overview
from stockapp.fetchers.sec import get_pb_quarters as sec_pb_quarters

st.set_page_config(page_title="K-pf-rslag", layout="wide")

# ---------- tid ----------
def _now_sthlm() -> datetime:
    try:
        import pytz
        return datetime.now(pytz.timezone("Europe/Stockholm"))
    except Exception:
        return datetime.now()

def now_stamp() -> str:
    return _now_sthlm().strftime("%Y-%m-%d")

# ---------- snapshot ----------
SNAP_PREFIX = "SNAP__"
def _fmt(dt: datetime) -> str: return dt.strftime("%Y%m%d_%H%M%S")
def _parse_ts(s: str) -> Optional[datetime]:
    try: return datetime.strptime(s[-15:], "%Y%m%d_%H%M%S")
    except: return None

def snapshot_on_start(df: pd.DataFrame, base_ws_title: str):
    if st.session_state.get("_snap_done"): return
    st.session_state["_snap_done"] = True
    try:
        # grov cellkontroll – skapa snapshot endast om <= 50k rader * 70 kolumner
        rows, cols = (len(df) or 1), len(CANON_COLS)
        if rows*cols > 3_000_000:  # säker spärr för stora ark
            st.sidebar.warning("Snapshot hoppades över (för stort blad).")
            return
        title = f"{SNAP_PREFIX}{base_ws_title}__{_fmt(_now_sthlm())}"
        ws_write_df(title, df)
        # städa äldre än 5 dagar eller fler än 5 snapshots
        all_titles = list_worksheet_titles()
        snaps = [t for t in all_titles if t.startswith(SNAP_PREFIX)]
        snaps_sorted = sorted([(t,_parse_ts(t)) for t in snaps], key=lambda x: x[1] or datetime.min, reverse=True)
        for i,(t,ts) in enumerate(snaps_sorted):
            if i>=5 or (ts and ts < (_now_sthlm()-timedelta(days=5))):
                try: delete_worksheet(t)
                except: pass
    except Exception as e:
        st.sidebar.warning(f"Kunde inte spara snapshot: {e}")

# ---------- IO ----------
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str, _nonce: int) -> pd.DataFrame:
    return ws_read_df(ws_title)

def load_df(ws_title: str) -> pd.DataFrame:
    n = st.session_state.get("_reload_nonce", 0)
    raw = load_df_cached(ws_title, n)
    return enforce_schema(raw, ws_title, write_back=True)

def save_df(ws_title: str, df: pd.DataFrame):
    ws_write_df(ws_title, df)
    st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1

# ---------- utils ----------
def _to_float(x) -> float:
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"nan","none"}: return 0.0
        s = s.replace("\u00a0"," ").replace("%","").replace(",","_").replace(".","").replace("_",".")
        return float(s)
    except Exception:
        try: return float(x)
        except: return 0.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ---------- sidopanel: valutakurser ----------
def sidebar_rates() -> Dict[str,float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")
    if "rates_loaded" not in st.session_state:
        saved = read_rates()
        for k in ["USD","NOK","CAD","EUR"]:
            st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
        st.session_state["rates_loaded"] = True

    c1,c2 = st.sidebar.columns(2)
    if c1.button("🌐 Hämta live"):
        try:
            live = fetch_live_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(live[k])
            st.sidebar.success("Livekurser hämtade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")
    if c2.button("↻ Läs sparade"):
        try:
            saved = read_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
            st.sidebar.success("Sparade kurser inlästa.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte läsa sparade kurser: {e}")

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.000001, format="%.6f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.000001, format="%.6f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.000001, format="%.6f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.000001, format="%.6f")

    c3,c4 = st.sidebar.columns(2)
    if c3.button("💾 Spara"):
        try:
            save_rates({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.sidebar.success("Kurser sparade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")
    if c4.button("🔁 Rensa cache"):
        st.cache_data.clear()
        st.sidebar.info("Cache rensad.")

    return {"USD":usd,"NOK":nok,"CAD":cad,"EUR":eur,"SEK":1.0}

# ---------- beräkningar ----------
def compute_ps_pb_avg(row: pd.Series) -> Tuple[float,float]:
    ps = [row.get("P/S Q1",0),row.get("P/S Q2",0),row.get("P/S Q3",0),row.get("P/S Q4",0)]
    ps = [float(x) for x in ps if _to_float(x)>0]
    pb = [row.get("P/B Q1",0),row.get("P/B Q2",0),row.get("P/B Q3",0),row.get("P/B Q4",0)]
    pb = [float(x) for x in pb if _to_float(x)>0]
    return (round(float(np.mean(ps)),2) if ps else 0.0,
            round(float(np.mean(pb)),2) if pb else 0.0)

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i,r in out.iterrows():
        ps_avg, pb_avg = compute_ps_pb_avg(r)
        out.at[i,"P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i,"P/B-snitt (Q1..Q4)"] = pb_avg

        cagr = _to_float(r.get("CAGR 5 år (%)",0.0))
        g = clamp(cagr, 2.0, 50.0)/100.0

        next_rev = _to_float(r.get("Omsättning nästa år",0.0))
        if next_rev>0:
            out.at[i,"Omsättning om 2 år"] = round(next_rev*(1.0+g),2)
            out.at[i,"Omsättning om 3 år"] = round(next_rev*((1.0+g)**2),2)

        shares_m = _to_float(r.get("Utestående aktier",0.0))  # antas miljoner
        if shares_m>0 and ps_avg>0:
            out.at[i,"Riktkurs idag"]    = round(_to_float(r.get("Omsättning idag",0.0))*ps_avg/shares_m, 2)
            out.at[i,"Riktkurs om 1 år"] = round(_to_float(r.get("Omsättning nästa år",0.0))*ps_avg/shares_m, 2)
            out.at[i,"Riktkurs om 2 år"] = round(_to_float(out.at[i,"Omsättning om 2 år"])*ps_avg/shares_m, 2)
            out.at[i,"Riktkurs om 3 år"] = round(_to_float(out.at[i,"Omsättning om 3 år"])*ps_avg/shares_m, 2)
        else:
            out.at[i,"Riktkurs idag"]=out.at[i,"Riktkurs om 1 år"]=out.at[i,"Riktkurs om 2 år"]=out.at[i,"Riktkurs om 3 år"]=0.0
    # Uppsida + DA
    price = out["Aktuell kurs"].map(_to_float).replace(0,np.nan)
    for col, tgt in [("Riktkurs idag","Uppsida idag (%)"),
                     ("Riktkurs om 1 år","Uppsida 1 år (%)"),
                     ("Riktkurs om 2 år","Uppsida 2 år (%)"),
                     ("Riktkurs om 3 år","Uppsida 3 år (%)")]:
        rk = out[col].map(_to_float).replace(0,np.nan)
        out[tgt] = ((rk - price)/price*100.0).fillna(0.0)
    out["DA (%)"] = np.where(out["Aktuell kurs"].map(_to_float)>0,
                             (out["Årlig utdelning"].map(_to_float)/out["Aktuell kurs"].map(_to_float))*100.0, 0.0)
    out["Senast beräknad"] = now_stamp()
    return out

def _horizon_tag(h: str) -> str:
    if "om 1 år" in h: return "1 år"
    if "om 2 år" in h: return "2 år"
    if "om 3 år" in h: return "3 år"
    return "Idag"

def score_rows(df: pd.DataFrame, horizon: str, strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["Uppsida (%)"] = np.where(out["Aktuell kurs"].map(_to_float)>0,
                                  (out[horizon].map(_to_float)-out["Aktuell kurs"].map(_to_float))/out["Aktuell kurs"].map(_to_float)*100.0,0.0)
    cur_ps = out["P/S"].map(_to_float).replace(0,np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].map(_to_float).replace(0,np.nan)
    cheap_ps = (ps_avg/(cur_ps*2.0)).clip(upper=1.0).fillna(0.0)

    g_norm = (out["CAGR 5 år (%)"].map(_to_float)/30.0).clip(0,1)
    u_norm = (out["Uppsida (%)"]/50.0).clip(0,1)
    out["Score (Growth)"] = (0.4*g_norm + 0.4*u_norm + 0.2*cheap_ps)*100.0

    payout = out["Payout (%)"].map(_to_float)
    payout_health = 1 - (abs(payout - 60.0)/60.0)
    payout_health = payout_health.clip(0,1)
    payout_health = np.where(payout<=0, 0.85, payout_health)
    y_norm = (out["DA (%)"]/8.0).clip(0,1)
    grow_ok = np.where(out["CAGR 5 år (%)"].map(_to_float)>=0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6*y_norm + 0.3*payout_health + 0.1*grow_ok)*100.0

    cur_pb = out["P/B"].map(_to_float).replace(0,np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].map(_to_float).replace(0,np.nan)
    cheap_pb = (pb_avg/(cur_pb*2.0)).clip(upper=1.0).fillna(0.0)
    out["Score (Financials)"] = (0.7*cheap_pb + 0.3*u_norm)*100.0

    def wts(sektor: str, strat: str):
        if strat=="Tillväxt": return (0.70,0.10,0.20)
        if strat=="Utdelning":return (0.15,0.70,0.15)
        if strat=="Finans":   return (0.20,0.20,0.60)
        s=(sektor or "").lower()
        if any(k in s for k in ["bank","finans","insurance","financial"]): return (0.25,0.25,0.50)
        if any(k in s for k in ["utility","utilities","consumer staples","telecom"]): return (0.20,0.60,0.20)
        if any(k in s for k in ["tech","information technology","semiconductor","software"]): return (0.70,0.10,0.20)
        return (0.45,0.35,0.20)

    Wg, Wd, Wf = [], [], []
    for _,r in out.iterrows():
        a,b,c = wts(str(r.get("Sektor","")), strategy)
        Wg.append(a); Wd.append(b); Wf.append(c)
    out["Score (Total)"] = (np.array(Wg)*out["Score (Growth)"] + np.array(Wd)*out["Score (Dividend)"] + np.array(Wf)*out["Score (Financials)"]).round(2)

    need = [out["Aktuell kurs"].map(_to_float)>0,
            out["P/S-snitt (Q1..Q4)"].map(_to_float)>0,
            out["Omsättning idag"].map(_to_float)>=0,
            out["Omsättning nästa år"].map(_to_float)>=0]
    out["Confidence"] = (np.stack(need,axis=0).astype(float).mean(axis=0)*100.0).round(0)
    return out

def compute_scores_all(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    for col, tag in [("Riktkurs idag","Idag"),("Riktkurs om 1 år","1 år"),("Riktkurs om 2 år","2 år"),("Riktkurs om 3 år","3 år")]:
        tmp = score_rows(out, horizon=col, strategy=strategy)
        out[f"Score Growth ({tag})"]=tmp["Score (Growth)"].round(2)
        out[f"Score Dividend ({tag})"]=tmp["Score (Dividend)"].round(2)
        out[f"Score Financials ({tag})"]=tmp["Score (Financials)"].round(2)
        out[f"Score Total ({tag})"]=tmp["Score (Total)"].round(2)
    return out

# ---------- massuppdatering ----------
import time
def sidebar_massupdate(df: pd.DataFrame, ws_title: str) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Massuppdatering")
    dly = st.sidebar.slider("Fördröjning mellan anrop (sek)", 0.3, 1.5, 0.6, 0.1)

    if st.sidebar.button("⚡ Snabb uppdatering (Yahoo pris m.m.)"):
        miss=[]
        prog = st.sidebar.progress(0.0)
        for i,(idx,row) in enumerate(df.iterrows(), start=1):
            tkr=str(row["Ticker"]).strip().upper()
            if not tkr: continue
            try:
                q = yf_quick(tkr)
                if q.get("Bolagsnamn"): df.at[idx,"Bolagsnamn"]=q["Bolagsnamn"]
                if q.get("Valuta"):     df.at[idx,"Valuta"]=q["Valuta"]
                df.at[idx,"Aktuell kurs"]=float(q.get("Aktuell kurs",0.0) or 0.0)
                df.at[idx,"Årlig utdelning"]=float(q.get("Årlig utdelning",0.0) or 0.0)
                df.at[idx,"CAGR 5 år (%)"]=float(q.get("CAGR 5 år (%)",0.0) or 0.0)
                df.at[idx,"Senast auto uppdaterad"]=now_stamp()
                df.at[idx,"Auto källa"]="Yahoo (snabb)"
            except Exception as e:
                miss.append(f"{tkr}: {e}")
            prog.progress(i/len(df))
            time.sleep(dly)
        df = update_calculations(df)
        save_df(ws_title, df)
        if miss:
            st.sidebar.warning("Vissa misslyckanden:\n" + "\n".join(miss))

    if st.sidebar.button("🛠 Full uppdatering (alla fetchers)"):
        miss=[]
        prog = st.sidebar.progress(0.0)
        for i,(idx,row) in enumerate(df.iterrows(), start=1):
            tkr=str(row["Ticker"]).strip().upper()
            if not tkr: continue
            try:
                y = yf_all(tkr) or {}
                if y.get("name"): df.at[idx,"Bolagsnamn"]=y["name"]
                if y.get("currency"): df.at[idx,"Valuta"]=y["currency"]
                if y.get("price"): df.at[idx,"Aktuell kurs"]=float(y["price"])
                if y.get("dividend_rate"): df.at[idx,"Årlig utdelning"]=float(y["dividend_rate"])
                if y.get("ps_ttm"): df.at[idx,"P/S"]=float(y["ps_ttm"])
                if y.get("pb"): df.at[idx,"P/B"]=float(y["pb"])
                if y.get("shares_outstanding"): df.at[idx,"Utestående aktier"]=float(y["shares_outstanding"])/1e6
                if y.get("cagr5_pct"): df.at[idx,"CAGR 5 år (%)"]=float(y["cagr5_pct"])
                df.at[idx,"Senast auto uppdaterad"]=now_stamp()
                df.at[idx,"Auto källa"]="Yahoo"

                fz = fz_overview(tkr) or {}
                if float(fz.get("ps_ttm",0))>0: df.at[idx,"P/S"]=float(fz["ps_ttm"])
                if float(fz.get("pb",0))>0: df.at[idx,"P/B"]=float(fz["pb"])

                ms = ms_overview(tkr) or {}
                if float(ms.get("ps_ttm",0))>0: df.at[idx,"P/S"]=float(ms["ps_ttm"])
                if float(ms.get("pb",0))>0: df.at[idx,"P/B"]=float(ms["pb"])

                sec = sec_pb_quarters(tkr) or {}
                pairs = sec.get("pb_quarters") or []
                if pairs:
                    for qi,(d,pbv) in enumerate(pairs[:4], start=1):
                        df.at[idx, f"P/B Q{qi}"] = float(pbv)
            except Exception as e:
                miss.append(f"{tkr}: {e}")
            prog.progress(i/len(df))
            time.sleep(dly)
        df = update_calculations(df)
        save_df(ws_title, df)
        if miss:
            st.sidebar.warning("Vissa misslyckanden:\n" + "\n".join(miss))
    return df

# ---------- vyer ----------
def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data")
    st.dataframe(df, use_container_width=True)

    c1,c2 = st.columns(2)
    horizon = c1.selectbox("Score-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    strat = c2.selectbox("Strategi", ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0)
    strat2 = "Auto" if strat.startswith("Auto") else strat

    if st.button("💾 Spara beräkningar"):
        df2 = update_calculations(df)
        df2 = score_rows(df2, horizon=horizon, strategy=strat2)
        df2 = compute_scores_all(df2, strat2)
        save_df(ws_title, df2)
        st.success("Beräkningar sparade.")

def view_manual(df: pd.DataFrame, ws_title: str):
    st.subheader("🧩 Manuell insamling")
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn","")).strip() else r["Ticker"] for _,r in vis.iterrows()]
    labels = ["➕ Lägg till nytt..."] + labels
    if "manual_idx" not in st.session_state: st.session_state["manual_idx"]=0
    sel = st.selectbox("Välj bolag", list(range(len(labels))), format_func=lambda i: labels[i], index=st.session_state["manual_idx"])
    st.session_state["manual_idx"]=sel
    is_new = (sel==0)
    row = (vis.iloc[sel-1] if not is_new and len(vis)>0 else pd.Series({c: "" for c in CANON_COLS}))

    st.markdown("### Obligatoriska")
    c1,c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker", value=str(row.get("Ticker","")).upper() if not is_new else "")
        antal  = st.number_input("Antal aktier", value=float(_to_float(row.get("Antal aktier",0.0))), step=1.0, min_value=0.0)
        gav    = st.number_input("GAV (SEK)", value=float(_to_float(row.get("GAV (SEK)",0.0))), step=0.01, min_value=0.0)
    with c2:
        oms_idag = st.number_input("Omsättning idag (M)", value=float(_to_float(row.get("Omsättning idag",0.0))), step=1.0, min_value=0.0)
        oms_nxt  = st.number_input("Omsättning nästa år (M)", value=float(_to_float(row.get("Omsättning nästa år",0.0))), step=1.0, min_value=0.0)

    with st.expander("🌐 Fält som hämtas (kan manuellt korrigeras)"):
        cL,cR = st.columns(2)
        with cL:
            bolagsnamn = st.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn","")))
            sektor     = st.text_input("Sektor", value=str(row.get("Sektor","")))
            valuta     = st.text_input("Valuta", value=str(row.get("Valuta","") or "USD").upper())
            aktuell    = st.number_input("Aktuell kurs", value=float(_to_float(row.get("Aktuell kurs",0.0))), step=0.01, min_value=0.0)
            utd        = st.number_input("Årlig utdelning", value=float(_to_float(row.get("Årlig utdelning",0.0))), step=0.01, min_value=0.0)
            payout     = st.number_input("Payout (%)", value=float(_to_float(row.get("Payout (%)",0.0))), step=1.0, min_value=0.0)
        with cR:
            utest      = st.number_input("Utestående aktier (miljoner)", value=float(_to_float(row.get("Utestående aktier",0.0))), step=1.0, min_value=0.0)
            ps  = st.number_input("P/S", value=float(_to_float(row.get("P/S",0.0))), step=0.01, min_value=0.0)
            ps1 = st.number_input("P/S Q1", value=float(_to_float(row.get("P/S Q1",0.0))), step=0.01, min_value=0.0)
            ps2 = st.number_input("P/S Q2", value=float(_to_float(row.get("P/S Q2",0.0))), step=0.01, min_value=0.0)
            ps3 = st.number_input("P/S Q3", value=float(_to_float(row.get("P/S Q3",0.0))), step=0.01, min_value=0.0)
            ps4 = st.number_input("P/S Q4", value=float(_to_float(row.get("P/S Q4",0.0))), step=0.01, min_value=0.0)
            pb  = st.number_input("P/B", value=float(_to_float(row.get("P/B",0.0))), step=0.01, min_value=0.0)
            pb1 = st.number_input("P/B Q1", value=float(_to_float(row.get("P/B Q1",0.0))), step=0.01, min_value=0.0)
            pb2 = st.number_input("P/B Q2", value=float(_to_float(row.get("P/B Q2",0.0))), step=0.01, min_value=0.0)
            pb3 = st.number_input("P/B Q3", value=float(_to_float(row.get("P/B Q3",0.0))), step=0.01, min_value=0.0)
            pb4 = st.number_input("P/B Q4", value=float(_to_float(row.get("P/B Q4",0.0))), step=0.01, min_value=0.0)

    if st.button("💾 Spara (hämtar även snabb Yahoo)"):
        if not ticker.strip():
            st.error("Ticker krävs."); return
        exists = (df["Ticker"].astype(str).str.upper() == ticker.upper())
        upd = {
            "Ticker": ticker.upper(),
            "Antal aktier": float(antal),
            "GAV (SEK)": float(gav),
            "Omsättning idag": float(oms_idag),
            "Omsättning nästa år": float(oms_nxt),
            "Bolagsnamn": str(bolagsnamn).strip(),
            "Sektor": str(sektor).strip(),
            "Valuta": str(valuta).strip().upper(),
            "Aktuell kurs": float(aktuell),
            "Årlig utdelning": float(utd),
            "Payout (%)": float(payout),
            "Utestående aktier": float(utest),
            "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
            "P/B": float(pb), "P/B Q1": float(pb1), "P/B Q2": float(pb2), "P/B Q3": float(pb3), "P/B Q4": float(pb4),
        }
        if exists.any():
            for k,v in upd.items(): df.loc[exists, k] = v
            df.loc[exists, "Senast manuellt uppdaterad"] = now_stamp()
        else:
            base = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad","Auto källa","Senast beräknad","Div_Månader","Div_Vikter"] else "") for c in CANON_COLS}
            base.update(upd)
            base["Senast manuellt uppdaterad"] = now_stamp()
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
            exists = (df["Ticker"].astype(str).str.upper() == ticker.upper())

        # snabb yahoo
        try:
            quick = yf_quick(ticker.upper())
            if quick.get("Bolagsnamn"): df.loc[exists, "Bolagsnamn"] = quick["Bolagsnamn"]
            if quick.get("Valuta"): df.loc[exists, "Valuta"] = quick["Valuta"]
            df.loc[exists, "Aktuell kurs"] = float(quick.get("Aktuell kurs",0.0) or 0.0)
            df.loc[exists, "Årlig utdelning"] = float(quick.get("Årlig utdelning",0.0) or 0.0)
            df.loc[exists, "CAGR 5 år (%)"] = float(quick.get("CAGR 5 år (%)",0.0) or 0.0)
            df.loc[exists, "Senast auto uppdaterad"] = now_stamp()
            df.loc[exists, "Auto källa"] = "Yahoo (snabb)"
        except Exception:
            pass

        df2 = update_calculations(df)
        save_df(ws_title, df2)
        st.success("Sparat.")

def view_portfolio(df: pd.DataFrame, rates: Dict[str,float]):
    st.subheader("📦 Portfölj")
    port = df[df["Antal aktier"].map(_to_float) > 0].copy()
    if port.empty:
        st.info("Inga innehav ännu."); return
    port["vx"] = port["Valuta"].astype(str).str.upper().map(lambda c: float(rates.get(c,1.0)))
    port["Värde (SEK)"] = port["Antal aktier"].map(_to_float) * port["Aktuell kurs"].map(_to_float) * port["vx"]
    port["Anskaffn (SEK)"] = port["Antal aktier"].map(_to_float) * port["GAV (SEK)"].map(_to_float)
    tot = float(port["Värde (SEK)"].sum())
    tot_acq = float(port["Anskaffn (SEK)"].sum())
    gain = tot - tot_acq
    pct = (gain / tot_acq * 100.0) if tot_acq > 0 else 0.0

    st.markdown(f"**Portföljvärde:** {round(tot,2)} SEK")
    st.markdown(f"**Anskaffningsvärde:** {round(tot_acq,2)} SEK")
    st.markdown(f"**Vinst:** {round(gain,2)} SEK ({round(pct,2)}%)")

    port["Andel (%)"] = np.where(tot>0, (port["Värde (SEK)"]/tot)*100.0, 0.0).round(2)
    port["DA (%)"] = np.where(port["Aktuell kurs"].map(_to_float)>0, (port["Årlig utdelning"].map(_to_float)/port["Aktuell kurs"].map(_to_float))*100.0, 0.0).round(2)

    cols = ["Ticker","Bolagsnamn","Sektor","Antal aktier","GAV (SEK)","Aktuell kurs","Valuta","Värde (SEK)","Anskaffn (SEK)","Andel (%)","Årlig utdelning","DA (%)"]
    st.dataframe(port[cols].sort_values("Andel (%)", ascending=False), use_container_width=True)

def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")
    if df.empty:
        st.info("Tomt blad."); return

    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    strategy = st.selectbox("Strategi", ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0)
    strat = "Auto" if strategy.startswith("Auto") else strategy

    subset = st.radio("Visa", ["Alla bolag","Endast portfölj"], horizontal=True)
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"].map(_to_float) > 0].copy()

    base = update_calculations(base)
    base = base[(base[horizon].map(_to_float) > 0) & (base["Aktuell kurs"].map(_to_float) > 0)].copy()
    if base.empty:
        st.info("Inget att visa just nu."); return

    base = score_rows(base, horizon=horizon, strategy=strat)

    base["Uppsida (%)"] = ((base[horizon].map(_to_float) - base["Aktuell kurs"].map(_to_float)) / base["Aktuell kurs"].map(_to_float) * 100.0).round(2)
    base["DA (%)"] = np.where(base["Aktuell kurs"].map(_to_float) > 0, (base["Årlig utdelning"].map(_to_float)/base["Aktuell kurs"].map(_to_float))*100.0, 0.0).round(2)

    sort_on = st.selectbox("Sortera på", ["Score (Total)","Uppsida (%)","DA (%)"], index=0)
    ascending = st.checkbox("Omvänd sortering", value=False)
    if sort_on=="Uppsida (%)" and st.checkbox("Trim/sälj-läge (minst uppsida först)", value=False):
        ascending=True

    cols = ["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)","DA (%)","Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence"]
    base = base.sort_values(by=[sort_on], ascending=ascending).reset_index(drop=True)
    st.dataframe(base[cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### Kortvisning")
    if "idea_idx" not in st.session_state: st.session_state["idea_idx"]=0
    st.session_state["idea_idx"] = st.number_input("Visa rad #", 0, max(0,len(base)-1), st.session_state["idea_idx"], 1)
    r = base.iloc[st.session_state["idea_idx"]]
    tag=_horizon_tag(horizon)
    c1,c2 = st.columns(2)
    with c1:
        st.write(f"**{r['Bolagsnamn']} ({r['Ticker']})**")
        st.write(f"Sektor: {r.get('Sektor','—')}")
        st.write(f"Aktuell kurs: {round(_to_float(r['Aktuell kurs']),2)} {r['Valuta']}")
        st.write(f"Riktkurs {tag}: {round(_to_float(r[horizon]),2)} {r['Valuta']} (Uppsida: {round(_to_float(r['Uppsida (%)']),2)}%)")
    with c2:
        st.write(f"P/S-snitt: {round(_to_float(r['P/S-snitt (Q1..Q4)']),2)}  |  P/B-snitt: {round(_to_float(r['P/B-snitt (Q1..Q4)']),2)}")
        st.write(f"Omsättning idag/1/2/3 år (M): {round(_to_float(r['Omsättning idag']),2)} / {round(_to_float(r['Omsättning nästa år']),2)} / {round(_to_float(r['Omsättning om 2 år']),2)} / {round(_to_float(r['Omsättning om 3 år']),2)}")
        st.write(f"DA: {round(_to_float(r['DA (%)']),2)}%  |  Payout: {round(_to_float(r['Payout (%)']),2)}%")
        st.write(f"Score G/D/F/T: {round(_to_float(r['Score (Growth)']),1)} / {round(_to_float(r['Score (Dividend)']),1)} / {round(_to_float(r['Score (Financials)']),1)} / **{round(_to_float(r['Score (Total)']),1)}** (Conf {int(_to_float(r['Confidence']))}%)")

def view_dividend_calendar(df: pd.DataFrame, ws_title: str, rates: Dict[str,float]):
    st.subheader("📅 Utdelningskalender")
    months = st.number_input("Antal månader framåt", 3, 24, 12, 1)
    if st.button("Bygg kalender"):
        summ, det, df_out = build_dividend_calendar(df, rates, months_forward=int(months), write_back_schedule=False)
        st.session_state["div_summ"]=summ
        st.session_state["div_det"]=det
        st.session_state["div_df_out"]=df_out
        st.success("Kalender skapad.")
    if "div_summ" in st.session_state:
        st.markdown("### Summering per månad (SEK)")
        st.dataframe(st.session_state["div_summ"], use_container_width=True)
    if "div_det" in st.session_state:
        st.markdown("### Detalj (SEK)")
        st.dataframe(st.session_state["div_det"], use_container_width=True)
    if st.button("💾 Spara schema + kalender till Google Sheets"):
        try:
            df2 = update_calculations(st.session_state.get("div_df_out", df))
            save_df(ws_title, df2)
            ws_write_df("Utdelningskalender – Summering", st.session_state.get("div_summ", pd.DataFrame()))
            ws_write_df("Utdelningskalender – Detalj", st.session_state.get("div_det", pd.DataFrame()))
            st.success("Sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

# ---------- main ----------
def main():
    st.title("K-pf-rslag")

    try:
        titles = list_worksheet_titles() or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → data-blad", titles, index=0)

    if st.sidebar.button("↻ Läs om data"):
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce",0)+1
        st.rerun()

    rates = sidebar_rates()

    df = load_df(ws_title)
    snapshot_on_start(df, ws_title)
    df = update_calculations(df)

    # Massuppdatering i sidomeny
    df = sidebar_massupdate(df, ws_title)

    tabs = st.tabs(["📄 Data","🧩 Manuell","📦 Portfölj","💡 Förslag","📅 Utdelningar"])
    with tabs[0]:
        view_data(df, ws_title)
    with tabs[1]:
        view_manual(df, ws_title)
    with tabs[2]:
        view_portfolio(df, rates)
    with tabs[3]:
        view_ideas(df)
    with tabs[4]:
        view_dividend_calendar(df, ws_title, rates)

if __name__ == "__main__":
    main()
