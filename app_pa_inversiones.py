import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import traceback
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
from io import StringIO, BytesIO
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════
#  IMPORTACIONES SEGURAS
# ═══════════════════════════════════════════════════════════════════════════
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_OK = True
except ImportError:
    GSHEETS_OK = False

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

try:
    from PyPDF2 import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
    def page_forecast(): st.warning("📦 forecast_module.py no encontrado.")
    def page_iol_explorer(): st.warning("📦 iol_client.py no encontrado.")
    def get_iol_client(): return None

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="INVERSIONES PRO", page_icon="📈")

SHEET_NAME = st.secrets.get("google_sheets", {}).get("sheet_name", "Epre_Inversiones")
SHEET_ID   = st.secrets.get("google_sheets", {}).get("sheet_id", "")
WORKSHEET_NAME = "portfolios"
PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES & CORRECCIÓN DE ERRORES
# ═══════════════════════════════════════════════════════════════════════════

def safe_join_list(data, fallback="No disponible"):
    """Corrige TypeError: can only join an iterable"""
    if isinstance(data, list):
        return ", ".join(str(x) for x in data)
    elif isinstance(data, str):
        return data
    elif data is None:
        return fallback
    return str(data)

def extract_text_from_file(uploaded_file, max_chars: int = 15000) -> str:
    if uploaded_file is None:
        return ""
    try:
        file_type = uploaded_file.type
        content = ""
        if file_type == "application/pdf" and PDF_OK:
            reader = PdfReader(uploaded_file)
            pages_to_read = min(10, len(reader.pages))
            for i in range(pages_to_read):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    content += page_text + "\n"
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and DOCX_OK:
            doc = Document(uploaded_file)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
            content = df.head(50).to_markdown(index=False)
        elif file_type in ["text/plain", "text/markdown"]:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        else:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")[:max_chars]
        
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...contenido truncado...]"
        return content.strip()
    except Exception as e:
        return f"⚠️ Error al procesar: {e}"

def truncate_for_tokens(text: str, max_tokens: int = 8000) -> str:
    if len(text) <= max_tokens * 4:
        return text
    chunk = max_tokens * 2
    return text[:chunk] + "\n\n[...omitido...]\n\n" + text[-chunk:]

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTRUCTOR DE CONTEXTO PARA IA
# ═══════════════════════════════════════════════════════════════════════════

def build_portfolio_context(res: dict, prices: pd.DataFrame = None, 
                           portfolio_name: str = "Portafolio",
                           include_correlations: bool = True) -> str:
    lines = []
    lines.append(f"📊 ANÁLISIS: {portfolio_name}")
    lines.append(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("🎯 MÉTRICAS GLOBALES:")
    lines.append(f"- Retorno esperado anual: {res['expected_return']:.2%}")
    lines.append(f"- Volatilidad anualizada: {res['volatility']:.2%}")
    lines.append(f"- Ratio Sharpe: {res['sharpe_ratio']:.2f}")
    lines.append(f"- Método: {res.get('method', 'N/A')}")
    lines.append("")
    
    lines.append("🧩 COMPOSICIÓN:")
    active_assets = [(t, w) for t, w in zip(res['tickers'], res['weights']) if w > 0.001]
    active_assets.sort(key=lambda x: x[1], reverse=True)
    total_weight = sum(w for _, w in active_assets)
    
    for ticker, weight in active_assets:
        pct = weight / total_weight * 100 if total_weight > 0 else 0
        lines.append(f"- {ticker:<10}: {pct:5.1f}% (peso: {weight:.4f})")
    
    if active_assets and active_assets[0][1]/total_weight > 0.4:
        lines.append(f"  ⚠️ Concentración alta en {active_assets[0][0]}")
    lines.append("")
    
    if prices is not None and not prices.empty and len(prices) >= 30:
        returns = prices.pct_change().dropna()
        lines.append("📈 MÉTRICAS INDIVIDUALES:")
        for ticker, weight in active_assets:
            if ticker in prices.columns and ticker in returns.columns:
                ann_ret = returns[ticker].mean() * 252
                ann_vol = returns[ticker].std() * np.sqrt(252)
                sharpe_ind = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
                lines.append(f"- {ticker}: Ret {ann_ret:6.1%} | Vol {ann_vol:5.1%} | Sharpe {sharpe_ind:5.2f}")
        lines.append("")
    
    if include_correlations and prices is not None and len(prices.columns) >= 2:
        returns = prices.pct_change().dropna()
        if len(returns) >= 30:
            corr = returns.corr()
            high_corr_pairs = []
            for i, col in enumerate(corr.columns):
                for j, idx in enumerate(corr.index):
                    if i < j and col in [t for t,_ in active_assets] and idx in [t for t,_ in active_assets]:
                        c_val = corr.loc[idx, col]
                        if abs(c_val) > 0.65:
                            high_corr_pairs.append((idx, col, c_val))
            if high_corr_pairs:
                lines.append("🔗 CORRELACIONES ALTAS:")
                for asset1, asset2, corr_val in high_corr_pairs:
                    w1 = next((w for t, w in active_assets if t == asset1), 0)
                    w2 = next((w for t, w in active_assets if t == asset2), 0)
                    combined_weight = (w1 + w2) / total_weight if total_weight > 0 else 0
                    signal = "🔴" if corr_val > 0.8 else "🟡"
                    lines.append(f"  {signal} {asset1} ↔ {asset2}: {corr_val:+.2f}")
                lines.append("")
    
    lines.append("🌍 EXPOSICIÓN IMPLÍCITA:")
    exposures = {"ARS": 0, "USD": 0, "Equity": 0, "FixedIncome": 0}
    for ticker, weight in active_assets:
        t_upper = ticker.upper()
        if any(x in t_upper for x in ["AL30", "GD30", "GGAL", "YPF", "PAM", "TX26", "CEPU", "AR"]) and ".BA" not in t_upper:
            exposures["ARS"] += weight
            exposures["Equity"] += weight if any(x in t_upper for x in ["GGAL", "YPF", "PAM", "CEPU"]) else 0
            exposures["FixedIncome"] += weight if any(x in t_upper for x in ["AL30", "GD30", "TX26"]) else 0
        elif "=X" in t_upper or any(x in t_upper for x in ["USD", "EUR"]):
            exposures["USD"] += weight
        elif any(x in t_upper for x in ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ"]):
            exposures["USD"] += weight
            exposures["Equity"] += weight
    
    for exp_type, exp_weight in exposures.items():
        if exp_weight > 0.01:
            pct = exp_weight / total_weight * 100 if total_weight > 0 else 0
            lines.append(f"- {exp_type}: {pct:.1f}%")
    
    if prices is not None and not prices.empty:
        lines.append(f"\n📊 DATOS: {len(prices)} observaciones")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
#  GOOGLE SHEETS & GESTIÓN DE PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_gsheets_client():
    if not GSHEETS_OK: return None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        return None

def get_or_create_worksheet(client):
    try:
        ss = client.open_by_key(SHEET_ID) if SHEET_ID else client.open(SHEET_NAME)
        try: return ss.worksheet(WORKSHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = ss.add_worksheet(title=WORKSHEET_NAME, rows=200, cols=3)
            ws.append_row(["name", "tickers", "weights"])
            return ws
    except: return None

def load_portfolios() -> dict:
    client = get_gsheets_client()
    if client:
        ws = get_or_create_worksheet(client)
        if ws:
            try:
                records = ws.get_all_records()
                portfolios = {}
                for row in records:
                    name = str(row.get("name", "")).strip()
                    raw_tickers = str(row.get("tickers", "")).strip()
                    raw_weights = str(row.get("weights", "")).strip()
                    if not name or not raw_tickers: continue
                    tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
                    try: weights = [float(w.strip()) for w in raw_weights.split(",") if w.strip()]
                    except: weights = [1.0 / len(tickers)] * len(tickers)
                    total_w = sum(weights)
                    if total_w > 0: weights = [w / total_w for w in weights]
                    portfolios[name] = {"tickers": tickers, "weights": weights}
                return portfolios
            except: pass
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f: return json.load(f)
        except: pass
    return {}

def save_portfolios(pf: dict) -> bool:
    try:
        with open(PORTFOLIO_FILE, "w") as f: json.dump(pf, f, indent=4)
        client = get_gsheets_client()
        if client:
            ws = get_or_create_worksheet(client)
            if ws:
                ws.clear()
                rows = [["name", "tickers", "weights"]]
                for n, d in pf.items():
                    rows.append([n, ", ".join(d["tickers"]), ", ".join(str(w) for w in d["weights"])])
                ws.update(rows, "A1")
        return True
    except: return False

# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    client = get_iol_client()
    all_prices = {}
    yf_tickers = []
    for ticker in tickers:
        fetched = False
        if client:
            try:
                df_hist = client.get_serie_historica(ticker.split(".")[0].upper(), 
                                                     pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                                                     pd.to_datetime(end_date).strftime("%Y-%m-%d"))
                if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                    s = df_hist["ultimoPrecio"].rename(ticker)
                    if s.index.tz is not None: s.index = s.index.tz_localize(None)
                    all_prices[ticker] = s
                    fetched = True
            except: pass
        if not fetched:
            yf_tickers.append(ticker)
    
    if yf_tickers:
        try:
            adjusted = [t if "." in t or t.endswith("=X") else t+".BA" for t in yf_tickers]
            raw = yf.download(adjusted, start=start_date, end=end_date, progress=False)
            if not raw.empty:
                close_data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) and 'Close' in raw.columns.levels[0] else raw
                if isinstance(close_data, pd.Series): close_data = close_data.to_frame()
                for col in close_data.columns:
                    clean = str(col).replace(".BA", "")
                    for orig in yf_tickers:
                        if clean == orig: all_prices[orig] = close_data[col]
        except: pass
            
    if not all_prices: return None
    prices = pd.concat(all_prices.values(), axis=1).ffill().dropna()
    if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)
    return prices

def optimize_portfolio_corporate(prices, risk_free_rate=0.02, opt_type="Maximo Ratio Sharpe"):
    returns = prices.pct_change().dropna()
    if returns.empty or len(returns) < 30: return None
    
    if PYPFOPT_OK:
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.sample_cov(prices, frequency=252)
            if not mu.isnull().any() and not S.isnull().values.any():
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                if opt_type == "Maximo Ratio Sharpe": ef.max_sharpe(risk_free_rate=risk_free_rate)
                elif opt_type == "Minima Volatilidad": ef.min_volatility()
                else: ef.max_quadratic_utility(risk_aversion=0.01)
                weights = ef.clean_weights()
                ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                return {"weights": np.array([weights.get(col, 0) for col in prices.columns]), 
                        "expected_return": float(ret), "volatility": float(vol), "sharpe_ratio": float(sharpe),
                        "tickers": list(prices.columns), "method": "PyPortfolioOpt-Markowitz"}
        except: pass
    
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(mean_returns)
    def get_metrics(w):
        ret = np.sum(mean_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0
        return np.array([ret, vol, sr])
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.array([1/n] * n)
    
    if opt_type == "Minima Volatilidad": fun = lambda w: get_metrics(w)[1]
    elif opt_type == "Retorno Maximo": fun = lambda w: -get_metrics(w)[0]
    else: fun = lambda w: -get_metrics(w)[2] if (mean_returns >= risk_free_rate).any() else get_metrics(w)[1]
    
    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    w = np.maximum(res.x, 0)
    if w.sum() > 0: w /= w.sum()
    m = get_metrics(w)
    return {"weights": w, "expected_return": float(m[0]), "volatility": float(m[1]), 
            "sharpe_ratio": float(m[2]), "tickers": list(prices.columns), "method": "Scipy-SLSQP"}

def optimize_risk_parity(prices, risk_free_rate=0.02):
    if not PYPFOPT_OK: return None
    returns = prices.pct_change().dropna()
    if len(returns) < 30: return None
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        rp = EfficientFrontier(None, S, weight_bounds=(0, 1))
        w = rp.risk_parity()
        wa = np.array([w.get(c, 0) for c in prices.columns])
        mu = returns.mean() * 252
        r, v = np.dot(mu, wa), np.sqrt(np.dot(wa, np.dot(S, wa)))
        return {"weights": wa, "expected_return": r, "volatility": v, "sharpe_ratio": (r-risk_free_rate)/v if v>0 else 0, "tickers": list(prices.columns), "method": "Risk Parity"}
    except: return None

def optimize_hierarchical_risk_parity(prices, risk_free_rate=0.02):
    if not PYPFOPT_OK: return None
    returns = prices.pct_change().dropna()
    if len(returns) < 30: return None
    try:
        S = risk_models.sample_cov(prices, frequency=252)
        hrp = HRPOpt(cov_matrix=S, returns=returns)
        w = hrp.optimize()
        wa = np.array([w.get(c, 0) for c in prices.columns])
        mu = returns.mean() * 252
        r, v = np.dot(mu, wa), np.sqrt(np.dot(wa, np.dot(S, wa)))
        return {"weights": wa, "expected_return": r, "volatility": v, "sharpe_ratio": (r-risk_free_rate)/v if v>0 else 0, "tickers": list(prices.columns), "method": "Hierarchical Risk Parity"}
    except: return None

def optimize_black_litterman(prices, risk_free_rate=0.02, views=None, confs=None):
    if not PYPFOPT_OK: return None
    returns = prices.pct_change().dropna()
    if len(returns) < 30: return None
    try:
        S = risk_models.sample_cov(prices, frequency=252)
        mc = {c: 1.0 for c in prices.columns}
        bl = EfficientFrontier(None, S, weight_bounds=(0, 1))
        if views and confs:
            tks = list(views.keys())
            P = np.zeros((len(tks), len(prices.columns)))
            for i, t in enumerate(tks):
                if t in prices.columns: P[i, list(prices.columns).index(t)] = 1
            bl.black_litterman(mc, P, list(views.values()), [confs.get(t, 0.5) for t in tks], risk_free_rate)
        else:
            bl.black_litterman(mc)
        w = bl.clean_weights()
        wa = np.array([w.get(c, 0) for c in prices.columns])
        mu = returns.mean() * 252
        r, v = np.dot(mu, wa), np.sqrt(np.dot(wa, np.dot(S, wa)))
        return {"weights": wa, "expected_return": r, "volatility": v, "sharpe_ratio": (r-risk_free_rate)/v if v>0 else 0, "tickers": list(prices.columns), "method": "Black-Litterman"}
    except: return None

# ═══════════════════════════════════════════════════════════════════════════
#  REBALANCEO & MÉTRICAS AVANZADAS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_rebalancing_trades(current_portfolio, target_weights, current_prices, total_value, min_trade_size=100):
    trades = []
    for ticker, tw in target_weights.items():
        cv = current_portfolio.get(ticker, 0)
        tv = total_value * tw
        diff = tv - cv
        if abs(diff) >= min_trade_size:
            p = current_prices.get(ticker, 1)
            trades.append({"Activo": ticker, "Valor Actual": f"${cv:,.0f}", "Valor Target": f"${tv:,.0f}",
                           "Diferencia": f"${diff:,.0f}", "Precio": f"${p:,.2f}", "Cantidad": f"{abs(diff/p):.2f}",
                           "Acción": "🟢 COMPRAR" if diff > 0 else "🔴 VENDER"})
    return pd.DataFrame(trades)

def get_rebalancing_strategy(current_weights, target_weights, threshold=0.05):
    drifts = {t: abs(current_weights.get(t, 0) - target_weights.get(t, 0)) for t in target_weights}
    mx = max(drifts.values()) if drifts else 0
    return {"max_drift": mx, "needs": mx > threshold, "status": "⚠️ REBALANCEAR" if mx > threshold else "✅ OK"}

def calculate_portfolio_metrics(prices, weights, risk_free_rate=0.02):
    returns = prices.pct_change().dropna()
    port_ret = (returns * weights).sum(axis=1)
    ann_r = port_ret.mean() * 252; ann_v = port_ret.std() * np.sqrt(252)
    sharpe = (ann_r - risk_free_rate) / ann_v if ann_v > 0 else 0
    down = port_ret[port_ret < 0]; dv = down.std() * np.sqrt(252) if len(down) > 0 else ann_v
    sortino = (ann_r - risk_free_rate) / dv if dv > 0 else 0
    cum = (1 + port_ret).cumprod(); mdd = (cum / cum.cummax() - 1).min()
    calmar = ann_r / abs(mdd) if mdd != 0 else 0
    var95 = np.percentile(port_ret, 5)
    cvar95 = port_ret[port_ret <= var95].mean() if len(port_ret[port_ret <= var95]) > 0 else var95
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "mdd": mdd, "var95": var95, 
            "cvar95": cvar95, "skew": port_ret.skew(), "kurt": port_ret.kurtosis(), "rets": port_ret}

def display_advanced_metrics(prices, res):
    w = {t: w for t, w in zip(res['tickers'], res['weights'])}
    m = calculate_portfolio_metrics(prices, w)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sharpe", f"{m['sharpe']:.2f}"); c2.metric("Sortino", f"{m['sortino']:.2f}")
    c3.metric("Calmar", f"{m['calmar']:.2f}"); c4.metric("Max DD", f"{m['mdd']:.1%}")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("VaR 95%", f"{m['var95']:.2%}"); c6.metric("CVaR", f"{m['cvar95']:.2%}")
    c7.metric("Skewness", f"{m['skew']:.2f}"); c8.metric("Kurtosis", f"{m['kurt']:.2f}")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=m['rets'], nbinsx=50, marker_color='rgba(0,204,150,0.7)'))
    fig.add_vline(x=m['var95'], line_dash="dash", line_color="red", annotation_text="VaR 95%")
    fig.update_layout(title="Distribución de Retornos Diarios", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo Integral")
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("👈 Crea tu primer portafolio en la pestaña 'Gestión'")
        return

    tabs = st.tabs(["💼 Gestión", "🚀 Optimización Avanzada", "🔄 Rebalanceo", "🔮 Forecast"])

    with tabs[0]:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("Administrar Carteras")
            action = st.radio("Acción:", ["✨ Crear", "✏️ Editar/🗑️ Eliminar"], horizontal=True, key="dash_action")
            if action == "✨ Crear":
                p_name = st.text_input("Nombre", key="new_name")
                p_tickers = st.text_area("Tickers (coma)", "AAPL, SPY", key="new_tks").upper()
                p_weights = st.text_area("Pesos (coma)", "0.5, 0.5", key="new_ws")
                if st.button("💾 Guardar", type="primary", key="save_new"):
                    try:
                        ts = [x.strip() for x in p_tickers.split(",") if x.strip()]
                        ws = [float(x) for x in p_weights.split(",") if x.strip()]
                        if p_name and len(ts) == len(ws):
                            tot = sum(ws); ws = [w/tot for w in ws]
                            portfolios[p_name] = {"tickers": ts, "weights": ws}
                            if save_portfolios(portfolios): st.success("✅ Guardado"); st.rerun()
                        else: st.error("Verifica datos")
                    except: st.error("Formato inválido")
            else:
                if portfolios:
                    sel = st.selectbox("Seleccionar", list(portfolios.keys()), key="edit_sel")
                    d = portfolios[sel]
                    nn = st.text_input("Renombrar", value=sel, key="ren")
                    nt = st.text_area("Tickers", value=", ".join(d["tickers"]), key="et")
                    nw = st.text_area("Pesos", value=", ".join(f"{w:.3f}" for w in d["weights"]), key="ew")
                    col_b1, col_b2 = st.columns(2)
                    if col_b1.button("🔄 Actualizar", key="upd"):
                        try:
                            ts = [x.strip() for x in nt.split(",") if x.strip()]
                            ws = [float(x) for x in nw.split(",") if x.strip()]
                            if len(ts) == len(ws):
                                tot = sum(ws); ws = [w/tot for w in ws]
                                if nn != sel: del portfolios[sel]
                                portfolios[nn] = {"tickers": ts, "weights": ws}
                                save_portfolios(portfolios); st.success("✅ Actualizado"); st.rerun()
                        except: st.error("Error")
                    if col_b2.button("🗑️ Eliminar", key="del"):
                        del portfolios[sel]; save_portfolios(portfolios); st.warning("Eliminado"); st.rerun()
                else: st.info("No hay carteras")
        with c2:
            st.subheader("📋 Base de Datos")
            if portfolios:
                df = pd.DataFrame([{"Nombre": k, "Activos": len(v["tickers"]), "Tickers": ", ".join(v["tickers"]), "Peso Mayor": f"{max(v['weights'])*100:.1f}%"} for k, v in portfolios.items()])
                st.dataframe(df, hide_index=True, use_container_width=True)

    with tabs[1]:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        p_sel = col1.selectbox("📦 Cartera", list(portfolios.keys()), key="opt_pf")
        ds = col2.date_input("Desde", pd.to_datetime("2023-01-01"), key="opt_ds")
        de = col3.date_input("Hasta", pd.to_datetime("today"), key="opt_de")
        
        if st.button("📊 Ver Histórico", key="hist_btn"):
            with st.spinner("Cargando..."):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], ds, de)
            if prices is not None:
                tks = [t for t in portfolios[p_sel]["tickers"] if t in prices.columns]
                if tks:
                    w = np.array([portfolios[p_sel]["weights"][portfolios[p_sel]["tickers"].index(t)] for t in tks])
                    w /= w.sum()
                    norm = prices[tks] / prices[tks].iloc[0]
                    val = (norm * w).sum(axis=1) * 100
                    r = (val.iloc[-1]/val.iloc[0]-1)*100
                    v = val.pct_change().dropna().std()*np.sqrt(252)*100
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Retorno", f"{r:.1f}%"); m2.metric("Volatilidad", f"{v:.1f}%")
                    m3.metric("Max DD", f"{((val/val.cummax()-1).min()*100):.1f}%")
                    m4.metric("Sharpe", f"{(val.pct_change().mean()*252)/(val.pct_change().std()*np.sqrt(252)):.2f}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=val.index, y=val, name=p_sel, line=dict(width=3, color='#00CC96')))
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("⚙️ Optimización Avanzada")
        opt_tabs = st.tabs(["📊 Markowitz", "⚖️ Risk Parity", "🎯 Black-Litterman", "🌳 HRP", "🔄 Comparar"])
        
        with opt_tabs[0]:
            rf = st.number_input("RF", 0.0, 0.5, 0.04, key="rf_m")
            tgt = st.selectbox("Objetivo", ["Maximo Ratio Sharpe","Minima Volatilidad","Retorno Maximo"], key="tgt_m")
            if st.button("Optimizar Markowitz", key="run_m"):
                with st.spinner("Optimizando..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], ds, de)
                if prices:
                    res = optimize_portfolio_corporate(prices, rf, tgt)
                    if res: st.session_state['opt_res'] = res; st.session_state['opt_prices'] = prices; st.success("✅ Listo"); st.rerun()
        
        with opt_tabs[1]:
            if st.button("Optimizar Risk Parity", key="run_rp"):
                with st.spinner("Calculando..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], ds, de)
                if prices:
                    res = optimize_risk_parity(prices)
                    if res: st.session_state['opt_res'] = res; st.session_state['opt_prices'] = prices; st.success("✅ Listo"); st.rerun()
        
        with opt_tabs[2]:
            if st.button("Optimizar Black-Litterman", key="run_bl"):
                with st.spinner("Calculando..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], ds, de)
                if prices:
                    res = optimize_black_litterman(prices)
                    if res: st.session_state['opt_res'] = res; st.session_state['opt_prices'] = prices; st.success("✅ Listo"); st.rerun()
        
        with opt_tabs[3]:
            if st.button("Optimizar HRP", key="run_hrp"):
                with st.spinner("Calculando..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], ds, de)
                if prices:
                    res = optimize_hierarchical_risk_parity(prices)
                    if res: st.session_state['opt_res'] = res; st.session_state['opt_prices'] = prices; st.success("✅ Listo"); st.rerun()
        
        with opt_tabs[4]:
            if st.button("Comparar Todos", key="cmp"):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], ds, de)
                if prices:
                    rows = []
                    for fn, name in [(optimize_portfolio_corporate, "Markowitz"), (optimize_risk_parity, "Risk Parity"), (optimize_hierarchical_risk_parity, "HRP")]:
                        r = fn(prices) if fn != optimize_portfolio_corporate else fn(prices, 0.04, "Maximo Ratio Sharpe")
                        if r: rows.append({"Método": r["method"], "Retorno": r["expected_return"], "Vol": r["volatility"], "Sharpe": r["sharpe_ratio"]})
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df.style.format({"Retorno":"{:.2%}","Vol":"{:.2%}","Sharpe":"{:.3f}"}), use_container_width=True)
                        fig = px.bar(df, x="Método", y="Sharpe", color="Sharpe", color_continuous_scale="Viridis")
                        st.plotly_chart(fig, use_container_width=True)

        if 'opt_res' in st.session_state and st.session_state.get('opt_prices') is not None:
            res = st.session_state['opt_res']; prices = st.session_state['opt_prices']
            c1,c2,c3 = st.columns(3)
            c1.metric("Retorno", f"{res['expected_return']:.1%}")
            c2.metric("Volatilidad", f"{res['volatility']:.1%}")
            c3.metric("Sharpe", f"{res['sharpe_ratio']:.2f}")
            wdf = pd.DataFrame({"Activo": res['tickers'], "Peso": res['weights']})
            fig = px.pie(wdf[wdf["Peso"]>0.001], values="Peso", names="Activo", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            display_advanced_metrics(prices, res)
            
            if st.button("🧠 Analizar con IA", key="ia_analyze"):
                if not st.session_state.get("preferred_ai"): st.warning("Configura API Key"); return
                with st.spinner("IA analizando..."):
                    ctx = build_portfolio_context(res, prices, p_sel)
                    prompt = f"Actúa como CIO. Analiza:\n{ctx}\n1. Diversificación 2. Riesgos 3. Ajustes 4. Alertas. Español."
                    try:
                        if st.session_state.preferred_ai == "OpenAI":
                            client = OpenAI(api_key=st.session_state.openai_api_key)
                            st.info(client.chat.completions.create(model=st.session_state.openai_model, messages=[{"role":"user","content":prompt}]).choices[0].message.content)
                        else:
                            genai.configure(api_key=st.session_state.gemini_api_key)
                            st.info(genai.GenerativeModel(st.session_state.gemini_model).generate_content(prompt).text)
                    except Exception as e: st.error(f"Error IA: {e}")

    with tabs[2]:
        if 'opt_res' not in st.session_state: st.info("Optimiza primero"); return
        res = st.session_state['opt_res']
        tgt_w = {t: w for t, w in zip(res['tickers'], res['weights'])}
        cur_w = {t: w for t, w in zip(portfolios[p_sel]["tickers"], portfolios[p_sel]["weights"])}
        drift = get_rebalancing_strategy(cur_w, tgt_w)
        
        c1,c2,c3 = st.columns(3)
        c1.metric("Drift Máx", f"{drift['max_drift']:.2%}"); c2.metric("Umbral", "5.0%"); c3.metric("Estado", drift['status'])
        
        df = pd.DataFrame({"Activo": list(tgt_w.keys()), "Actual (%)": [cur_w.get(t,0)*100 for t in tgt_w], "Target (%)": [tgt_w[t]*100 for t in tgt_w]})
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Activo"], y=df["Actual (%)"], name="Actual", marker_color='rgba(239,85,59,0.7)'))
        fig.add_trace(go.Bar(x=df["Activo"], y=df["Target (%)"], name="Target", marker_color='rgba(0,204,150,0.7)'))
        fig.update_layout(barmode='group', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        if drift['needs']:
            val = st.number_input("Valor Portafolio ($)", 10000, 1000000, 100000)
            if st.button("📥 Generar Órdenes"):
                prices_now = {t: yf.Ticker(t).info.get("currentPrice", 1) for t in tgt_w}
                trades = calculate_rebalancing_trades({t: val*cur_w.get(t,0) for t in tgt_w}, tgt_w, prices_now, val)
                if not trades.empty:
                    st.dataframe(trades, use_container_width=True)
                    st.download_button("📥 Descargar CSV", trades.to_csv(index=False).encode('utf-8'), "ordenes.csv")

    with tabs[3]:
        if 'opt_res' not in st.session_state: st.info("Optimiza primero"); return
        res = st.session_state['opt_res']
        days = st.slider("Días", 30, 365, 90); sims = st.selectbox("Simulaciones", [100,500,1000], 1)
        if st.button("🔮 Simular"):
            dt = 1/252; mu = res['expected_return']*dt; sig = res['volatility']*np.sqrt(dt)
            paths = np.zeros((days,sims)); paths[0] = 100
            for t in range(1,days): paths[t] = paths[t-1] * np.exp((mu - 0.5*sig**2) + sig*np.random.standard_normal(sims))
            fig = go.Figure()
            p95,p50,p05 = np.percentile(paths,[95,50,5],axis=1)
            x = np.arange(days)
            fig.add_trace(go.Scatter(x=np.concatenate([x,x[::-1]]), y=np.concatenate([p95,p05[::-1]]), fill='toself', fillcolor='rgba(0,204,150,0.15)', line=dict(color='rgba(255,255,255,0)')))
            fig.add_trace(go.Scatter(x=x, y=p50, line=dict(color='#00CC96',width=2)))
            fig.add_trace(go.Scatter(x=x, y=p05, line=dict(color='#EF553B',dash='dash')))
            fig.add_trace(go.Scatter(x=x, y=p95, line=dict(color='#636EFA',dash='dash')))
            fig.update_layout(template="plotly_dark", height=400); st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mediana", f"${np.median(paths[-1]):.0f}"); c2.metric("P95", f"${np.percentile(paths[-1],95):.0f}")
            c3.metric("P05", f"${np.percentile(paths[-1],5):.0f}"); c4.metric("Prob. Pérdida", f"{np.mean(paths[-1]<100)*100:.1f}%")

def page_fixed_income():
    st.title("🏛️ Renta Fija")
    if 'bonds' not in st.session_state:
        st.session_state.bonds = pd.DataFrame({"Bono":["AL30","GD30","TX26"], "Cupón (%)":[2.5,3.0,1.5], "YTM (%)":[15,14,16], "Años":[5,8,2], "Nominal":[100000,150000,50000]})
    df = st.data_editor(st.session_state.bonds, num_rows="dynamic", use_container_width=True, key="bond_editor")
    st.session_state.bonds = df
    results = []; tot = 0
    for _, r in df.iterrows():
        try:
            c = r["Cupón (%)"]/100/2; y = r["YTM (%)"]/100/2; n = int(r["Años"]*2)
            price = sum([c/(1+y)**t for t in range(1, n+1)]) + 1/(1+y)**n
            macd = sum([(t/2)*c/(1+y)**t for t in range(1, n+1)]) / (price*100)
            modd = macd / (1+y)
            tot += r["Nominal"]
            results.append({"Bono": r["Bono"], "Precio": price*100, "MacDur": macd, "ModDur": modd, "Nominal": r["Nominal"]})
        except: pass
    if tot > 0 and results:
        dr = pd.DataFrame(results); dr["Peso"] = dr["Nominal"]/tot
        pmd = (dr["ModDur"]*dr["Peso"]).sum()
        c1,c2,c3 = st.columns(3)
        c1.metric("Duration Mod.", f"{pmd:.2f}"); c2.metric("Inversión Total", f"${tot:,.0f}")
        c3.slider("Horizonte (Años)", 1, 20, 5)
        st.dataframe(dr.style.format({"Precio":"{:.2f}", "MacDur":"{:.2f}", "ModDur":"{:.2f}", "Peso":"{:.1%}"}), use_container_width=True)
    else: st.info("Agrega bonos para ver análisis.")

def page_yahoo_explorer():
    st.title("🌎 Explorador Yahoo Finance")
    t = st.text_input("Ticker", "AAPL").upper()
    if not t: return
    try:
        s = yf.Ticker(t); h = s.history(period="1y")
        if h.empty: st.error("Sin datos"); return
        info = s.info
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Precio", f"${info.get('currentPrice', h['Close'].iloc[-1]):.2f}")
        c2.metric("Cap", f"${info.get('marketCap','N/A'):,}")
        c3.metric("Beta", info.get('beta','N/A'))
        c4.metric("Sector", info.get('sector','N/A'))
        fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
        fig.update_layout(template="plotly_dark", height=500); st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error: {e}")

def page_event_analyzer():
    st.header("📰 Analizador de Noticias")
    if not st.session_state.get("preferred_ai"): st.warning("Configura API Key"); return
    txt = st.text_area("Pega noticia o análisis", height=150)
    if st.button("Analizar"):
        if not txt.strip(): st.warning("Escribe algo"); return
        with st.spinner("Analizando..."):
            try:
                p = f"Analiza como experto financiero:\n{txt}\n1. Resumen 2. Impacto 3. Activos 4. Recomendación. Español."
                if st.session_state.preferred_ai == "OpenAI":
                    st.info(OpenAI(api_key=st.session_state.openai_api_key).chat.completions.create(model=st.session_state.openai_model, messages=[{"role":"user","content":p}]).choices[0].message.content)
                else:
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    st.info(genai.GenerativeModel(st.session_state.gemini_model).generate_content(p).text)
            except Exception as e: st.error(f"Error: {e}")

def page_chat_general():
    st.header("💬 Chat IA General")
    if not st.session_state.get("preferred_ai"): st.warning("Configura API Key"); return
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs: st.chat_message(m["role"]).write(m["content"])
    if p := st.chat_input("Consulta..."):
        st.session_state.msgs.append({"role":"user","content":p}); st.chat_message("user").write(p)
        try:
            if st.session_state.preferred_ai == "OpenAI":
                r = OpenAI(api_key=st.session_state.openai_api_key).chat.completions.create(model=st.session_state.openai_model, messages=[{"role":"user","content":p}]).choices[0].message.content
            else:
                genai.configure(api_key=st.session_state.gemini_api_key)
                r = genai.GenerativeModel(st.session_state.gemini_model).generate_content(p).text
            st.session_state.msgs.append({"role":"assistant","content":r}); st.chat_message("assistant").write(r)
        except Exception as e: st.error(f"Error: {e}")

@st.cache_data(ttl=1800)
def get_iol_tickers_cache():
    client = get_iol_client()
    if not client: return {}
    try:
        cats = {"Acciones":[], "CEDEARs":[], "Bonos":[], "ONs":[]}
        fallback = {"Acciones": ["GGAL","YPF","PAM","TX26","CEPU","ALUA","SUPV","MIRG","COME","VALO"],
                    "CEDEARs": ["AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","KO","MCD","V"],
                    "Bonos": ["AL30","GD30","TX26","AL35","GD35","TX29","AL30D","GD30D","TX26D"],
                    "ONs": ["YPFD","PAMP","GGAL","ALUA","TX26","CEPU","SUPV","MIRG"]}
        for c in cats:
            try:
                df = client.get_instruments(category=c.lower())
                if df is not None and not df.empty and "simbolo" in df.columns:
                    cats[c] = df["simbolo"].unique().tolist()[:20]
            except: pass
            if not cats[c]: cats[c] = fallback[c]
        return cats
    except: return {}

@st.cache_data(ttl=3600)
def get_yahoo_tickers_cache():
    return {
        "Tech Giants": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","CRM","ADBE"],
        "S&P 500 ETFs": ["SPY","VOO","IVV","SPLG","RSP"],
        "Nasdaq ETFs": ["QQQ","QQQM","TQQQ","PSQ","QQQJ"],
        "Bonds": ["AGG","BND","TLT","IEF","SHY","LQD","HYG","EMB","MUB","VCIT"],
        "Commodities": ["GLD","SLV","USO","DBA","PDBC","DBB","UGA"],
        "Emerging": ["EEM","VWO","IEMG","FXI","EWZ","INDA","RSX"],
        "Dividend": ["SCHD","VYM","DGRO","DVY","HDV","NOBL","SPYD"]
    }

def page_ai_strategy_assistant():
    st.header("🧠 Asistente Quant: Estrategia con Tickers Reales")
    if not st.session_state.get("preferred_ai"): st.warning("⚠️ Configura API Key en la barra lateral."); return
    
    tabs = st.tabs(["🎯 Generador IA", "🏦 Tickers IOL", "🌎 Tickers Yahoo", "🔥 Movers"])
    
    with tabs[0]:
        st.subheader("Diseña tu Portafolio con IA")
        with st.expander("📋 Tickers de Referencia"):
            c1,c2 = st.columns(2)
            iol = get_iol_tickers_cache(); yah = get_yahoo_tickers_cache()
            with c1:
                st.markdown("**🏦 IOL (Argentina)**")
                for k,v in iol.items():
                    if v: st.text(f"{k}: {len(v)} activos")
            with c2:
                st.markdown("**🌎 Yahoo Finance**")
                for k,v in yah.items():
                    st.text(f"{k}: {', '.join(v[:5])}...")
        
        strat = st.text_area("Describe tu estrategia:", height=120, placeholder="Ej: Portafolio conservador con bonos ARS y dividendos USA...")
        col1,col2,col3 = st.columns(3)
        risk = col1.select_slider("Riesgo", ["Conservador","Moderado","Agresivo","Muy Agresivo"], "Moderado")
        hor = col2.selectbox("Horizonte", ["Corto","Mediano","Largo"], "Mediano")
        mkt = col3.multiselect("Mercados", ["Argentina","USA","Internacional","Emergentes"], ["USA"])
        
        if st.button(" Generar Estrategia", type="primary"):
            if not strat.strip(): st.warning("Describe tu estrategia."); return
            iol = get_iol_tickers_cache(); yah = get_yahoo_tickers_cache()
            ctx = "\n📋 TICKERS DISPONIBLES REALES:\n"
            if iol: ctx += "\n=== IOL ===\n" + "\n".join(f"- {k}: {safe_join_list(v)}" for k,v in iol.items() if v)
            if yah: ctx += "\n=== YAHOO ===\n" + "\n".join(f"- {k}: {safe_join_list(v)}" for k,v in yah.items())
            
            sys_p = """Eres un gestor cuantitativo senior. Selecciona tickers REALES de las listas. Devuelve EXCLUSIVAMENTE JSON válido:
{
  "strategy_name": "string",
  "risk_profile": "string",
  "asset_allocation": {"stocks": float, "bonds": float, "etfs": float, "cash": float},
  "portfolios": {
    "argentina": [{"ticker":"string", "weight":float, "reason":"string"}],
    "usa": [{"ticker":"string", "weight":float, "reason":"string"}]
  },
  "filters": {"beta_range":[min,max], "pe_range":[min,max]},
  "expected_metrics": {"expected_return":"string", "volatility":"string", "sharpe_target":"string"},
  "rebalancing_frequency": "string",
  "notes": "string"
}
Pesos deben sumar 1.0. Usa SOLO tickers de las listas."""
            
            full = f"{sys_p}\n\nUsuario: {risk} | {hor} | {', '.join(mkt)}\nEstrategia: {strat}\n{ctx}"
            with st.spinner("🤖 IA analizando..."):
                try:
                    raw = ""
                    if st.session_state.preferred_ai == "OpenAI":
                        raw = OpenAI(api_key=st.session_state.openai_api_key).chat.completions.create(
                            model=st.session_state.openai_model, 
                            messages=[{"role":"system","content":"JSON ONLY. No markdown."},
                                      {"role":"user","content":full}], temperature=0.3).choices[0].message.content
                    else:
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        raw = genai.GenerativeModel(st.session_state.gemini_model).generate_content(
                            full, generation_config={"temperature":0.3}).text
                    
                    match = re.search(r'\{.*\}', raw, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        st.success("✅ Estrategia generada")
                        
                        rt1,rt2,rt3,rt4 = st.tabs(["📊 Resumen", "🥧 Asignación", "📋 Tickers", "💾 Exportar"])
                        with rt1:
                            c1,c2,c3 = st.columns(3)
                            c1.metric("Nombre", data.get("strategy_name","N/A"))
                            c2.metric("Riesgo", data.get("risk_profile","N/A"))
                            c3.metric("Rebalanceo", data.get("rebalancing_frequency","N/A"))
                            st.info(data.get("notes",""))
                        with rt2:
                            alloc = data.get("asset_allocation",{})
                            if alloc:
                                df = pd.DataFrame({"Clase": list(alloc.keys()), "%": [v*100 if isinstance(v,(int,float)) else 0 for v in alloc.values()]})
                                st.plotly_chart(px.pie(df, values="%", names="Clase", hole=0.4, template="plotly_dark"), use_container_width=True)
                        with rt3:
                            for mkt_name, assets in data.get("portfolios", {}).items():
                                if assets:
                                    st.markdown(f"**{mkt_name.upper()}**")
                                    st.dataframe(pd.DataFrame(assets).style.format({"weight":"{:.1%}"}), use_container_width=True)
                                    if st.button(f"📊 Usar en Optimizador ({mkt_name})", key=f"opt_{mkt_name}"):
                                        st.session_state['quant_pf'] = {
                                            "tickers": [a["ticker"] for a in assets], 
                                            "weights": [a["weight"] for a in assets], 
                                            "name": data.get("strategy_name","IA Strategy"),
                                            "market": mkt_name
                                        }
                                        st.success(f"✅ Guardado. Ve al Dashboard.")
                        with rt4:
                            st.download_button("📥 JSON", json.dumps(data, indent=4, ensure_ascii=False), f"strategy_{data.get('strategy_name','quant')}.json")
                            # FIX: Corrección de la condición incompleta
                            if "portfolios" in data:
                                rows = []
                                for m, a in data["portfolios"].items():
                                    for x in a: rows.append({"Mercado":m,"Ticker":x["ticker"],"Peso":x["weight"],"Razón":x["reason"]})
                                st.download_button("📥 CSV", pd.DataFrame(rows).to_csv(index=False).encode('utf-8'), f"tickers_{data.get('strategy_name','quant')}.csv")
                        st.session_state['last_quant'] = data
                    else: st.error("❌ JSON inválido"); st.code(raw)
                except json.JSONDecodeError as e: st.error(f"❌ Error JSON: {e}"); st.code(raw)
                except Exception as e: st.error(f"❌ Error: {type(e).__name__}: {e}")

    with tabs[1]:
        st.subheader("🏦 Tickers IOL")
        iol = get_iol_tickers_cache()
        if not iol: st.info("Conecta IOL o usa lista estática")
        for k,v in iol.items():
            if v:
                with st.expander(f"{k} ({len(v)})"):
                    cols = st.columns(5)
                    for i,t in enumerate(sorted(v)):
                        with cols[i%5]: st.code(t)

    with tabs[2]:
        st.subheader("🌎 Tickers Yahoo")
        yah = get_yahoo_tickers_cache()
        for k,v in yah.items():
            with st.expander(f"{k} ({len(v)})"):
                cols = st.columns(6)
                for i,t in enumerate(v):
                    with cols[i%6]: st.code(t)

    with tabs[3]:
        st.subheader("🔥 Movers (Referencia)")
        c1,c2,c3 = st.columns(3)
        c1.metric("Top Gainers", "NVDA, TSLA, META")
        c2.metric("Top Volume", "AAPL, SPY, AMD")
        c3.metric("IOL Top", "AL30, GGAL, YPF")
        if st.button("🔄 Refrescar"): st.cache_data.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR & ROUTER
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios()

st.sidebar.title("⚙️ Configuración")
if get_gsheets_client(): st.sidebar.success("🟢 Google Sheets")
else: st.sidebar.info("📁 Local")

# FIX: Inputs con keys únicos para evitar StreamlitDuplicateElementId
with st.sidebar.expander("🤖 IA (OpenAI)", expanded=True):
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('openai_api_key',''), key="sk_openai")
    st.session_state.openai_model = st.selectbox("Modelo OpenAI", ["gpt-4o","gpt-4o-mini"], index=0, key="sel_openai_model")

with st.sidebar.expander("🧠 IA (Gemini)", expanded=False):
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get('gemini_api_key',''), key="sk_gemini")
    st.session_state.gemini_model = st.selectbox("Modelo Gemini", ["gemini-2.0-flash","gemini-1.5-pro"], index=0, key="sel_gemini_model")

ais = []
if OPENAI_OK and st.session_state.get('openai_api_key'): ais.append("OpenAI")
if GEMINI_OK and st.session_state.get('gemini_api_key'): ais.append("Gemini")
if ais: st.session_state.preferred_ai = st.sidebar.radio("✨ Motor IA Activo", ais)
else: st.session_state.preferred_ai = None; st.sidebar.warning("⚠️ Ingresa API Key")

with st.sidebar.expander("🏦 Conexión IOL", expanded=True):
    u = st.text_input("Usuario IOL", value=st.session_state.get('iol_username',''))
    p = st.text_input("Contraseña IOL", type="password", value=st.session_state.get('iol_password',''))
    if st.button("Conectar", use_container_width=True):
        st.session_state.iol_username = u; st.session_state.iol_password = p
        try: st.session_state.iol_connected = get_iol_client() is not None
        except: st.session_state.iol_connected = False
    st.markdown("---")
    if st.session_state.get('iol_connected'): st.success(f"🟢 Conectado: {u}")
    else: st.info("🔴 Desconectado")

st.sidebar.markdown("---")
pages = ["Inicio", "📊 Dashboard Corporativo", "🏛️ Renta Fija", "🧠 Asistente Quant", 
         "🏦 Explorador IOL", "🌎 Yahoo Finance", "📰 Analizador Eventos", "💬 Chat IA General"]
idx = pages.index(st.session_state.selected_page) if st.session_state.selected_page in pages else 0
sel = st.sidebar.radio("Navegación", pages, index=idx)

if sel != st.session_state.selected_page: 
    st.session_state.selected_page = sel
    st.rerun()

if sel == "Inicio":
    st.title("📈 INVERSIONES PRO")
    st.markdown("### Plataforma Integral de Gestión de Portafolios\n🔹 Optimización Avanzada | 🔹 Rebalanceo Inteligente | 🔹 IA Integrada | 🔹 Tickers Reales IOL/Yahoo")
elif sel == "📊 Dashboard Corporativo": page_corporate_dashboard()
elif sel == "🏛️ Renta Fija": page_fixed_income()
elif sel == "🧠 Asistente Quant": page_ai_strategy_assistant()
elif sel == "🏦 Explorador IOL": page_iol_explorer()
elif sel == "🌎 Yahoo Finance": page_yahoo_explorer()
elif sel == "📰 Analizador Eventos": page_event_analyzer()
elif sel == "💬 Chat IA General": page_chat_general()
