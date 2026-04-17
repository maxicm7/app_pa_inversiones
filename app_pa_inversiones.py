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

# ── IMPORTACIÓN SEGURA DE GOOGLE SHEETS ──────────────────────────────────
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_OK = True
except ImportError:
    GSHEETS_OK = False
    st.warning("⚠️ Instala gspread y google-auth: pip install gspread google-auth")

# ── IMPORTACIÓN SEGURA DE GOOGLE GEMINI ──
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# ── IMPORTACIÓN SEGURA DE OPENAI (Copilot engine) ──
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

# ── Dependencia opcional para optimización institucional ──
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False
    st.warning("⚠️ Para optimización avanzada: pip install PyPortfolioOpt")

# ── Dependencias para procesamiento de documentos ──
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

# ── Módulos propios (Manejo de errores si no existen) ──
try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
    def page_forecast(): st.warning("📦 Módulo forecast_module.py no encontrado. Crea el archivo o instala las dependencias.")
    def page_iol_explorer(): st.warning("📦 Módulo iol_client.py no encontrado.")
    def get_iol_client(): return None

# ── Configuración Global ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="INVERSIONES PRO", page_icon="📈")

# ───────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE GOOGLE SHEETS
# ───────────────────────────────────────────────────────────────────────────
SHEET_NAME = st.secrets.get("google_sheets", {}).get("sheet_name", "Epre_Inversiones")
SHEET_ID   = st.secrets.get("google_sheets", {}).get("sheet_id", "")
WORKSHEET_NAME = "portfolios"
PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES: PROCESAMIENTO DE ARCHIVOS PARA IA
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_file(uploaded_file, max_chars: int = 15000) -> str:
    """
    Extrae texto de archivos PDF, DOCX, CSV o TXT con límite de caracteres.
    """
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
            content = content[:max_chars] + "\n\n[...contenido truncado por límite de tokens...]"
            
        return content.strip()
        
    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {type(e).__name__}: {e}")
        return ""


def truncate_for_tokens(text: str, max_tokens: int = 8000) -> str:
    """
    Trunca texto estimando ~4 caracteres por token (aproximación conservadora).
    """
    if len(text) <= max_tokens * 4:
        return text
    chunk = max_tokens * 2
    return text[:chunk] + "\n\n[...resumen intermedio omitido...]\n\n" + text[-chunk:]


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN MEJORADA: CONSTRUCCIÓN DE CONTEXTO PARA ANÁLISIS DE PORTAFOLIO
# ═══════════════════════════════════════════════════════════════════════════

def build_portfolio_context(res: dict, prices: pd.DataFrame = None, 
                           portfolio_name: str = "Portafolio",
                           include_correlations: bool = True) -> str:
    """
    Construye un prompt enriquecido con TODA la información relevante del portafolio.
    """
    lines = []
    
    lines.append(f"📊 ANÁLISIS DE PORTAFOLIO: {portfolio_name}")
    lines.append(f"Generado el: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Métricas globales
    lines.append("🎯 MÉTRICAS GLOBALES:")
    lines.append(f"- Retorno esperado anual: {res['expected_return']:.2%}")
    lines.append(f"- Volatilidad anualizada: {res['volatility']:.2%}")
    lines.append(f"- Ratio Sharpe (RF={0.02:.1%}): {res['sharpe_ratio']:.2f}")
    lines.append(f"- Método de optimización: {res.get('method', 'N/A')}")
    lines.append("")
    
    # Composición DETALLADA con pesos
    lines.append("🧩 COMPOSICIÓN DEL PORTAFOLIO (Pesos Reales):")
    active_assets = [(t, w) for t, w in zip(res['tickers'], res['weights']) if w > 0.001]
    active_assets.sort(key=lambda x: x[1], reverse=True)
    
    total_weight = sum(w for _, w in active_assets)
    for ticker, weight in active_assets:
        pct = weight / total_weight * 100 if total_weight > 0 else 0
        lines.append(f"- {ticker:<10} : {pct:5.1f}%  (peso: {weight:.4f})")
    
    if active_assets:
        top_weight = active_assets[0][1] / total_weight if total_weight > 0 else 0
        if top_weight > 0.4:
            lines.append(f"  ⚠️ ALERTA: Concentración alta en {active_assets[0][0]} ({top_weight:.1%})")
    lines.append("")
    
    # Métricas individuales por activo
    if prices is not None and not prices.empty and len(prices) >= 30:
        returns = prices.pct_change().dropna()
        lines.append("📈 MÉTRICAS INDIVIDUALES (histórico):")
        
        for ticker, weight in active_assets:
            if ticker in prices.columns and ticker in returns.columns:
                ann_ret = returns[ticker].mean() * 252
                ann_vol = returns[ticker].std() * np.sqrt(252)
                sharpe_ind = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
                neg_rets = returns[ticker][returns[ticker] < 0]
                downside = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 5 else ann_vol
                
                lines.append(f"- {ticker}:")
                lines.append(f"    • Retorno: {ann_ret:6.1%} | Vol: {ann_vol:5.1%} | Sharpe: {sharpe_ind:5.2f}")
                lines.append(f"    • Downside Vol: {downside:5.1%} | Peso: {weight/total_weight:.1%}")
        lines.append("")
    
    # Matriz de correlaciones
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
                lines.append("🔗 CORRELACIONES SIGNIFICATIVAS:")
                for asset1, asset2, corr_val in high_corr_pairs:
                    w1 = next((w for t, w in active_assets if t == asset1), 0)
                    w2 = next((w for t, w in active_assets if t == asset2), 0)
                    combined_weight = (w1 + w2) / total_weight if total_weight > 0 else 0
                    signal = "🔴" if corr_val > 0.8 else "🟡" if corr_val > 0.65 else "🟢"
                    lines.append(f"  {signal} {asset1} ↔ {asset2}: {corr_val:+.2f}  (peso combinado: {combined_weight:.1%})")
                lines.append("")
    
    # Exposición implícita
    lines.append("🌍 EXPOSICIÓN IMPLÍCITA (estimada):")
    exposures = {"ARS": 0, "USD": 0, "Equity": 0, "FixedIncome": 0, "Other": 0}
    
    for ticker, weight in active_assets:
        t_upper = ticker.upper()
        if any(x in t_upper for x in ["AL30", "GD30", "GGAL", "YPF", "PAM", "TX26", "CEPU", "AR"]) and ".BA" not in t_upper:
            exposures["ARS"] += weight
            exposures["Equity"] += weight if any(x in t_upper for x in ["GGAL", "YPF", "PAM", "CEPU"]) else 0
            exposures["FixedIncome"] += weight if any(x in t_upper for x in ["AL30", "GD30", "TX26"]) else 0
        elif "=X" in t_upper or any(x in t_upper for x in ["USD", "EUR", "BRL"]):
            exposures["USD"] += weight
        elif any(x in t_upper for x in ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ", ".US"]):
            exposures["USD"] += weight
            exposures["Equity"] += weight
        else:
            exposures["Other"] += weight
    
    for exp_type, exp_weight in exposures.items():
        if exp_weight > 0.01:
            pct = exp_weight / total_weight * 100 if total_weight > 0 else 0
            lines.append(f"- {exp_type}: {pct:.1f}%")
    
    if exposures["ARS"] / total_weight > 0.7 if total_weight > 0 else False:
        lines.append("  ⚠️ ALERTA: Alta exposición a Argentina")
    
    lines.append("")
    
    if prices is not None and not prices.empty:
        lines.append(f"📊 DATOS: {len(prices)} observaciones | {prices.index.min().date()} a {prices.index.max().date()}")
        missing_data = prices.isna().sum().sum()
        if missing_data > 0:
            lines.append(f"  ⚠️ {missing_data} valores faltantes imputados")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  CONEXIÓN A GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_gsheets_client():
    if not GSHEETS_OK:
        return None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Error Google Sheets: {e}")
        return None


def get_or_create_worksheet(client, sheet_name: str, worksheet_name: str):
    try:
        if SHEET_ID:
            spreadsheet = client.open_by_key(SHEET_ID)
        else:
            spreadsheet = client.open(sheet_name)
    except Exception as e:
        st.error(f"❌ No se pudo abrir el Sheet. Detalle: {e}")
        raise

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=200, cols=3)
        worksheet.append_row(["name", "tickers", "weights"])
    return worksheet


# ═══════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE PORTAFOLIOS — GOOGLE SHEETS + FALLBACK LOCAL
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_gsheet() -> dict:
    client = get_gsheets_client()
    if client is None:
        return _load_portfolios_local_fallback()
    try:
        ws = get_or_create_worksheet(client, SHEET_NAME, WORKSHEET_NAME)
        records = ws.get_all_records()
        portfolios = {}
        for row in records:
            name = str(row.get("name", "")).strip()
            raw_tickers = str(row.get("tickers", "")).strip()
            raw_weights = str(row.get("weights", "")).strip()
            if not name or not raw_tickers:
                continue
            tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
            try:
                weights = [float(w.strip()) for w in raw_weights.split(",") if w.strip()]
            except ValueError:
                weights = [1.0 / len(tickers)] * len(tickers)
            total_w = sum(weights)
            if total_w > 0 and abs(total_w - 1.0) > 0.01:
                weights = [w / total_w for w in weights]
            portfolios[name] = {"tickers": tickers, "weights": weights}
        return portfolios
    except Exception as e:
        st.error(f"Error al leer Google Sheets: {e}")
        return _load_portfolios_local_fallback()


def save_portfolios_to_gsheet(portfolios_dict: dict) -> tuple[bool, str]:
    client = get_gsheets_client()
    if client is None:
        return _save_portfolios_local_fallback(portfolios_dict)
    try:
        ws = get_or_create_worksheet(client, SHEET_NAME, WORKSHEET_NAME)
        ws.clear()
        rows = [["name", "tickers", "weights"]]
        for name, data in portfolios_dict.items():
            tickers_str = ", ".join(data["tickers"])
            weights_str = ", ".join(str(w) for w in data["weights"])
            rows.append([name, tickers_str, weights_str])
        ws.update(rows, "A1")
        _save_portfolios_local_fallback(portfolios_dict)
        return True, ""
    except Exception as e:
        st.error(f"Error al guardar en Google Sheets: {e}")
        return _save_portfolios_local_fallback(portfolios_dict)


def _load_portfolios_local_fallback() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error de lectura JSON local: {e}")
    return {}


def _save_portfolios_local_fallback(portfolios_dict: dict) -> tuple[bool, str]:
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        return False, str(e)


def load_portfolios_from_file() -> dict:
    return load_portfolios_from_gsheet()


def save_portfolios_to_file(portfolios_dict: dict) -> tuple[bool, str]:
    return save_portfolios_to_gsheet(portfolios_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO: DESCARGA Y OPTIMIZACIÓN BASE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    client = get_iol_client()
    all_prices = {}
    yf_tickers = []

    for ticker in tickers:
        fetched = False
        if client:
            simbolo_iol = ticker.split(".")[0].upper()
            fmt_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            fmt_end   = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            try:
                df_hist = client.get_serie_historica(simbolo_iol, fmt_start, fmt_end)
                if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                    s = df_hist["ultimoPrecio"].rename(ticker)
                    if s.index.tz is not None: s.index = s.index.tz_localize(None)
                    all_prices[ticker] = s
                    fetched = True
            except:
                pass
        if not fetched:
            yf_tickers.append(ticker)

    if yf_tickers:
        try:
            adjusted_tickers = [t if "." in t or t.endswith("=X") else t+".BA" for t in yf_tickers]
            raw = yf.download(adjusted_tickers, start=start_date, end=end_date, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    if 'Close' in raw.columns.levels[0]:
                        close_data = raw['Close']
                    elif 'Adj Close' in raw.columns.levels[0]:
                        close_data = raw['Adj Close']
                    else:
                        close_data = raw.iloc[:, 0:len(adjusted_tickers)]
                else:
                    close_data = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close_data, pd.Series):
                    close_data = close_data.to_frame(name=yf_tickers[0])
                for col in close_data.columns:
                    clean_col = str(col).replace(".BA", "")
                    for original in yf_tickers:
                        if clean_col == original or str(col) == original:
                            all_prices[original] = close_data[col]
                            break
        except Exception as e:
            st.warning(f"Yahoo Finance warning: {e}")

    if not all_prices: return None
    prices = pd.concat(all_prices.values(), axis=1)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)
    prices.sort_index(inplace=True)
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)
    return prices


def optimize_portfolio_corporate(prices, risk_free_rate=0.02, opt_type="Maximo Ratio Sharpe"):
    """Optimización Mean-Variance tradicional (Markowitz)"""
    returns = prices.pct_change().dropna()
    if returns.empty or len(returns) < 2: return None

    if PYPFOPT_OK:
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.sample_cov(prices, frequency=252)
            if not mu.isnull().any() and not S.isnull().values.any():
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                if opt_type == "Maximo Ratio Sharpe":
                    ef.max_sharpe(risk_free_rate=risk_free_rate)
                elif opt_type == "Minima Volatilidad":
                    ef.min_volatility()
                else:
                    ef.max_quadratic_utility(risk_aversion=0.01)
                weights = ef.clean_weights()
                ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                ow_array = np.array([weights.get(col, 0) for col in prices.columns])
                return {"weights": ow_array, "expected_return": float(ret), "volatility": float(vol),
                        "sharpe_ratio": float(sharpe), "tickers": list(prices.columns),
                        "returns": returns, "method": "PyPortfolioOpt-Markowitz"}
        except Exception:
            pass

    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    n = len(mean_returns)

    def get_metrics(w):
        ret = np.sum(mean_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0
        return np.array([ret, vol, sr])

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.array([1/n] * n)

    if opt_type == "Minima Volatilidad":
        fun = lambda w: get_metrics(w)[1]
    elif opt_type == "Retorno Maximo":
        fun = lambda w: -get_metrics(w)[0]
    else:
        if (mean_returns < risk_free_rate).all():
            fun = lambda w: get_metrics(w)[1]
        else:
            fun = lambda w: -get_metrics(w)[2]

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    final_weights = res.x if res.success else init
    final_weights = np.maximum(final_weights, 0)
    if np.sum(final_weights) > 0:
        final_weights = final_weights / np.sum(final_weights)

    final_metrics = get_metrics(final_weights)
    return {"weights": final_weights, "expected_return": float(final_metrics[0]),
            "volatility": float(final_metrics[1]), "sharpe_ratio": float(final_metrics[2]),
            "tickers": list(prices.columns), "returns": returns, "method": "Scipy/SLSQP"}


# ═══════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN AVANZADA: RISK PARITY, BLACK-LITTERMAN, HRP
# ═══════════════════════════════════════════════════════════════════════════

def optimize_risk_parity(prices, risk_free_rate=0.02):
    """Risk Parity: Equal risk contribution from each asset"""
    if not PYPFOPT_OK:
        st.warning("⚠️ Instala PyPortfolioOpt para Risk Parity: pip install PyPortfolioOpt")
        return None
        
    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return None
    
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        
        rp_portfolio = EfficientFrontier(
            expected_returns=None,
            cov_matrix=S,
            weight_bounds=(0, 1)
        )
        weights = rp_portfolio.risk_parity()
        cleaned_weights = rp_portfolio.clean_weights()
        
        w_array = np.array([cleaned_weights.get(col, 0) for col in prices.columns])
        
        mean_returns = returns.mean() * 252
        portfolio_return = np.sum(mean_returns * w_array)
        portfolio_vol = np.sqrt(np.dot(w_array.T, np.dot(S, w_array)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            "weights": w_array,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe),
            "tickers": list(prices.columns),
            "method": "Risk Parity (Equal Risk Contribution)",
            "risk_contributions": cleaned_weights
        }
    except Exception as e:
        st.warning(f"Risk Parity optimization failed: {e}")
        return None


def optimize_black_litterman(prices, market_caps=None, views=None, 
                             view_confidences=None, risk_free_rate=0.02):
    """Black-Litterman: Combina equilibrio de mercado con views del inversor"""
    if not PYPFOPT_OK:
        st.warning("⚠️ Instala PyPortfolioOpt para Black-Litterman")
        return None
        
    try:
        returns = prices.pct_change().dropna()
        if len(returns) < 30:
            return None
        
        if market_caps is None:
            market_caps = {col: 1.0 for col in prices.columns}
        
        S = risk_models.sample_cov(prices, frequency=252)
        
        bl_portfolio = EfficientFrontier(
            expected_returns=None,
            cov_matrix=S,
            weight_bounds=(0, 1)
        )
        
        if views and view_confidences:
            view_tickers = list(views.keys())
            view_values = list(views.values())
            confidences = [view_confidences.get(t, 0.5) for t in view_tickers]
            
            P = np.zeros((len(views), len(prices.columns)))
            for i, ticker in enumerate(view_tickers):
                if ticker in prices.columns:
                    P[i, list(prices.columns).index(ticker)] = 1
            
            bl_portfolio.black_litterman(
                market_caps=market_caps,
                P=P,
                Q=view_values,
                view_confidences=confidences,
                risk_free_rate=risk_free_rate
            )
        else:
            bl_portfolio.black_litterman(market_caps=market_caps)
        
        weights = bl_portfolio.clean_weights()
        w_array = np.array([weights.get(col, 0) for col in prices.columns])
        
        mean_returns = returns.mean() * 252
        port_return = np.sum(mean_returns * w_array)
        port_vol = np.sqrt(np.dot(w_array.T, np.dot(S, w_array)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return {
            "weights": w_array,
            "expected_return": float(port_return),
            "volatility": float(port_vol),
            "sharpe_ratio": float(sharpe),
            "tickers": list(prices.columns),
            "method": "Black-Litterman"
        }
    except Exception as e:
        st.warning(f"Black-Litterman optimization failed: {e}")
        return None


def optimize_hierarchical_risk_parity(prices, risk_free_rate=0.02):
    """Hierarchical Risk Parity (HRP): Usa clustering para diversificación"""
    if not PYPFOPT_OK:
        st.warning("⚠️ Instala PyPortfolioOpt para HRP")
        return None
        
    try:
        returns = prices.pct_change().dropna()
        if len(returns) < 30:
            return None
        
        S = risk_models.sample_cov(prices, frequency=252)
        
        hrp = HRPOpt(cov_matrix=S, returns=returns)
        weights = hrp.optimize()
        
        w_array = np.array([weights.get(col, 0) for col in prices.columns])
        
        mean_returns = returns.mean() * 252
        port_return = np.sum(mean_returns * w_array)
        port_vol = np.sqrt(np.dot(w_array.T, np.dot(S, w_array)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return {
            "weights": w_array,
            "expected_return": float(port_return),
            "volatility": float(port_vol),
            "sharpe_ratio": float(sharpe),
            "tickers": list(prices.columns),
            "method": "Hierarchical Risk Parity (HRP)"
        }
    except Exception as e:
        st.warning(f"HRP optimization failed: {e}")
        return None


def optimize_maximum_diversification(prices, risk_free_rate=0.02):
    """Maximum Diversification Ratio"""
    if not PYPFOPT_OK:
        st.warning("⚠️ Instala PyPortfolioOpt para Max Diversification")
        return None
        
    try:
        returns = prices.pct_change().dropna()
        if len(returns) < 30:
            return None
        
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        S = risk_models.sample_cov(prices, frequency=252)
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.max_quadratic_utility(risk_aversion=0.01)
        
        weights = ef.clean_weights()
        w_array = np.array([weights.get(col, 0) for col in prices.columns])
        
        ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
        return {
            "weights": w_array,
            "expected_return": float(ret),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "tickers": list(prices.columns),
            "method": "Maximum Diversification"
        }
    except Exception as e:
        st.warning(f"Max Diversification failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  MOTOR DE REBALANCEO DE PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_rebalancing_trades(current_portfolio, target_weights, 
                                  current_prices, total_value, 
                                  min_trade_size=100):
    """Calcula las operaciones necesarias para rebalancear"""
    trades = []
    
    for ticker, target_weight in target_weights.items():
        current_value = current_portfolio.get(ticker, 0)
        target_value = total_value * target_weight
        difference = target_value - current_value
        
        if abs(difference) >= min_trade_size:
            current_price = current_prices.get(ticker, 1)
            shares_to_trade = difference / current_price if current_price > 0 else 0
            
            trades.append({
                "Activo": ticker,
                "Valor Actual ($)": f"{current_value:,.0f}",
                "Valor Target ($)": f"{target_value:,.0f}",
                "Diferencia ($)": f"{difference:,.0f}",
                "Precio Actual": f"${current_price:,.2f}",
                "Cantidad": round(abs(shares_to_trade), 4),
                "Acción": "🟢 COMPRAR" if difference > 0 else "🔴 VENDER"
            })
    
    return pd.DataFrame(trades)


def get_rebalancing_strategy(current_weights, target_weights, drift_threshold=0.05):
    """Determina si es necesario rebalancear basado en drift"""
    max_drift = 0
    for ticker in target_weights:
        current = current_weights.get(ticker, 0)
        target = target_weights.get(ticker, 0)
        drift = abs(current - target)
        max_drift = max(max_drift, drift)
    
    needs_rebalancing = max_drift > drift_threshold
    
    return {
        "needs_rebalancing": needs_rebalancing,
        "max_drift": max_drift,
        "threshold": drift_threshold,
        "status": "⚠️ REQUERIDO" if needs_rebalancing else "✅ OK"
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ANÁLISIS AVANZADO DE PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_portfolio_var(portfolio_returns, confidence_level=0.95, holding_period=1):
    """Calcula Value at Risk (VaR) histórico y paramétrico"""
    var_hist = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    var_param = norm.ppf(1 - confidence_level, mu, sigma)
    cvar = portfolio_returns[portfolio_returns <= var_hist].mean() if len(portfolio_returns[portfolio_returns <= var_hist]) > 0 else var_hist
    
    return {
        "var_historical": var_hist * holding_period,
        "var_parametric": var_param * holding_period,
        "cvar": cvar * holding_period,
        "confidence": confidence_level
    }


def calculate_portfolio_metrics(prices, weights, risk_free_rate=0.02):
    """Calcula métricas avanzadas de riesgo y rendimiento"""
    returns = prices.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    
    total_return = (1 + portfolio_returns).prod() - 1
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    cum_returns = (1 + portfolio_returns).cumprod()
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 and not np.isinf(max_dd) else 0
    
    pos_returns = portfolio_returns[portfolio_returns > 0]
    neg_returns = portfolio_returns[portfolio_returns < 0]
    omega = (pos_returns.sum() / abs(neg_returns.sum()) if len(neg_returns) > 0 else np.inf)
    
    var_metrics = calculate_portfolio_var(portfolio_returns)
    
    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        "max_drawdown": max_dd,
        "var_95": var_metrics["var_historical"],
        "cvar_95": var_metrics["cvar"],
        "skewness": portfolio_returns.skew(),
        "kurtosis": portfolio_returns.kurtosis(),
        "daily_returns": portfolio_returns
    }


def analyze_factor_exposure(prices, benchmark_ticker="SPY"):
    """Analiza exposición a factores (beta, momentum, value, etc.)"""
    try:
        benchmark = yf.download(benchmark_ticker, start=prices.index.min(), 
                               end=prices.index.max(), progress=False)
        if benchmark.empty:
            return {}
        benchmark_returns = benchmark['Close'].pct_change().dropna()
        
        factor_analysis = {}
        
        for ticker in prices.columns:
            asset_returns = prices[ticker].pct_change().dropna()
            common_idx = asset_returns.index.intersection(benchmark_returns.index)
            
            if len(common_idx) < 30:
                continue
                
            asset_ret = asset_returns.loc[common_idx]
            bench_ret = benchmark_returns.loc[common_idx]
            
            covariance = np.cov(asset_ret, bench_ret)[0, 1]
            variance = np.var(bench_ret)
            beta = covariance / variance if variance > 0 else 1
            
            alpha = asset_ret.mean() * 252 - beta * (bench_ret.mean() * 252 - 0.02)
            
            correlation = np.corrcoef(asset_ret, bench_ret)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            momentum = (prices[ticker].iloc[-1] / prices[ticker].iloc[-252] - 1) if len(prices) > 252 else 0
            
            factor_analysis[ticker] = {
                "beta": round(beta, 3),
                "alpha": round(alpha, 4),
                "r_squared": round(r_squared, 3),
                "correlation": round(correlation, 3) if not np.isnan(correlation) else 0,
                "momentum_12m": round(momentum, 4)
            }
        
        return factor_analysis
    except Exception as e:
        st.warning(f"Factor analysis failed: {e}")
        return {}


def display_advanced_metrics(prices, res):
    """Muestra métricas avanzadas del portafolio optimizado"""
    st.subheader("📊 Métricas Avanzadas de Riesgo")
    
    weights_dict = {t: w for t, w in zip(res['tickers'], res['weights'])}
    metrics = calculate_portfolio_metrics(prices, weights_dict)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{metrics['sharpe']:.3f}")
    c2.metric("Sortino Ratio", f"{metrics['sortino']:.3f}")
    c3.metric("Calmar Ratio", f"{metrics['calmar']:.3f}")
    c4.metric("Omega Ratio", f"{metrics['omega']:.2f}" if metrics['omega'] != np.inf else "∞")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    c6.metric("VaR 95% (1 día)", f"{metrics['var_95']:.2%}")
    c7.metric("CVaR 95%", f"{metrics['cvar_95']:.2%}")
    c8.metric("Skewness", f"{metrics['skewness']:.3f}")
    
    # Gráfico de distribución de retornos
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=metrics['daily_returns'].dropna(), 
                               name='Distribución', 
                               nbinsx=50,
                               marker_color='rgba(0,204,150,0.7)'))
    fig.add_vline(x=metrics['var_95'], line_dash="dash", 
                  line_color="red", annotation_text="VaR 95%")
    fig.update_layout(title="📈 Distribución de Retornos Diarios", 
                     template="plotly_dark", 
                     xaxis_title="Retorno",
                     yaxis_title="Frecuencia")
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor exposure
    st.subheader("🎯 Exposición a Factores vs S&P 500")
    factor_exp = analyze_factor_exposure(prices)
    
    if factor_exp:
        factor_df = pd.DataFrame([{
            "Activo": ticker,
            "Beta": data["beta"],
            "Alpha": f"{data['alpha']:.2%}",
            "R²": f"{data['r_squared']:.2f}",
            "Momentum 12M": f"{data['momentum_12m']:.1%}"
        } for ticker, data in factor_exp.items()])
        
        st.dataframe(factor_df, use_container_width=True)
        
        # Gráfico de Beta
        fig_beta = px.bar(factor_df, x="Activo", y="Beta", 
                         title="📊 Beta por Activo (vs S&P 500)",
                         color="Beta", color_continuous_scale="RdYlGn_r")
        fig_beta.add_hline(y=1, line_dash="dash", line_color="white", annotation_text="Beta=1 (Mercado)")
        st.plotly_chart(fig_beta, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ESTADO DEL SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_gsheets_status():
    client = get_gsheets_client()
    if client:
        st.sidebar.success(f"🟢 Google Sheets: {SHEET_NAME}")
    else:
        st.sidebar.info("📁 Google Sheets: usando almacenamiento local")


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA PRINCIPAL: DASHBOARD CORPORATIVO MEJORADO
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo Integral")
    
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("👈 Crea tu primer portafolio en la pestaña 'Gestión de Carteras'")
    
    tabs = st.tabs([
        "💼 Gestión de Portafolios", 
        "🚀 Optimización Avanzada", 
        "🔄 Rebalanceo Inteligente",
        "🔮 Forecast & Simulación"
    ])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1: GESTIÓN DE PORTAFOLIOS
    # ─────────────────────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("Administrar Carteras")
            action = st.radio("Acción:", ["✨ Crear Nueva", "✏️ Editar / 🗑️ Eliminar"], horizontal=True)

            if action == "✨ Crear Nueva":
                p_name = st.text_input("Nombre de la Cartera", key="new_portfolio_name")
                p_tickers = st.text_area("Tickers (separados por coma)", "AL30, GGAL, AAPL, SPY", key="new_tickers").upper()
                p_weights = st.text_area("Pesos (separados por coma, deben sumar 1.0)", "0.25, 0.25, 0.25, 0.25", key="new_weights")

                if st.button("💾 Guardar Nueva Cartera", type="primary", key="save_new"):
                    try:
                        t = [x.strip() for x in p_tickers.split(",") if x.strip()]
                        w = [float(x) for x in p_weights.split(",") if x.strip()]
                        if not p_name:
                            st.error("El nombre no puede estar vacío.")
                        elif len(t) == len(w):
                            total_w = sum(w)
                            if abs(total_w - 1.0) > 0.02:
                                st.warning(f"⚠️ Los pesos suman {total_w:.3f}. Normalizando a 1.0")
                                w = [weight/total_w for weight in w]
                            
                            st.session_state.portfolios[p_name] = {"tickers": t, "weights": w}
                            ok, err = save_portfolios_to_file(st.session_state.portfolios)
                            if ok:
                                st.success("✅ Cartera guardada exitosamente")
                            else:
                                st.error(f"Error al guardar: {err}")
                            st.rerun()
                        else:
                            st.error(f"Error: {len(t)} tickers pero {len(w)} pesos")
                    except ValueError as e:
                        st.error(f"Error de formato en pesos: {e}")

            else:
                if st.session_state.portfolios:
                    edit_sel = st.selectbox("Seleccionar Cartera:", list(st.session_state.portfolios.keys()), key="edit_select")
                    curr_data = st.session_state.portfolios[edit_sel]
                    new_name = st.text_input("Renombrar Cartera", value=edit_sel, key="rename")
                    new_tickers = st.text_area("Modificar Tickers", value=", ".join(curr_data["tickers"]), key="edit_tickers").upper()
                    new_weights = st.text_area("Modificar Pesos", value=", ".join(f"{w:.4f}" for w in curr_data["weights"]), key="edit_weights")

                    col_b1, col_b2 = st.columns(2)
                    if col_b1.button("🔄 Actualizar", type="primary", use_container_width=True, key="update_btn"):
                        try:
                            t = [x.strip() for x in new_tickers.split(",") if x.strip()]
                            w = [float(x) for x in new_weights.split(",") if x.strip()]
                            if len(t) == len(w):
                                total_w = sum(w)
                                if abs(total_w - 1.0) > 0.02:
                                    w = [weight/total_w for weight in w]
                                if new_name != edit_sel:
                                    del st.session_state.portfolios[edit_sel]
                                st.session_state.portfolios[new_name] = {"tickers": t, "weights": w}
                                save_portfolios_to_file(st.session_state.portfolios)
                                st.success("✅ Cartera actualizada")
                                st.rerun()
                            else:
                                st.error("La cantidad de tickers y pesos no coincide")
                        except Exception as e:
                            st.error(f"Error: {e}")

                    if col_b2.button("🗑️ Eliminar", type="secondary", use_container_width=True, key="delete_btn"):
                        del st.session_state.portfolios[edit_sel]
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.warning("🗑️ Cartera eliminada")
                        st.rerun()
                else:
                    st.info("No hay carteras guardadas aún")

        with c2:
            st.subheader("📋 Base de Datos de Portafolios")
            if st.session_state.portfolios:
                df_ports = pd.DataFrame([
                    {
                        "Nombre": k, 
                        "Activos": len(v["tickers"]),
                        "Tickers": ", ".join(v["tickers"]),
                        "Peso Mayor": f"{max(v['weights'])*100:.1f}%",
                        "Diversificación": f"{1 - max(v['weights']):.1%}"
                    }
                    for k, v in st.session_state.portfolios.items()
                ])
                st.dataframe(df_ports, use_container_width=True, hide_index=True)
            else:
                st.info("💡 Crea tu primera cartera para comenzar")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2: OPTIMIZACIÓN AVANZADA
    # ─────────────────────────────────────────────────────────────────────
    with tabs[1]:
        if not portfolios:
            st.info("👈 Primero crea un portafolio en la pestaña anterior")
            return
            
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        p_sel = col1.selectbox("📦 Analizar Cartera:", list(portfolios.keys()), key="opt_portfolio")
        d_start = col2.date_input("📅 Desde", pd.to_datetime("2023-01-01"), key="opt_start")
        d_end = col3.date_input("📅 Hasta", pd.to_datetime("today"), key="opt_end")

        # ── Rendimiento Histórico ──
        st.subheader("📈 Rendimiento Histórico del Portafolio")
        if st.button("📊 Ver Rendimiento Histórico", key="show_perf"):
            with st.spinner("Descargando datos históricos..."):
                prices_perf = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)

            if prices_perf is not None:
                current_weights = list(portfolios[p_sel]["weights"])
                tickers_in_prices = [t for t in portfolios[p_sel]["tickers"] if t in prices_perf.columns]
                
                if not tickers_in_prices:
                    st.error("⚠️ No se encontraron datos de precios para ningún ticker")
                    st.stop()
                    
                if len(tickers_in_prices) < len(portfolios[p_sel]["tickers"]):
                    missing = set(portfolios[p_sel]["tickers"]) - set(tickers_in_prices)
                    st.warning(f"⚠️ Sin datos para: {', '.join(missing)}")
                    idx_valid = [portfolios[p_sel]["tickers"].index(t) for t in tickers_in_prices]
                    raw_w = [current_weights[i] for i in idx_valid]
                    total_w = sum(raw_w)
                    current_weights = [w / total_w for w in raw_w] if total_w > 0 else [1.0/len(tickers_in_prices)]*len(tickers_in_prices)

                prices_filtered = prices_perf[tickers_in_prices]
                norm_prices = prices_filtered / prices_filtered.iloc[0]
                weights_arr = np.array(current_weights[:len(tickers_in_prices)])
                portfolio_value = (norm_prices * weights_arr).sum(axis=1) * 100

                total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
                daily_rets = portfolio_value.pct_change().dropna()
                ann_vol = daily_rets.std() * np.sqrt(252) * 100
                max_dd = ((portfolio_value / portfolio_value.cummax()) - 1).min() * 100
                sharpe_hist = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Retorno Total", f"{total_return:.1f}%")
                m2.metric("Volatilidad Anual", f"{ann_vol:.1f}%")
                m3.metric("Máximo Drawdown", f"{max_dd:.1f}%")
                m4.metric("Sharpe Histórico", f"{sharpe_hist:.2f}")

                # Gráfico de evolución
                fig_perf = go.Figure()
                colors_ind = px.colors.qualitative.Pastel
                for i, ticker in enumerate(tickers_in_prices):
                    fig_perf.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[ticker] * 100,
                        mode='lines', name=ticker, line=dict(width=1.5, dash='dot', color=colors_ind[i % len(colors_ind)]), opacity=0.5))
                fig_perf.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines',
                    name=f"📂 {p_sel}", line=dict(width=3, color='#00CC96'), fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)'))
                fig_perf.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)
                fig_perf.update_layout(title=f"📈 Evolución – {p_sel} (Base 100)", template="plotly_dark", height=400, hovermode="x unified")
                st.plotly_chart(fig_perf, use_container_width=True)

                # Drawdown
                drawdown_series = (portfolio_value / portfolio_value.cummax() - 1) * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, mode='lines',
                    fill='tozeroy', fillcolor='rgba(239,85,59,0.2)', line=dict(color='#EF553B', width=1.5)))
                fig_dd.update_layout(title="📉 Drawdown del Portafolio (%)", template="plotly_dark", height=200)
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.error("❌ No se pudieron obtener datos de precios")

        # ── Optimización Avanzada ──
        st.markdown("---")
        st.subheader("⚙️ Optimización Avanzada de Portafolio")
        
        opt_tabs = st.tabs([
            "📊 Mean-Variance (Markowitz)",
            "⚖️ Risk Parity", 
            "🎯 Black-Litterman",
            "🌳 Hierarchical Risk Parity",
            "🔄 Comparar Métodos"
        ])

        with opt_tabs[0]:  # Markowitz
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            risk_free = c_opt1.number_input("Tasa Libre Riesgo (RF)", 0.0, 0.5, 0.04, step=0.01, key="rf_mv")
            target = c_opt2.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"], key="target_mv")
            
            if c_opt3.button("🚀 Optimizar Markowitz", type="primary", key="run_mv"):
                with st.spinner("Optimizando con Mean-Variance..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                if prices is not None:
                    res = optimize_portfolio_corporate(prices, risk_free_rate=risk_free, opt_type=target)
                    if res:
                        st.session_state['last_opt_res'] = res
                        st.session_state['last_opt_prices'] = prices
                        st.session_state['last_opt_portfolio_name'] = p_sel
                        st.success(f"✅ Optimizado: {res['method']}")
                        st.rerun()
                    else:
                        st.error("❌ Error en optimización")

        with opt_tabs[1]:  # Risk Parity
            st.info("⚖️ **Risk Parity**: Distribuye el riesgo equitativamente entre activos. Ideal para diversificación verdadera sin depender de estimaciones de retorno.")
            if st.button("🚀 Optimizar Risk Parity", type="primary", key="run_rp"):
                with st.spinner("Calculando Risk Parity..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                if prices is not None:
                    res = optimize_risk_parity(prices, risk_free_rate=0.04)
                    if res:
                        st.session_state['last_opt_res'] = res
                        st.session_state['last_opt_prices'] = prices
                        st.session_state['last_opt_portfolio_name'] = p_sel
                        st.success(f"✅ Optimizado: {res['method']}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Risk Parity no disponible (instala PyPortfolioOpt)")

        with opt_tabs[2]:  # Black-Litterman
            st.info("🎯 **Black-Litterman**: Combina equilibrio de mercado con tus views personales sobre activos específicos.")
            
            with st.expander("📝 Configurar Views de Mercado (Opcional)"):
                view_tickers = st.multiselect("Activos con views", portfolios[p_sel]["tickers"], key="bl_tickers")
                views, confidences = {}, {}
                for ticker in view_tickers:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        views[ticker] = col_a.number_input(f"Retorno esperado {ticker}", -0.5, 0.5, 0.1, step=0.01, key=f"view_{ticker}")
                    with col_b:
                        confidences[ticker] = col_b.slider(f"Confianza {ticker}", 0.0, 1.0, 0.5, step=0.1, key=f"conf_{ticker}")
            
            if st.button("🚀 Optimizar Black-Litterman", type="primary", key="run_bl"):
                with st.spinner("Calculando Black-Litterman..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                if prices is not None:
                    market_caps = {t: 1.0 for t in portfolios[p_sel]["tickers"]}
                    res = optimize_black_litterman(prices, market_caps=market_caps, 
                                                  views=views if views else None,
                                                  view_confidences=confidences if confidences else None)
                    if res:
                        st.session_state['last_opt_res'] = res
                        st.session_state['last_opt_prices'] = prices
                        st.session_state['last_opt_portfolio_name'] = p_sel
                        st.success(f"✅ Optimizado: {res['method']}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Black-Litterman no disponible")

        with opt_tabs[3]:  # HRP
            st.info("🌳 **Hierarchical Risk Parity**: Usa machine learning (clustering) para agrupar activos similares y diversificar mejor ante correlaciones inestables.")
            if st.button("🚀 Optimizar HRP", type="primary", key="run_hrp"):
                with st.spinner("Calculando Hierarchical Risk Parity..."):
                    prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                if prices is not None:
                    res = optimize_hierarchical_risk_parity(prices)
                    if res:
                        st.session_state['last_opt_res'] = res
                        st.session_state['last_opt_prices'] = prices
                        st.session_state['last_opt_portfolio_name'] = p_sel
                        st.success(f"✅ Optimizado: {res['method']}")
                        st.rerun()
                    else:
                        st.warning("⚠️ HRP no disponible")

        with opt_tabs[4]:  # Comparar
            st.subheader("📊 Comparación de Métodos de Optimización")
            if st.button("🔄 Comparar Todos los Métodos", key="compare_all"):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                if prices is not None:
                    methods_results = {}
                    
                    with st.spinner("Ejecutando optimizaciones..."):
                        # Markowitz
                        res_mv = optimize_portfolio_corporate(prices, opt_type="Maximo Ratio Sharpe")
                        if res_mv: methods_results["Markowitz"] = res_mv
                        
                        # Risk Parity
                        res_rp = optimize_risk_parity(prices)
                        if res_rp: methods_results["Risk Parity"] = res_rp
                        
                        # HRP
                        res_hrp = optimize_hierarchical_risk_parity(prices)
                        if res_hrp: methods_results["HRP"] = res_hrp
                    
                    if methods_results:
                        comparison_df = pd.DataFrame({
                            "Método": list(methods_results.keys()),
                            "Retorno Esperado": [r['expected_return']*100 for r in methods_results.values()],
                            "Volatilidad": [r['volatility']*100 for r in methods_results.values()],
                            "Sharpe Ratio": [r['sharpe_ratio'] for r in methods_results.values()]
                        })
                        
                        st.dataframe(comparison_df.style.format({
                            "Retorno Esperado": "{:.2f}%",
                            "Volatilidad": "{:.2f}%",
                            "Sharpe Ratio": "{:.3f}"
                        }), use_container_width=True)
                        
                        # Gráfico comparativo
                        fig = px.bar(comparison_df, x="Método", y="Sharpe Ratio",
                                    color="Sharpe Ratio", color_continuous_scale="Viridis",
                                    title="🏆 Comparación de Sharpe Ratio por Método")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recomendación
                        best_method = comparison_df.loc[comparison_df["Sharpe Ratio"].idxmax(), "Método"]
                        st.success(f"🎯 **Recomendación**: {best_method} ofrece el mejor ratio riesgo/retorno para este portafolio")

        # ── Display de resultados optimizados ──
        if 'last_opt_res' in st.session_state and st.session_state.get('last_opt_portfolio_name') == p_sel:
            res = st.session_state['last_opt_res']
            prices_ctx = st.session_state.get('last_opt_prices')
            
            # KPIs
            c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
            c_kpi1.metric("📈 Retorno Esperado", f"{res['expected_return']:.1%}")
            c_kpi2.metric("📊 Volatilidad Anual", f"{res['volatility']:.1%}")
            c_kpi3.metric("⭐ Ratio Sharpe", f"{res['sharpe_ratio']:.2f}")
            
            # Gráfico de asignación
            w_df = pd.DataFrame({"Activo": res['tickers'], "Peso": res['weights']})
            df_pie = w_df[w_df["Peso"] > 0.001]
            if not df_pie.empty:
                fig = px.pie(df_pie, values="Peso", names="Activo", title="🥧 Asignación Optimizada", hole=0.4, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Métricas avanzadas
            if prices_ctx is not None:
                display_advanced_metrics(prices_ctx, res)
            
            # Análisis con IA
            st.markdown("---")
            if st.button("🧠 Analizar Portafolio con IA", key="ai_analyze"):
                if not st.session_state.get("preferred_ai"):
                    st.warning("⚠️ Ingresa tu API Key en el menú lateral")
                else:
                    with st.spinner(f"Consultando IA ({st.session_state.preferred_ai})..."):
                        try:
                            context = build_portfolio_context(
                                res=res, prices=prices_ctx, 
                                portfolio_name=st.session_state.get('last_opt_portfolio_name', 'Portafolio'),
                                include_correlations=True
                            )
                            
                            prompt = f"""Actúa como asesor financiero institucional senior.

CONTEXTO TÉCNICO DEL PORTAFOLIO:
{context}

TAREA:
1️⃣ EVALUACIÓN DE DIVERSIFICACIÓN: ¿Los pesos asignados realmente diversifican el riesgo?
2️⃣ IDENTIFICACIÓN DE RIESGOS: País/moneda, sectorial, liquidez, concentración
3️⃣ RECOMENDACIONES: 2-3 ajustes concretos de pesos para mejorar ratio riesgo/retorno
4️⃣ ALERTAS: Concentración >40%, correlación >0.8, exposición >70% a una moneda

FORMATO: Viñetas claras, lenguaje profesional, específico con tickers. Responde en español."""
                            
                            if st.session_state.preferred_ai == "OpenAI":
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                response = client.chat.completions.create(
                                    model=st.session_state.get('openai_model', 'gpt-4o'),
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.2)
                                st.info(response.choices[0].message.content)
                            elif st.session_state.preferred_ai == "Gemini":
                                genai.configure(api_key=st.session_state.gemini_api_key)
                                model = genai.GenerativeModel(st.session_state.gemini_model)
                                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
                                st.info(response.text)
                                
                        except Exception as e:
                            st.error(f"Error API IA: {type(e).__name__}: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3: REBALANCEO INTELIGENTE
    # ─────────────────────────────────────────────────────────────────────
    with tabs[2]:
        if 'last_opt_res' not in st.session_state:
            st.info("👈 Primero optimiza un portafolio en la pestaña anterior")
            return
            
        st.subheader("🔄 Análisis de Rebalanceo")
        
        rebal_tabs = st.tabs(["📋 Estado Actual", "🎯 Calcular Rebalanceo"])
        
        with rebal_tabs[0]:
            if portfolios[p_sel]["weights"]:
                current_weights = {t: w for t, w in zip(portfolios[p_sel]["tickers"], portfolios[p_sel]["weights"])}
                target_weights = {t: w for t, w in zip(st.session_state['last_opt_res']['tickers'], st.session_state['last_opt_res']['weights'])}
                
                drift_analysis = get_rebalancing_strategy(current_weights, target_weights)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("📏 Drift Máximo", f"{drift_analysis['max_drift']:.2%}")
                c2.metric("🎯 Umbral", f"{drift_analysis['threshold']:.2%}")
                c3.metric("📊 Estado", drift_analysis['status'], 
                         delta_color="inverse" if drift_analysis['needs_rebalancing'] else "normal")
                
                # Visualizar drift
                drift_df = pd.DataFrame({
                    "Activo": list(current_weights.keys()),
                    "Peso Actual (%)": [current_weights.get(t, 0)*100 for t in current_weights],
                    "Peso Target (%)": [target_weights.get(t, 0)*100 for t in current_weights]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=drift_df["Activo"], y=drift_df["Peso Actual (%)"], 
                                    name="Actual", marker_color='rgba(239,85,59,0.7)'))
                fig.add_trace(go.Bar(x=drift_df["Activo"], y=drift_df["Peso Target (%)"], 
                                    name="Target", marker_color='rgba(0,204,150,0.7)'))
                fig.update_layout(barmode='group', title="📊 Drift vs Target", template="plotly_dark", yaxis_title="Peso (%)")
                st.plotly_chart(fig, use_container_width=True)
        
        with rebal_tabs[1]:
            st.subheader("🎯 Calcular Operaciones de Rebalanceo")
            
            total_portfolio_value = st.number_input("💰 Valor Total del Portafolio ($)", min_value=1000, value=100000, step=1000)
            min_trade = st.number_input("📐 Tamaño Mínimo de Operación ($)", min_value=50, value=100, step=50)
            
            if st.button("🔄 Calcular Rebalanceo", type="primary"):
                # Obtener precios actuales
                current_prices = {}
                for ticker in st.session_state['last_opt_res']['tickers']:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        current_prices[ticker] = info.get('currentPrice') or info.get('regularMarketPrice') or 1
                    except:
                        current_prices[ticker] = 1
                
                target_weights = {t: w for t, w in zip(st.session_state['last_opt_res']['tickers'], st.session_state['last_opt_res']['weights'])}
                
                # Asumir portfolio actual según pesos originales
                original_weights = {t: w for t, w in zip(portfolios[p_sel]["tickers"], portfolios[p_sel]["weights"])}
                current_portfolio = {t: total_portfolio_value * original_weights.get(t, 0) for t in target_weights}
                
                trades_df = calculate_rebalancing_trades(
                    current_portfolio, target_weights, current_prices, total_portfolio_value, min_trade
                )
                
                if not trades_df.empty:
                    st.dataframe(trades_df.style.format({"Cantidad": "{:.2f}"}), use_container_width=True)
                    
                    # Resumen
                    total_buy = trades_df[trades_df["Acción"].str.contains("COMPRAR")]["Diferencia ($)"].str.replace('$','').str.replace(',','').astype(float).sum()
                    total_sell = trades_df[trades_df["Acción"].str.contains("VENDER")]["Diferencia ($)"].str.replace('$','').str.replace(',','').astype(float).abs().sum()
                    
                    c1, c2 = st.columns(2)
                    c1.metric("🟢 Total a Comprar", f"${total_buy:,.0f}")
                    c2.metric("🔴 Total a Vender", f"${total_sell:,.0f}")
                    
                    # Exportar
                    csv = trades_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar Órdenes (CSV)", csv, "rebalance_orders.csv", "text/csv")
                else:
                    st.success("✅ No se requieren operaciones de rebalanceo. Tu portafolio está bien balanceado")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4: FORECAST & SIMULACIÓN
    # ─────────────────────────────────────────────────────────────────────
    with tabs[3]:
        if 'last_opt_res' not in st.session_state:
            st.info("👈 Primero optimiza un portafolio")
            return
            
        res = st.session_state['last_opt_res']
        st.subheader("🔮 Simulación Montecarlo de Escenarios Futuros")
        
        c_sim1, c_sim2 = st.columns(2)
        days = c_sim1.slider("📅 Días de Proyección", 30, 365, 90)
        n_sims = c_sim2.selectbox("🎲 Cantidad de Simulaciones", [100, 500, 1000, 5000], index=1)
        
        if st.button("🚀 Simular Escenarios Futuros", type="primary"):
            dt = 1/252
            mu = res['expected_return'] * dt
            sigma = res['volatility'] * np.sqrt(dt)
            
            paths = np.zeros((days, n_sims))
            paths[0] = 100
            
            for t in range(1, days):
                rand = np.random.standard_normal(n_sims)
                paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand)
            
            fig = go.Figure()
            p95, p50, p05 = np.percentile(paths, [95, 50, 5], axis=1)
            x_axis = np.arange(days)
            
            # Intervalo de confianza
            fig.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([p95, p05[::-1]]), fill='toself',
                fillcolor='rgba(0,204,150,0.15)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'))
            
            # Líneas
            fig.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', name='Mediana (P50)', line=dict(color='#00CC96', width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='Pesimista (P5)', line=dict(color='#EF553B', dash='dash')))
            fig.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', name='Optimista (P95)', line=dict(color='#636EFA', dash='dash')))
            
            fig.update_layout(
                template="plotly_dark",
                title=f"🔮 Proyección Montecarlo: {days} días | {n_sims} simulaciones",
                xaxis_title="Días",
                yaxis_title="Valor del Portafolio (Base 100)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas finales
            final_values = paths[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Mediana Final", f"${np.median(final_values):.0f}")
            c2.metric("📈 P95 (Optimista)", f"${np.percentile(final_values, 95):.0f}")
            c3.metric("📉 P05 (Pesimista)", f"${np.percentile(final_values, 5):.0f}")
            c4.metric("⚠️ Prob. Pérdida", f"{np.mean(final_values < 100)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA: RENTA FIJA (sin cambios significativos, solo mejoras menores)
# ═══════════════════════════════════════════════════════════════════════════

def calc_bond_metrics(face_value, coupon_rate, ytm, years_to_maturity, freq=2):
    periods = int(years_to_maturity * freq)
    coupon = (coupon_rate / freq) * face_value
    rate = ytm / freq
    price = 0
    mac_dur_num = 0
    conv_num = 0
    for t in range(1, periods + 1):
        cf = coupon if t < periods else coupon + face_value
        pv = cf / ((1 + rate)**t)
        price += pv
        mac_dur_num += (t / freq) * pv
        conv_num += (t / freq) * ((t / freq) + (1/freq)) * cf / ((1 + rate)**(t + 2))
    mac_dur = mac_dur_num / price if price > 0 else 0
    mod_dur = mac_dur / (1 + rate)
    convexity = conv_num / price if price > 0 else 0
    return price, mac_dur, mod_dur, convexity


def page_fixed_income():
    st.title("🏛️ Renta Fija: Análisis, Sensibilidad e Inmunización")
    st.markdown("Ingresa tu cartera de bonos para calcular duración, medir el riesgo frente a cambios en las tasas y evaluar la inmunización del portafolio.")
    
    st.subheader("⚙️ Configuración de Cartera")
    rf_mode = st.radio("Método de Ingreso de Datos", ["✍️ Carga Manual", "🏦 Importar Precios desde IOL"], horizontal=True)
    
    if rf_mode == "🏦 Importar Precios desde IOL":
        c_iol1, c_iol2 = st.columns([3, 1])
        iol_tickers = c_iol1.text_input("Tickers a Importar (ej. AL30, GD30, TX26)", key="iol_bonds")
        if c_iol2.button("⬇️ Consultar IOL", key="fetch_iol"):
            client = get_iol_client()
            if not client or not st.session_state.get('iol_username'):
                st.error("⚠️ Conéctate a IOL en la barra lateral primero.")
            else:
                tickers_list = [t.strip().upper() for t in iol_tickers.split(",") if t.strip()]
                fetched_bonds = []
                with st.spinner("Buscando cotizaciones en IOL..."):
                    for t in tickers_list:
                        try:
                            start_d = (pd.to_datetime("today") - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                            end_d = pd.to_datetime("today").strftime("%Y-%m-%d")
                            df_hist = client.get_serie_historica(t, start_d, end_d)
                            if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                                last_price = df_hist["ultimoPrecio"].iloc[-1]
                                fetched_bonds.append({
                                    "Bono": t, "Cupón (%)": 5.0, "YTM (%)": 15.0,
                                    "Años a Venc.": 3.0, "Nominal Invertido": 10000
                                })
                        except Exception as e:
                            st.warning(f"No se pudo obtener {t}: {e}")
                
                if fetched_bonds:
                    st.session_state.bonds_data = pd.DataFrame(fetched_bonds)
                    st.success("✅ Tickers cargados. Ajusta Cupón, YTM y Vencimiento manualmente.")

    if 'bonds_data' not in st.session_state:
        st.session_state.bonds_data = pd.DataFrame({
            "Bono": ["Bono Corto", "Bono Medio", "Bono Largo"],
            "Cupón (%)": [3.0, 4.5, 6.0], "YTM (%)": [4.0, 5.0, 6.5],
            "Años a Venc.": [2.0, 5.0, 10.0], "Nominal Invertido": [100000, 150000, 50000]
        })

    edited_bonds = st.data_editor(st.session_state.bonds_data, num_rows="dynamic", use_container_width=True, key="bonds_editor")
    st.session_state.bonds_data = edited_bonds

    tabs = st.tabs(["📊 Análisis e Inmunización", "📉 Sensibilidad", "📈 Curva de Rendimiento", "💬 Chat IA"])

    results = []
    total_investment = 0
    
    for _, row in edited_bonds.iterrows():
        try:
            p, macd, modd, conv = calc_bond_metrics(
                face_value=100, coupon_rate=row["Cupón (%)"]/100, 
                ytm=row["YTM (%)"]/100, years_to_maturity=row["Años a Venc."]
            )
            weight = row["Nominal Invertido"]
            total_investment += weight
            results.append({
                "Bono": row["Bono"], "Precio Calc.": p, "Mac. Dur": macd, 
                "Mod. Dur": modd, "Convexidad": conv, "Peso $": weight
            })
        except Exception as e:
            pass

    if total_investment > 0 and results:
        df_res = pd.DataFrame(results)
        df_res["Peso %"] = df_res["Peso $"] / total_investment
        port_mac_dur = (df_res["Mac. Dur"] * df_res["Peso %"]).sum()
        port_mod_dur = (df_res["Mod. Dur"] * df_res["Peso %"]).sum()
        port_convexity = (df_res["Convexidad"] * df_res["Peso %"]).sum()

        with tabs[0]:
            st.subheader("📊 Métricas de Riesgo del Portafolio de Bonos")
            c1, c2, c3 = st.columns(3)
            c1.metric("Macaulay Duration (Años)", f"{port_mac_dur:.2f}")
            c2.metric("Modified Duration", f"{port_mod_dur:.2f}")
            c3.metric("Convexidad Total", f"{port_convexity:.2f}")
            st.dataframe(df_res.style.format({
                "Precio Calc.": "{:.2f}", "Mac. Dur": "{:.2f}", "Mod. Dur": "{:.2f}", 
                "Convexidad": "{:.4f}", "Peso %": "{:.2%}"
            }), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🛡️ Análisis de Inmunización")
            horizonte = st.slider("Tu Horizonte de Inversión (Años)", 0.5, 20.0, 5.0, 0.5)
            gap = port_mac_dur - horizonte
            if abs(gap) < 0.25:
                st.success(f"✅ Portafolio Inmunizado: Duración Macaulay ({port_mac_dur:.2f}) ≈ horizonte")
            elif gap > 0:
                st.warning(f"⚠️ Riesgo de Precio: Duración > horizonte. Subida de tasas afecta valor final.")
            else:
                st.info(f"ℹ️ Riesgo de Reinversión: Duración < horizonte. Caída de tasas reduce ingresos.")

        with tabs[1]:
            st.subheader("📉 Test de Estrés de Tasas")
            shock_bps = st.slider("Shock en Tasas (puntos básicos)", -500, 500, 100, 10)
            shock_pct = shock_bps / 10000
            df_res["Impacto %"] = (-df_res["Mod. Dur"] * shock_pct + 0.5 * df_res["Convexidad"] * (shock_pct**2)) * 100
            port_impacto = (-port_mod_dur * shock_pct + 0.5 * port_convexity * (shock_pct**2)) * 100
            st.metric("Variación Estimada del Portafolio", f"{port_impacto:.2f}%", delta=f"{shock_bps} bps")
            fig_stress = px.bar(df_res, x="Bono", y="Impacto %", color="Impacto %", 
                                title=f"Impacto por Activo ante shock de {shock_bps} bps",
                                color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_stress, use_container_width=True)

        with tabs[2]:
            st.subheader("📈 Curva de Rendimiento")
            df_curve = edited_bonds.sort_values("Años a Venc.")
            fig_curve = px.line(df_curve, x="Años a Venc.", y="YTM (%)", markers=True, text="Bono")
            fig_curve.update_traces(textposition="top center")
            fig_curve.update_layout(template="plotly_dark", yaxis_title="Yield (YTM %)", xaxis_title="Plazo (Años)")
            st.plotly_chart(fig_curve, use_container_width=True)

        with tabs[3]:
            st.subheader("💬 Asistente IA Especialista en Renta Fija")
            if not st.session_state.get('preferred_ai'):
                st.warning("⚠️ Configura una API Key en el menú lateral.")
            else:
                bond_context = f"Portafolio con Duration Modificada: {port_mod_dur:.2f}, Convexidad: {port_convexity:.2f}"
                if "bond_chat_history" not in st.session_state:
                    st.session_state.bond_chat_history = []
                for m in st.session_state.bond_chat_history:
                    st.chat_message(m["role"]).write(m["content"])
                if prompt := st.chat_input("Pregunta sobre tu estrategia de bonos..."):
                    st.session_state.bond_chat_history.append({"role": "user", "content": prompt})
                    st.chat_message("user").write(prompt)
                    with st.spinner("Analizando..."):
                        try:
                            full_prompt = f"Contexto: {bond_context}. Pregunta: {prompt}"
                            if st.session_state.preferred_ai == "OpenAI":
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                resp = client.chat.completions.create(
                                    model=st.session_state.openai_model,
                                    messages=[{"role": "user", "content": full_prompt}]
                                )
                                response_txt = resp.choices[0].message.content
                            else:
                                genai.configure(api_key=st.session_state.gemini_api_key)
                                model = genai.GenerativeModel(st.session_state.gemini_model)
                                response_txt = model.generate_content(full_prompt).text
                            st.session_state.bond_chat_history.append({"role": "assistant", "content": response_txt})
                            st.chat_message("assistant").write(response_txt)
                        except Exception as e:
                            st.error(f"Error IA: {e}")
    else:
        st.info("➕ Agrega bonos a la tabla para ver el análisis de riesgo.")


# ═══════════════════════════════════════════════════════════════════════════
#  OTRAS PÁGINAS (simplificadas para brevedad)
# ═══════════════════════════════════════════════════════════════════════════

def page_yahoo_explorer():
    st.title("🌎 Explorador de Mercado (Yahoo Finance)")
    c1, c2 = st.columns([1, 2])
    with c1: 
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
    if not ticker: return
    with st.spinner("Descargando datos..."):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                stock = yf.Ticker(ticker + ".BA")
                hist = stock.history(period=period)
            if hist.empty: st.error("No hay datos."); return
            info = stock.info
            st.subheader(f"{info.get('longName', ticker)}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precio", f"${info.get('currentPrice', hist['Close'].iloc[-1]):,.2f}")
            m2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            m3.metric("Beta", info.get('beta', 'N/A'))
            m4.metric("Sector", info.get('sector', 'N/A'))
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'],
                high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(title="Evolución", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")


def page_event_analyzer():
    st.header("📰 Analizador de Noticias con IA")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configure una API Key en la barra lateral."); return
    uploaded_file = st.file_uploader("📎 Adjuntar documento", type=["pdf", "csv", "docx", "txt", "md"])
    news_text = extract_text_from_file(uploaded_file, max_chars=15000) if uploaded_file else ""
    manual_text = st.text_area("O pega la noticia/texto aquí:", value="", height=150)
    if manual_text and not news_text: news_text = manual_text
    elif manual_text and news_text: news_text = f"{news_text}\n\n---\n\nTexto adicional:\n{manual_text}"
    if st.button("🤖 Analizar"):
        if not news_text.strip(): st.warning("Por favor, adjunta un archivo o pega texto."); return
        with st.spinner(f"Analizando con {st.session_state.preferred_ai}..."):
            try:
                prompt = f"""Actúa como analista financiero senior. Analiza:
{truncate_for_tokens(news_text, max_tokens=4000)}
Proporciona: 1) Resumen ejecutivo, 2) Impacto en mercados, 3) Activos expuestos, 4) Recomendación.
Responde en español con viñetas claras."""
                if st.session_state.preferred_ai == "OpenAI":
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    response = client.chat.completions.create(
                        model=st.session_state.openai_model,
                        messages=[{"role": "user", "content": prompt}])
                    st.markdown(response.choices[0].message.content)
                elif st.session_state.preferred_ai == "Gemini":
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    model = genai.GenerativeModel(st.session_state.gemini_model)
                    st.markdown(model.generate_content(prompt).text)
            except Exception as e: st.error(f"Error: {e}")


def page_chat_general():
    st.header("💬 Asistente IA General")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configure una API Key en la barra lateral."); return
    if "general_messages" not in st.session_state: st.session_state.general_messages = []
    for msg in st.session_state.general_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Escribe tu consulta financiera..."):
        st.session_state.general_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            if st.session_state.preferred_ai == "OpenAI":
                client = OpenAI(api_key=st.session_state.openai_api_key)
                messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.general_messages[-10:]]
                response = client.chat.completions.create(model=st.session_state.openai_model, messages=messages_for_api)
                reply = response.choices[0].message.content
            elif st.session_state.preferred_ai == "Gemini":
                genai.configure(api_key=st.session_state.gemini_api_key)
                model = genai.GenerativeModel(st.session_state.gemini_model)
                hist = [{'role': ('user' if m['role']=='user' else 'model'), 'parts': [m['content']]} for m in st.session_state.general_messages[-10:-1]]
                chat = model.start_chat(history=hist)
                reply = chat.send_message(prompt).text
            st.session_state.general_messages.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)
        except Exception as e: st.error(f"Error: {e}")


def page_ai_strategy_assistant():
    st.header("🧠 Asistente Quant: Estrategia IA")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configura una API Key en la barra lateral."); return
    
    with st.expander("📊 Adjuntar datos personalizados (CSV)"):
        uploaded_csv = st.file_uploader("CSV con series temporales", type=["csv"])
        custom_data_context = ""
        if uploaded_csv is not None:
            try:
                df_custom = pd.read_csv(uploaded_csv)
                custom_data_context = f"\n📊 DATOS PERSONALIZADOS:\n{df_custom.head(30).to_markdown(index=False)}"
                st.success(f"✅ CSV cargado: {df_custom.shape}")
            except Exception as e: st.error(f"Error: {e}")
    
    user_strategy_prompt = st.text_area("Describe tu estrategia de inversión:", height=120)
    
    if st.button("Traducir Estrategia a Filtros", type="primary"):
        if not user_strategy_prompt: st.warning("Describe tu estrategia."); return
        system_prompt = """Eres un experto en finanzas cuantitativas. Traduce la estrategia en JSON con claves:
        k_assets, asset_allocation, beta_range, pe_range, duration_range, universe_stocks, universe_bonds.
        Responde SOLO con JSON válido."""
        full_prompt = f"{system_prompt}\n\nEstrategia: {user_strategy_prompt}{custom_data_context}"
        with st.spinner("IA Quant analizando..."):
            try:
                raw_response = ""
                if st.session_state.preferred_ai == "OpenAI":
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    response = client.chat.completions.create(
                        model=st.session_state.openai_model,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt}],
                        temperature=0.1)
                    raw_response = response.choices[0].message.content
                elif st.session_state.preferred_ai == "Gemini":
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    model = genai.GenerativeModel(st.session_state.gemini_model)
                    raw_response = model.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(temperature=0.1)).text
                
                if raw_response:
                    json_text = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if json_text:
                        suggested_params = json.loads(json_text.group(0))
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Distribución:**"); st.json(suggested_params.get("asset_allocation", {}))
                            st.write(f"Beta: {suggested_params.get('beta_range')}")
                            bonds_list = suggested_params.get("universe_bonds", [])
                            st.write("**Bonos:**", ", ".join(bonds_list) if isinstance(bonds_list, list) else "No especificado")
                        with col2:
                            st.write(f"Duración: {suggested_params.get('duration_range')}")
                            stocks_list = suggested_params.get("universe_stocks", [])
                            st.write("**Acciones:**", ", ".join(stocks_list) if isinstance(stocks_list, list) else "No especificado")
                        st.code(json.dumps(suggested_params, indent=4), language="json")
                    else:
                        st.warning("⚠️ La IA no devolvió JSON válido")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()

st.sidebar.title("⚙️ Configuración")
render_gsheets_status()
st.sidebar.markdown("---")

with st.sidebar.expander("🤖 IA (OpenAI)", expanded=True):
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password",
        value=st.session_state.get('openai_api_key', st.secrets.get("openai", {}).get("api_key", "")))
    st.session_state.openai_model = st.selectbox("Modelo", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

with st.sidebar.expander("🧠 IA (Gemini)", expanded=False):
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password",
        value=st.session_state.get('gemini_api_key', st.secrets.get("gemini", {}).get("api_key", "")))
    st.session_state.gemini_model = st.selectbox("Modelo",
        ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"])

st.sidebar.markdown("---")
available_ais = []
if OPENAI_OK and st.session_state.get('openai_api_key'): available_ais.append("OpenAI")
if GEMINI_OK and st.session_state.get('gemini_api_key'): available_ais.append("Gemini")

if available_ais:
    st.session_state.preferred_ai = st.sidebar.radio("✨ Motor IA Activo", available_ais)
else:
    st.session_state.preferred_ai = None
    st.sidebar.warning("⚠️ Ingresa una API Key para usar IA")

with st.sidebar.expander("🏦 Conexión IOL", expanded=True):
    user_iol = st.text_input("Usuario IOL", value=st.session_state.get('iol_username', ''))
    pass_iol = st.text_input("Contraseña IOL", type="password", value=st.session_state.get('iol_password', ''))
    if st.button("Conectar", use_container_width=True):
        st.session_state.iol_username = user_iol
        st.session_state.iol_password = pass_iol
        with st.spinner("Validando..."):
            client = get_iol_client()
            st.session_state.iol_connected = client is not None
    if st.session_state.get('iol_connected'):
        st.success(f"🟢 Conectado: {st.session_state.iol_username}")
    else:
        st.info("🔴 Desconectado")

st.sidebar.markdown("---")
opciones = [
    "Inicio", "📊 Dashboard Corporativo", "🏛️ Renta Fija",
    "🧠 Asistente Quant", "🏦 Explorador IOL",
    "🌎 Explorador Global", "🔭 Forecast Avanzado",
    "📰 Analizador Eventos", "💬 Chat IA General"
]
sel = st.sidebar.radio("Navegación", opciones,
    index=opciones.index(st.session_state.selected_page) if st.session_state.selected_page in opciones else 0)

if sel != st.session_state.selected_page: 
    st.session_state.selected_page = sel
    st.rerun()

# ───────────────────────────────────────────────────────────────────────────
#  ROUTER DE PÁGINAS
# ───────────────────────────────────────────────────────────────────────────

if sel == "Inicio":
    st.title("📈 INVERSIONES PRO - Finanzas Corporativas")
    st.markdown("""
    ### 🚀 Plataforma Integral de Gestión de Portafolios
    
    **Funcionalidades Principales:**
    
    🔹 **Optimización Avanzada**: Markowitz, Risk Parity, Black-Litterman, HRP
    🔹 **Análisis de Riesgo**: VaR, CVaR, Sortino, Calmar, Omega ratios
    🔹 **Rebalanceo Inteligente**: Detección de drift y generación de órdenes
    🔹 **Simulación Montecarlo**: Proyección de escenarios futuros
    🔹 **IA Integrada**: Análisis cualitativo con GPT-4 / Gemini
    🔹 **Multi-activo**: Acciones, bonos, ETFs, mercados emergentes
    
    ---
    💡 **Recomendación**: Comienza en "📊 Dashboard Corporativo" para crear y optimizar tu primer portafolio.
    """)
    
    st.info("🔧 **Dependencias recomendadas**: `pip install PyPortfolioOpt` para habilitar optimización avanzada")
    
elif sel == "📊 Dashboard Corporativo": 
    page_corporate_dashboard()
elif sel == "🏛️ Renta Fija": 
    page_fixed_income()
elif sel == "🧠 Asistente Quant": 
    page_ai_strategy_assistant()
elif sel == "🏦 Explorador IOL": 
    page_iol_explorer()
elif sel == "🌎 Explorador Global": 
    page_yahoo_explorer()
elif sel == "🔭 Forecast Avanzado": 
    page_forecast()
elif sel == "📰 Analizador Eventos": 
    page_event_analyzer()
elif sel == "💬 Chat IA General": 
    page_chat_general()
