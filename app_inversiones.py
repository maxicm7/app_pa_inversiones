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
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

# ── Módulos propios (Manejo de errores si no existen) ──
try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
    def page_forecast(): st.warning("Módulo forecast_module.py no encontrado.")
    def page_iol_explorer(): st.warning("Módulo iol_client.py no encontrado.")
    def get_iol_client(): return None

# ── Configuración Global ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Finanzas Corporativas", page_icon="📈")

# ───────────────────────────────────────────────────────────────────────────
# NOMBRE DE LA HOJA — debe coincidir exactamente con el nombre del Google
# Sheet que creaste y compartiste con el Service Account.
# En secrets.toml: [google_sheets] / sheet_name = "BPNos_Portfolios"
# ───────────────────────────────────────────────────────────────────────────
SHEET_NAME = st.secrets.get("google_sheets", {}).get("sheet_name", "Epre_Inversiones")
# ID directo del Sheet — permite abrir sin scope de Drive
SHEET_ID   = st.secrets.get("google_sheets", {}).get("sheet_id", "")

# Nombre de la pestaña (worksheet) dentro del Google Sheet
WORKSHEET_NAME = "portfolios"

# Archivo JSON local (fallback si Google Sheets no está disponible)
PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  CONEXIÓN A GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_gsheets_client():
    """
    Autenticación via Service Account.
    Acepta DOS formatos en secrets.toml:

    FORMATO A (recomendado — pegar el JSON completo):
        [google_sheets]
        service_account_json = '''{ "type": "service_account", ... }''' 

    FORMATO B (campos separados — legacy):
        [gcp_service_account]
        type = "service_account"
        private_key = "..."
        ...
    """
    if not GSHEETS_OK:
        return None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]

        # ── FORMATO A: JSON completo en un solo campo ──────────────────
        raw_json = st.secrets.get("google_sheets", {}).get("service_account_json", "")
        if raw_json:
            creds_dict = json.loads(raw_json)
        # ── FORMATO B: campos separados (fallback) ─────────────────────
        else:
            creds_dict = dict(st.secrets["gcp_service_account"])
            pk = creds_dict.get("private_key", "")
            # Reparar \n literales
            pk = pk.replace("\\n", "\n")
            # Reparar multilínea TOML: limpiar espacios y reconstruir
            lines = pk.splitlines()
            cleaned_lines = [l.strip() for l in lines if l.strip()]
            pk = "\n".join(cleaned_lines) + "\n"
            creds_dict["private_key"] = pk

        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Error Google Sheets: {e}")
        return None


def get_or_create_worksheet(client, sheet_name: str, worksheet_name: str):
    """
    Abre el Sheet por ID (open_by_key) — no necesita scope de Drive.
    Solo crea la pestaña si no existe.
    """
    try:
        if SHEET_ID:
            spreadsheet = client.open_by_key(SHEET_ID)
        else:
            spreadsheet = client.open(sheet_name)
    except Exception as e:
        st.error(
            f"❌ No se pudo abrir el Sheet. "
            f"Verificá el sheet_id en Secrets y que esté compartido con el Service Account. "
            f"Detalle: {e}"
        )
        raise

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=200, cols=3)
        worksheet.append_row(["name", "tickers", "weights"])

    return worksheet


# ═══════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE DATOS Y PORTAFOLIOS — GOOGLE SHEETS COMO BACKEND
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_gsheet() -> dict:
    """
    Lee todos los portafolios desde Google Sheets.
    Columnas esperadas: name | tickers (CSV) | weights (CSV)
    Devuelve dict con la misma estructura que antes:
      { "Nombre": {"tickers": [...], "weights": [...]} }
    """
    client = get_gsheets_client()
    if client is None:
        return _load_portfolios_local_fallback()

    try:
        ws = get_or_create_worksheet(client, SHEET_NAME, WORKSHEET_NAME)
        records = ws.get_all_records()  # lista de dicts usando fila 1 como cabecera
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
            portfolios[name] = {"tickers": tickers, "weights": weights}
        return portfolios
    except Exception as e:
        st.error(f"Error al leer Google Sheets: {e}")
        return _load_portfolios_local_fallback()


def save_portfolios_to_gsheet(portfolios_dict: dict) -> tuple[bool, str]:
    """
    Reescribe completamente la hoja con los portafolios actuales.
    """
    client = get_gsheets_client()
    if client is None:
        return _save_portfolios_local_fallback(portfolios_dict)

    try:
        ws = get_or_create_worksheet(client, SHEET_NAME, WORKSHEET_NAME)
        # Limpiar todo y reescribir
        ws.clear()
        rows = [["name", "tickers", "weights"]]
        for name, data in portfolios_dict.items():
            tickers_str = ", ".join(data["tickers"])
            weights_str = ", ".join(str(w) for w in data["weights"])
            rows.append([name, tickers_str, weights_str])
        ws.update(rows, "A1")
        # También guardamos local como respaldo
        _save_portfolios_local_fallback(portfolios_dict)
        return True, ""
    except Exception as e:
        st.error(f"Error al guardar en Google Sheets: {e}")
        return _save_portfolios_local_fallback(portfolios_dict)


# ── Fallbacks locales (mantiene compatibilidad si Sheets no está disponible) ──

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


# ── Funciones públicas (reemplazan las anteriores sin cambiar el resto del código) ──

def load_portfolios_from_file() -> dict:
    return load_portfolios_from_gsheet()


def save_portfolios_to_file(portfolios_dict: dict) -> tuple[bool, str]:
    return save_portfolios_to_gsheet(portfolios_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO: DESCARGA Y OPTIMIZACIÓN  (sin cambios)
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
                        "returns": returns, "method": "PyPortfolioOpt"}
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


# ═══════════════════════════════════════════════════════════════════════════
#  ESTADO DEL SIDEBAR — INDICADOR DE CONEXIÓN GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════════════════════

def render_gsheets_status():
    """Muestra en el sidebar si la conexión a Google Sheets está activa."""
    client = get_gsheets_client()
    if client:
        st.sidebar.success(f"🟢 Google Sheets: {SHEET_NAME}")
    else:
        st.sidebar.error("🔴 Google Sheets: no conectado (usando JSON local)")


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS DE LA APLICACIÓN  (sin cambios internos)
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo Integral")
    tabs = st.tabs(["💼 Gestión de Portafolios", "🚀 Optimización & Riesgo", "🔮 Forecast & Simulación"])

    with tabs[0]:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("Administrar Carteras")
            action = st.radio("Acción:", ["✨ Crear Nueva", "✏️ Editar / 🗑️ Eliminar"], horizontal=True)

            if action == "✨ Crear Nueva":
                p_name = st.text_input("Nombre de la Cartera")
                p_tickers = st.text_area("Tickers (separados por coma)", "AL30, GGAL").upper()
                p_weights = st.text_area("Pesos (separados por coma, sumar 1.0)", "0.5, 0.5")

                if st.button("💾 Guardar Nueva Cartera", type="primary"):
                    try:
                        t = [x.strip() for x in p_tickers.split(",") if x.strip()]
                        w = [float(x) for x in p_weights.split(",") if x.strip()]
                        if not p_name:
                            st.error("El nombre no puede estar vacío.")
                        elif len(t) == len(w) and abs(sum(w)-1.0) < 0.02:
                            st.session_state.portfolios[p_name] = {"tickers": t, "weights": w}
                            ok, err = save_portfolios_to_file(st.session_state.portfolios)
                            if ok:
                                st.success("✅ Guardado en Google Sheets exitosamente.")
                            else:
                                st.error(f"Error al guardar: {err}")
                            st.rerun()
                        else:
                            st.error("Error: La cantidad de pesos no coincide o no suman 1.0.")
                    except:
                        st.error("Error de formato.")

            else:
                if st.session_state.portfolios:
                    edit_sel = st.selectbox("Seleccionar Cartera:", list(st.session_state.portfolios.keys()))
                    curr_data = st.session_state.portfolios[edit_sel]
                    new_name = st.text_input("Renombrar Cartera", value=edit_sel)
                    new_tickers = st.text_area("Modificar Tickers", value=", ".join(curr_data["tickers"])).upper()
                    new_weights = st.text_area("Modificar Pesos", value=", ".join(map(str, curr_data["weights"])))

                    col_b1, col_b2 = st.columns(2)
                    if col_b1.button("🔄 Actualizar", type="primary", use_container_width=True):
                        try:
                            t = [x.strip() for x in new_tickers.split(",") if x.strip()]
                            w = [float(x) for x in new_weights.split(",") if x.strip()]
                            if len(t) == len(w) and abs(sum(w)-1.0) < 0.02:
                                if new_name != edit_sel:
                                    del st.session_state.portfolios[edit_sel]
                                st.session_state.portfolios[new_name] = {"tickers": t, "weights": w}
                                save_portfolios_to_file(st.session_state.portfolios)
                                st.success("Cartera actualizada.")
                                st.rerun()
                            else:
                                st.error("Revisar validación de pesos y tickers.")
                        except:
                            st.error("Error de formato.")

                    if col_b2.button("🗑️ Eliminar", type="primary", use_container_width=True):
                        del st.session_state.portfolios[edit_sel]
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.warning("Cartera eliminada permanentemente.")
                        st.rerun()
                else:
                    st.info("No hay carteras guardadas actualmente.")

        with c2:
            st.subheader("Base de Datos (Google Sheets)")
            if st.session_state.portfolios:
                df_ports = pd.DataFrame([
                    {"Nombre": k, "Activos": ", ".join(v["tickers"]),
                     "Ponderación (%)": ", ".join([f"{w*100:.1f}%" for w in v["weights"]])}
                    for k, v in st.session_state.portfolios.items()
                ])
                st.dataframe(df_ports, use_container_width=True, hide_index=True)

    portfolios = st.session_state.get("portfolios", {})
    if not portfolios: return

    with tabs[1]:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        p_sel = col1.selectbox("Analizar Cartera:", list(portfolios.keys()))
        d_start = col2.date_input("Desde", pd.to_datetime("2023-01-01"))
        d_end = col3.date_input("Hasta", pd.to_datetime("today"))

        st.subheader("📈 Rendimiento Histórico del Portafolio")
        if st.button("📊 Ver Rendimiento Histórico"):
            with st.spinner("Descargando datos históricos..."):
                prices_perf = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)

            if prices_perf is not None:
                current_weights = list(portfolios[p_sel]["weights"])
                tickers_in_prices = [t for t in portfolios[p_sel]["tickers"] if t in prices_perf.columns]
                if not tickers_in_prices:
                    st.error("⚠️ No se encontraron datos de precios.")
                    st.stop()
                if len(tickers_in_prices) < len(portfolios[p_sel]["tickers"]):
                    missing = set(portfolios[p_sel]["tickers"]) - set(tickers_in_prices)
                    st.warning(f"Sin datos para: {', '.join(missing)}. Pesos recalculados.")
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
                m1.metric("Retorno Total", f"{total_return:.2f}%")
                m2.metric("Volatilidad Anualizada", f"{ann_vol:.2f}%")
                m3.metric("Máximo Drawdown", f"{max_dd:.2f}%")
                m4.metric("Sharpe Histórico", f"{sharpe_hist:.2f}")

                fig_perf = go.Figure()
                colors_ind = px.colors.qualitative.Pastel
                for i, ticker in enumerate(tickers_in_prices):
                    fig_perf.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[ticker] * 100,
                        mode='lines', name=ticker,
                        line=dict(width=1.5, dash='dot', color=colors_ind[i % len(colors_ind)]), opacity=0.65))
                fig_perf.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines',
                    name=f"📂 {p_sel}", line=dict(width=3, color='#00CC96'),
                    fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.07)'))
                fig_perf.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)
                fig_perf.update_layout(title=f"Evolución – {p_sel} (Base 100)", template="plotly_dark",
                    height=430, hovermode="x unified")
                st.plotly_chart(fig_perf, use_container_width=True)

                drawdown_series = (portfolio_value / portfolio_value.cummax() - 1) * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, mode='lines',
                    fill='tozeroy', fillcolor='rgba(239,85,59,0.2)', line=dict(color='#EF553B', width=1.5)))
                fig_dd.update_layout(title="Drawdown del Portafolio (%)", template="plotly_dark", height=230)
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.error("No se pudieron obtener datos.")

        st.markdown("---")
        st.subheader("⚙️ Optimización de Portafolio")
        c_opt1, c_opt2 = st.columns(2)
        risk_free = c_opt1.number_input("Tasa Libre Riesgo (RF)", 0.0, 0.5, 0.04, step=0.01)
        target = c_opt2.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])

        if st.button("Ejecutar Optimización"):
            with st.spinner("Optimizando..."):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
            if prices is not None:
                res = optimize_portfolio_corporate(prices, risk_free_rate=risk_free, opt_type=target)
                if res:
                    st.session_state['last_opt_res'] = res
                    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                    c_kpi1.metric("Retorno Esperado", f"{res['expected_return']:.1%}")
                    c_kpi2.metric("Volatilidad Anual", f"{res['volatility']:.1%}")
                    c_kpi3.metric("Ratio Sharpe", f"{res['sharpe_ratio']:.2f}")
                    w_df = pd.DataFrame({"Activo": res['tickers'], "Peso": res['weights']})
                    df_pie = w_df[w_df["Peso"] > 0.001]
                    if not df_pie.empty:
                        fig = px.pie(df_pie, values="Peso", names="Activo", title="Asignación", hole=0.4, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                else: st.error("Error al optimizar.")
            else: st.error("Error en datos.")

        if 'last_opt_res' in st.session_state:
            st.markdown("---")
            if st.button("🧠 Analizar Portafolio con IA"):
                if not st.session_state.get("preferred_ai"):
                    st.warning("⚠️ Ingresa tu API Key en el menú lateral.")
                else:
                    with st.spinner(f"Consultando IA ({st.session_state.preferred_ai})..."):
                        try:
                            data_str = (f"Retorno: {st.session_state['last_opt_res']['expected_return']:.2f}, "
                                        f"Volatilidad: {st.session_state['last_opt_res']['volatility']:.2f}. "
                                        f"Activos: {st.session_state['last_opt_res']['tickers']}.")
                            prompt = f"Actúa como asesor financiero institucional. Analiza este portafolio: {data_str}. ¿Cuáles son los riesgos y fortalezas?"
                            if st.session_state.preferred_ai == "OpenAI":
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                response = client.chat.completions.create(
                                    model=st.session_state.get('openai_model', 'gpt-4o'),
                                    messages=[{"role": "user", "content": prompt}])
                                st.info(response.choices[0].message.content)
                            elif st.session_state.preferred_ai == "Gemini":
                                genai.configure(api_key=st.session_state.gemini_api_key)
                                model = genai.GenerativeModel(st.session_state.gemini_model)
                                st.info(model.generate_content(prompt).text)
                        except Exception as e:
                            st.error(f"Error API IA: {e}")

    with tabs[2]:
        if 'last_opt_res' in st.session_state:
            res = st.session_state['last_opt_res']
            st.subheader("Simulación Montecarlo")
            c_sim1, c_sim2 = st.columns(2)
            days = c_sim1.slider("Días Proyección", 30, 365, 90)
            n_sims = c_sim2.selectbox("Cantidad Simulaciones", [100, 500, 1000], index=1)
            if st.button("Simular Escenarios Futuros"):
                dt = 1/252
                mu = res['expected_return'] * dt
                sigma = res['volatility'] * np.sqrt(dt)
                paths = np.zeros((days, n_sims))
                paths[0] = 100
                for t in range(1, days):
                    rand = np.random.standard_normal(n_sims)
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand)
                fig = go.Figure()
                p95, p05 = np.percentile(paths, 95, axis=1), np.percentile(paths, 5, axis=1)
                x_axis = np.arange(days)
                fig.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]),
                    y=np.concatenate([p95, p05[::-1]]), fill='toself',
                    fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'))
                fig.add_trace(go.Scatter(x=x_axis, y=np.mean(paths, axis=1), mode='lines',
                    name='Media', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines',
                    name='Pesimista (5%)', line=dict(color='#EF553B', dash='dash')))
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)


def page_fixed_income():
    st.title("🏛️ Renta Fija: Análisis, Sensibilidad e Inmunización")
    # ... (sin cambios respecto al original)
    st.info("Módulo Renta Fija sin cambios — pegar el código original de page_fixed_income() aquí.")


def page_fixed_income():
    st.title("🏛️ Renta Fija: Análisis, Sensibilidad e Inmunización")
    st.markdown("Ingresa tu cartera de bonos para calcular duración, medir el riesgo frente a cambios en las tasas y evaluar la inmunización del portafolio.")
    
    st.subheader("Configuración de Cartera")
    rf_mode = st.radio("Método de Ingreso de Datos", ["✍️ Carga Manual", "🏦 Importar Precios desde IOL"], horizontal=True)
    
    if rf_mode == "🏦 Importar Precios desde IOL":
        c_iol1, c_iol2 = st.columns([3, 1])
        iol_tickers = c_iol1.text_input("Tickers a Importar (ej. AL30, GD30, TX26)")
        if c_iol2.button("⬇️ Consultar IOL"):
            client = get_iol_client()
            if not client or not st.session_state.get('iol_username'):
                st.error("⚠️ Conéctate a IOL en la barra lateral primero.")
            else:
                tickers_list = [t.strip().upper() for t in iol_tickers.split(",") if t.strip()]
                fetched_bonds = []
                with st.spinner("Buscando cotizaciones en IOL..."):
                    for t in tickers_list:
                        try:
                            # Obtenemos el último precio para referencia
                            start_d = (pd.to_datetime("today") - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                            end_d = pd.to_datetime("today").strftime("%Y-%m-%d")
                            df_hist = client.get_serie_historica(t, start_d, end_d)
                            if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                                last_price = df_hist["ultimoPrecio"].iloc[-1]
                                fetched_bonds.append({
                                    "Bono": t,
                                    "Cupón (%)": 5.0,
                                    "YTM (%)": 15.0,
                                    "Años a Venc.": 3.0,
                                    "Nominal Invertido": 10000
                                })
                        except Exception as e:
                            st.warning(f"No se pudo obtener {t}: {e}")
                
                if fetched_bonds:
                    st.session_state.bonds_data = pd.DataFrame(fetched_bonds)
                    st.success("✅ Tickers cargados. Ajusta el Cupón, YTM y Vencimiento manualmente.")

    if 'bonds_data' not in st.session_state:
        st.session_state.bonds_data = pd.DataFrame({
            "Bono": ["Bono Corto", "Bono Medio", "Bono Largo"],
            "Cupón (%)": [3.0, 4.5, 6.0],
            "YTM (%)": [4.0, 5.0, 6.5],
            "Años a Venc.": [2.0, 5.0, 10.0],
            "Nominal Invertido": [100000, 150000, 50000]
        })

    # Editor de tabla dinámico
    edited_bonds = st.data_editor(st.session_state.bonds_data, num_rows="dynamic", use_container_width=True)
    st.session_state.bonds_data = edited_bonds

    tabs = st.tabs(["📊 Análisis e Inmunización", "📉 Sensibilidad (Test de Estrés)", "📈 Curva de Rendimiento", "💬 Chat IA Especializado"])

    results = []
    total_investment = 0
    port_mac_dur, port_mod_dur, port_convexity = 0, 0, 0
    
    # Cálculos iterativos por cada bono
    for _, row in edited_bonds.iterrows():
        try:
            # Usamos la función calc_bond_metrics definida arriba en el core
            p, macd, modd, conv = calc_bond_metrics(
                face_value=100, 
                coupon_rate=row["Cupón (%)"]/100, 
                ytm=row["YTM (%)"]/100, 
                years_to_maturity=row["Años a Venc."]
            )
            weight = row["Nominal Invertido"]
            total_investment += weight
            results.append({
                "Bono": row["Bono"], 
                "Precio Calc.": p, 
                "Mac. Dur": macd, 
                "Mod. Dur": modd, 
                "Convexidad": conv, 
                "Peso $": weight
            })
        except Exception as e:
            pass

    if total_investment > 0 and results:
        df_res = pd.DataFrame(results)
        df_res["Peso %"] = df_res["Peso $"] / total_investment
        
        # Métricas ponderadas del portafolio
        port_mac_dur = (df_res["Mac. Dur"] * df_res["Peso %"]).sum()
        port_mod_dur = (df_res["Mod. Dur"] * df_res["Peso %"]).sum()
        port_convexity = (df_res["Convexidad"] * df_res["Peso %"]).sum()

        with tabs[0]:
            st.subheader("Métricas de Riesgo del Portafolio de Bonos")
            c1, c2, c3 = st.columns(3)
            c1.metric("Macaulay Duration (Años)", f"{port_mac_dur:.2f}")
            c2.metric("Modified Duration", f"{port_mod_dur:.2f}")
            c3.metric("Convexidad Total", f"{port_convexity:.2f}")
            
            st.dataframe(df_res.style.format({
                "Precio Calc.": "{:.2f}", 
                "Mac. Dur": "{:.2f}", 
                "Mod. Dur": "{:.2f}", 
                "Convexidad": "{:.4f}",
                "Peso %": "{:.2%}"
            }), use_container_width=True)

            st.markdown("---")
            st.subheader("🛡️ Análisis de Inmunización")
            horizonte = st.slider("Tu Horizonte de Inversión (Años)", 0.5, 20.0, 5.0, 0.5)
            gap = port_mac_dur - horizonte
            if abs(gap) < 0.25:
                st.success(f"✅ Portafolio Inmunizado: La Duración Macaulay ({port_mac_dur:.2f}) coincide con tu horizonte.")
            elif gap > 0:
                st.warning(f"⚠️ Riesgo de Precio: La duración es mayor al horizonte. Una subida de tasas afectará el valor final.")
            else:
                st.info(f"ℹ️ Riesgo de Reinversión: La duración es menor al horizonte. Una caída de tasas reducirá los ingresos por reinversión.")

        with tabs[1]:
            st.subheader("Test de Estrés de Tasas (Aproximación Taylor)")
            shock_bps = st.slider("Shock en Tasas (puntos básicos)", -500, 500, 100, 10)
            shock_pct = shock_bps / 10000
            
            # Cálculo de impacto: -Dmod * dy + 0.5 * Conv * dy^2
            df_res["Impacto %"] = (-df_res["Mod. Dur"] * shock_pct + 0.5 * df_res["Convexidad"] * (shock_pct**2)) * 100
            port_impacto = (-port_mod_dur * shock_pct + 0.5 * port_convexity * (shock_pct**2)) * 100
            
            st.metric("Variación Estimada del Portafolio", f"{port_impacto:.2f}%", delta=f"{shock_bps} bps")
            
            fig_stress = px.bar(df_res, x="Bono", y="Impacto %", color="Impacto %", 
                                title=f"Impacto por Activo ante shock de {shock_bps} bps",
                                color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_stress, use_container_width=True)

        with tabs[2]:
            st.subheader("Estructura Temporal de Tasas (Curva del Portafolio)")
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
                # Contexto técnico para la IA
                bond_context = f"El usuario tiene un portafolio de bonos con Duración Modificada de {port_mod_dur:.2f} y Convexidad de {port_convexity:.2f}."
                
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
        st.info("Agregue bonos a la tabla para ver el análisis de riesgo.")


def page_yahoo_explorer():
    st.title("🌎 Explorador de Mercado (Yahoo Finance)")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    with c2: period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
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
        st.warning("⚠️ Configure una API Key en la barra lateral.")
        return
    news_text = st.text_area("Pega la noticia aquí:", height=150)
    if st.button("🤖 Analizar"):
        with st.spinner(f"Analizando con {st.session_state.preferred_ai}..."):
            try:
                prompt = f"Analiza financieramente esta noticia:\n\n'{news_text}'"
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
            except Exception as e:
                st.error(f"Error: {e}")


def page_chat_general():
    st.header("💬 Asistente IA General")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configure una API Key en la barra lateral.")
        return
    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []
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
        except Exception as e:
            st.error(f"Error: {e}")


def page_ai_strategy_assistant():
    st.header("🧠 Asistente Quant: Estrategia IA (Acciones y Bonos)")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configura una API Key en la barra lateral.")
        return
    user_strategy_prompt = st.text_area("Describe tu estrategia de inversión:", height=120,
        placeholder="Ej: 'Busco un portafolio conservador...'")
    if st.button("Traducir Estrategia a Filtros", type="primary"):
        if not user_strategy_prompt:
            st.warning("Por favor, describe tu estrategia.")
        else:
            system_prompt = """Eres un experto en finanzas cuantitativas. Traduce la estrategia del usuario en JSON con estas claves exactas:
            k_assets, asset_allocation, beta_range, pe_range, duration_range, universe_stocks, universe_bonds.
            Responde SOLO con el bloque JSON válido."""
            with st.spinner("IA Quant analizando..."):
                try:
                    raw_response = ""
                    if st.session_state.preferred_ai == "OpenAI":
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        response = client.chat.completions.create(
                            model=st.session_state.openai_model,
                            messages=[{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": user_strategy_prompt}],
                            temperature=0.1)
                        raw_response = response.choices[0].message.content
                    elif st.session_state.preferred_ai == "Gemini":
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        model = genai.GenerativeModel(st.session_state.gemini_model)
                        raw_response = model.generate_content(
                            f"{system_prompt}\n\n{user_strategy_prompt}",
                            generation_config=genai.types.GenerationConfig(temperature=0.1)).text
                    if raw_response:
                        json_text = re.search(r'\{.*\}', raw_response, re.DOTALL)
                        if json_text:
                            suggested_params = json.loads(json_text.group(0))
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Distribución de Activos:**"); st.json(suggested_params.get("asset_allocation", {}))
                                st.write(f"Beta: {suggested_params.get('beta_range')} | P/E: {suggested_params.get('pe_range')}")
                                st.write("**Acciones:**", ", ".join(suggested_params.get("universe_stocks", [])))
                            with col2:
                                st.write(f"Duración: {suggested_params.get('duration_range')}")
                                st.write("**Bonos:**", ", ".join(suggested_params.get("universe_bonds", [])))
                            st.code(json.dumps(suggested_params, indent=4), language="json")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()

st.sidebar.title("Configuración y Accesos")

# ── Estado de conexión Google Sheets ────
render_gsheets_status()
st.sidebar.markdown("---")

with st.sidebar.expander("🤖 IA (OpenAI / Copilot)", expanded=True):
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password",
        value=st.session_state.get('openai_api_key', st.secrets.get("openai", {}).get("api_key", "")))
    st.session_state.openai_model = st.selectbox("Modelo OpenAI", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

with st.sidebar.expander("🧠 IA (Gemini)", expanded=False):
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password",
        value=st.session_state.get('gemini_api_key', st.secrets.get("gemini", {}).get("api_key", "")))
    st.session_state.gemini_model = st.selectbox("Modelo Gemini",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"])

st.sidebar.markdown("---")
available_ais = []
if OPENAI_OK and st.session_state.get('openai_api_key'): available_ais.append("OpenAI")
if GEMINI_OK and st.session_state.get('gemini_api_key'): available_ais.append("Gemini")

if available_ais:
    st.session_state.preferred_ai = st.sidebar.radio("✨ Motor IA Activo", available_ais)
else:
    st.session_state.preferred_ai = None
    st.sidebar.warning("⚠️ Ingresa una API Key para usar IA.")

with st.sidebar.expander("🏦 Conexión IOL", expanded=True):
    user_iol = st.text_input("Usuario IOL", value=st.session_state.get('iol_username', ''))
    pass_iol = st.text_input("Contraseña IOL", type="password", value=st.session_state.get('iol_password', ''))
    if st.button("Conectar / Validar", use_container_width=True):
        st.session_state.iol_username = user_iol
        st.session_state.iol_password = pass_iol
        with st.spinner("Validando credenciales..."):
            client = get_iol_client()
            st.session_state.iol_connected = client is not None
    st.markdown("---")
    if st.session_state.get('iol_connected'):
        st.success(f"🟢 Conectado: {st.session_state.iol_username}")
    else:
        st.error("🔴 Desconectado")

st.sidebar.markdown("---")
opciones = [
    "Inicio", "📊 Dashboard Corporativo", "🏛️ Renta Fija (Bonos y Curvas)",
    "🧠 Asistente Quant (Estrategia IA)", "🏦 Explorador IOL API",
    "🌎 Explorador Global (Yahoo)", "🔭 Modelos Avanzados (Forecast)",
    "📰 Analizador Eventos (IA)", "💬 Chat IA General"
]
sel = st.sidebar.radio("Navegación", opciones,
    index=opciones.index(st.session_state.selected_page) if st.session_state.selected_page in opciones else 0)

if sel != st.session_state.selected_page: st.session_state.selected_page = sel; st.rerun()

if sel == "Inicio":
    st.title("BPNos - Finanzas Corporativas")
    st.info("Seleccione un módulo en la barra lateral.")
elif sel == "📊 Dashboard Corporativo": page_corporate_dashboard()
elif sel == "🏛️ Renta Fija (Bonos y Curvas)": page_fixed_income()
elif sel == "🧠 Asistente Quant (Estrategia IA)": page_ai_strategy_assistant()
elif sel == "🏦 Explorador IOL API": page_iol_explorer()
elif sel == "🌎 Explorador Global (Yahoo)": page_yahoo_explorer()
elif sel == "🔭 Modelos Avanzados (Forecast)": page_forecast()
elif sel == "📰 Analizador Eventos (IA)": page_event_analyzer()
elif sel == "💬 Chat IA General": page_chat_general()
