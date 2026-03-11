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

PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE DATOS Y PORTAFOLIOS (ALMACENAMIENTO PERMANENTE)
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error de lectura JSON: {e}")
    return {}

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO: DESCARGA Y OPTIMIZACIÓN
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
            raw = yf.download(adjusted_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if not raw.empty:
                close_data = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close_data, pd.Series):
                    close_data = close_data.to_frame(name=yf_tickers[0])
                
                if len(adjusted_tickers) == 1:
                     all_prices[yf_tickers[0]] = close_data.iloc[:, 0]
                else:
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
    if returns.empty: return None

    if PYPFOPT_OK:
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.sample_cov(prices, frequency=252)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            
            if opt_type == "Maximo Ratio Sharpe": ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif opt_type == "Minima Volatilidad": ef.min_volatility()
            else: ef.max_quadratic_utility(risk_aversion=0.0001)
            
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            ow_array = np.array([weights.get(col, 0) for col in prices.columns])
            
            return {"weights": ow_array, "expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe, "tickers": list(prices.columns), "returns": returns, "method": "PyPortfolioOpt"}
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

    if opt_type == "Minima Volatilidad": fun = lambda w: get_metrics(w)[1]
    elif opt_type == "Retorno Maximo": fun = lambda w: -get_metrics(w)[0]
    else: fun = lambda w: -get_metrics(w)[2]

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    final_metrics = get_metrics(res.x) if res.success else [0,0,0]
    
    return {"weights": res.x, "expected_return": final_metrics[0], "volatility": final_metrics[1], "sharpe_ratio": final_metrics[2], "tickers": list(prices.columns), "returns": returns, "method": "Scipy/SLSQP"}

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
#  PÁGINAS DE LA APLICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo Integral")
    tabs = st.tabs(["💼 Gestión de Portafolios", "🚀 Optimización & Riesgo", "🔮 Forecast & Simulación"])
    
    # --- TAB 1: GESTIÓN AVANZADA (CRUD) ---
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
                        if not p_name: st.error("El nombre no puede estar vacío.")
                        elif len(t) == len(w) and abs(sum(w)-1.0) < 0.02:
                            st.session_state.portfolios[p_name] = {"tickers": t, "weights": w}
                            save_portfolios_to_file(st.session_state.portfolios)
                            st.success("Guardado exitosamente.")
                            st.rerun()
                        else: st.error("Error: La cantidad de pesos no coincide con los tickers o no suman 1.0.")
                    except: st.error("Error de formato (asegúrate de usar números para los pesos).")
            
            else: # Editar o Eliminar
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
                                    del st.session_state.portfolios[edit_sel] # Borrar el viejo si cambió de nombre
                                st.session_state.portfolios[new_name] = {"tickers": t, "weights": w}
                                save_portfolios_to_file(st.session_state.portfolios)
                                st.success("Cartera actualizada.")
                                st.rerun()
                            else: st.error("Revisar validación de pesos y tickers.")
                        except: st.error("Error de formato.")
                    
                    if col_b2.button("🗑️ Eliminar", type="primary", use_container_width=True):
                        del st.session_state.portfolios[edit_sel]
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.warning("Cartera eliminada permanentemente.")
                        st.rerun()
                else:
                    st.info("No hay carteras guardadas actualmente.")
        
        with c2:
            st.subheader("Base de Datos (Permanente)")
            if st.session_state.portfolios:
                df_ports = pd.DataFrame([
                    {"Nombre": k, "Activos": ", ".join(v["tickers"]), "Ponderación (%)": ", ".join([f"{w*100:.1f}%" for w in v["weights"]])} 
                    for k,v in st.session_state.portfolios.items()
                ])
                st.dataframe(df_ports, use_container_width=True, hide_index=True)

    portfolios = st.session_state.get("portfolios", {})
    if not portfolios: return

    # --- TAB 2: OPTIMIZACIÓN Y COPILOT IA ---
    with tabs[1]:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        p_sel = col1.selectbox("Analizar Cartera (Optimización):", list(portfolios.keys()))
        d_start = col2.date_input("Desde", pd.to_datetime("2023-01-01"))
        d_end = col3.date_input("Hasta", pd.to_datetime("today"))
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            risk_free = st.number_input("Tasa Libre Riesgo (RF)", 0.0, 0.5, 0.04, step=0.01)
        with c_opt2:
            target = st.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])
            
        if st.button("Ejecutar Optimización"):
            with st.spinner("Optimizando y descargando datos..."):
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
                    fig = px.pie(w_df[w_df["Peso"]>0.001], values="Peso", names="Activo", title="Asignación", hole=0.4, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Error al optimizar.")
            else: st.error("Error en datos.")

        if 'last_opt_res' in st.session_state:
            st.markdown("---")
            if st.button("🧠 Analizar Portafolio con IA"):
                if not st.session_state.get("preferred_ai"):
                    st.warning("⚠️ Ingresa tu API Key (OpenAI o Gemini) en el menú lateral.")
                else:
                    with st.spinner(f"Consultando a la IA ({st.session_state.preferred_ai})..."):
                        try:
                            data_str = f"Retorno: {st.session_state['last_opt_res']['expected_return']:.2f}, Volatilidad: {st.session_state['last_opt_res']['volatility']:.2f}. Activos: {st.session_state['last_opt_res']['tickers']}."
                            prompt = f"Actúa como un asesor financiero institucional. Analiza este portafolio: {data_str}. ¿Cuáles son los riesgos y fortalezas de esta distribución?"
                            
                            if st.session_state.preferred_ai == "OpenAI":
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                response = client.chat.completions.create(
                                    model=st.session_state.get('openai_model', 'gpt-4o'),
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                st.info(response.choices[0].message.content)
                            elif st.session_state.preferred_ai == "Gemini":
                                genai.configure(api_key=st.session_state.gemini_api_key)
                                model = genai.GenerativeModel(st.session_state.gemini_model)
                                response = model.generate_content(prompt)
                                st.info(response.text)
                        except Exception as e: 
                            st.error(f"Error con API de IA: {e}")

    # --- TAB 3: FORECAST ---
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
                fig.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]), y=np.concatenate([p95, p05[::-1]]), fill='toself', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'))
                fig.add_trace(go.Scatter(x=x_axis, y=np.mean(paths, axis=1), mode='lines', name='Media', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='Pesimista (5%)', line=dict(color='#EF553B', dash='dash')))
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  MÓDULO DE RENTA FIJA (BONOS, CURVA Y TEST DE ESTRÉS)
# ═══════════════════════════════════════════════════════════════════════════
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
                            start_d = (pd.to_datetime("today") - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                            end_d = pd.to_datetime("today").strftime("%Y-%m-%d")
                            df_hist = client.get_serie_historica(t, start_d, end_d)
                            if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                                last_price = df_hist["ultimoPrecio"].iloc[-1]
                                fetched_bonds.append({
                                    "Bono": t,
                                    "Cupón (%)": 5.0, # Valores a completar por el usuario
                                    "YTM (%)": 10.0,
                                    "Años a Venc.": 3.0,
                                    "Nominal Invertido": 10000,
                                    "Precio IOL (Ref.)": last_price
                                })
                        except Exception as e:
                            st.warning(f"No se pudo obtener {t}: {e}")
                
                if fetched_bonds:
                    st.session_state.bonds_data = pd.DataFrame(fetched_bonds)
                    st.success("✅ Datos importados correctamente. Ajusta el Cupón, YTM y Vencimiento.")

    if 'bonds_data' not in st.session_state:
        st.session_state.bonds_data = pd.DataFrame({
            "Bono": ["Bono Corto", "Bono Medio", "Bono Largo"],
            "Cupón (%)": [3.0, 4.5, 6.0],
            "YTM (%)": [4.0, 5.0, 6.5],
            "Años a Venc.": [2.0, 5.0, 10.0],
            "Nominal Invertido": [100000, 150000, 50000]
        })

    edited_bonds = st.data_editor(st.session_state.bonds_data, num_rows="dynamic", use_container_width=True)
    st.session_state.bonds_data = edited_bonds

    tabs = st.tabs(["📊 Análisis e Inmunización", "📉 Sensibilidad (Test de Estrés)", "📈 Curva de Rendimiento", "💬 Chat IA Especializado"])

    results = []
    total_investment = 0
    port_mac_dur, port_mod_dur, port_convexity = 0, 0, 0
    
    for _, row in edited_bonds.iterrows():
        try:
            p, macd, modd, conv = calc_bond_metrics(100, row["Cupón (%)"]/100, row["YTM (%)"]/100, row["Años a Venc."])
            weight = row["Nominal Invertido"]
            total_investment += weight
            results.append({"Bono": row["Bono"], "Precio Calc.": p, "Mac. Dur": macd, "Mod. Dur": modd, "Convexidad": conv, "Peso": weight})
        except: pass

    if total_investment > 0 and results:
        df_res = pd.DataFrame(results)
        df_res["Peso %"] = df_res["Peso"] / total_investment
        port_mac_dur = (df_res["Mac. Dur"] * df_res["Peso %"]).sum()
        port_mod_dur = (df_res["Mod. Dur"] * df_res["Peso %"]).sum()
        port_convexity = (df_res["Convexidad"] * df_res["Peso %"]).sum()

    with tabs[0]:
        st.subheader("Métricas de Riesgo del Portafolio")
        if total_investment > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Duración Macaulay (Años)", f"{port_mac_dur:.2f}", help="Centro de gravedad de los flujos. Mide riesgo de reinversión.")
            c2.metric("Duración Modificada", f"{port_mod_dur:.2f}", help="Sensibilidad lineal del precio ante cambios en la tasa.")
            c3.metric("Convexidad", f"{port_convexity:.2f}", help="Margen de error de la duración. A mayor convexidad, mejor comportamiento ante shocks.")
            st.dataframe(df_res[["Bono", "Precio Calc.", "Mac. Dur", "Mod. Dur", "Convexidad"]].style.format("{:.2f}", subset=["Precio Calc.", "Mac. Dur", "Mod. Dur", "Convexidad"]))

            st.markdown("---")
            st.subheader("🛡️ Inmunización de Portafolio")
            horizonte = st.slider("Tu Horizonte de Inversión Objetivo (Años)", 0.5, 30.0, 5.0, 0.5)
            gap = port_mac_dur - horizonte
            if abs(gap) < 0.2: st.success(f"✅ **Inmunizado**. La Duración Macaulay ({port_mac_dur:.2f}) coincide con tu horizonte. Riesgo de precio y reinversión cancelados.")
            elif gap > 0: st.warning(f"⚠️ **Riesgo de Tasa (Precio)**. La Duración ({port_mac_dur:.2f}) es mayor a tu horizonte ({horizonte}). Vulnerable a subidas de tasas.")
            else: st.info(f"ℹ️ **Riesgo de Reinversión**. La Duración ({port_mac_dur:.2f}) es menor a tu horizonte ({horizonte}). Vulnerable a bajadas de tasas.")

    with tabs[1]:
        st.subheader("Test de Estrés de Tasas (Aproximación de Taylor)")
        st.markdown("Calcula el impacto en el precio de tus bonos ante un shock (movimiento brusco) en la curva de tasas de interés.")
        
        shock_bps = st.slider("Shock en Tasas (Puntos Básicos - bps)", min_value=-300, max_value=300, value=100, step=10, help="100 bps = 1%")
        shock_pct = shock_bps / 10000.0  
        
        if total_investment > 0:
            df_stress = df_res.copy()
            df_stress["Cambio Estimado (%)"] = (-df_stress["Mod. Dur"] * shock_pct + 0.5 * df_stress["Convexidad"] * (shock_pct**2)) * 100
            df_stress["Nuevo Precio Est."] = df_stress["Precio Calc."] * (1 + df_stress["Cambio Estimado (%)"]/100)
            
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.dataframe(df_stress[["Bono", "Cambio Estimado (%)", "Nuevo Precio Est."]].style.format("{:.2f}", subset=["Cambio Estimado (%)", "Nuevo Precio Est."]))
                port_price_change = (-port_mod_dur * shock_pct) + (0.5 * port_convexity * (shock_pct**2))
                st.info(f"🏦 **Impacto en Cartera:** Un shock de **{shock_bps} bps** alteraría el valor del portafolio en un **{port_price_change * 100:.2f}%**.")
            with col_s2:
                fig_stress = px.bar(df_stress, x="Bono", y="Cambio Estimado (%)", title="Variación Porcentual del Precio", color="Cambio Estimado (%)", color_continuous_scale="RdYlGn")
                fig_stress.update_layout(template="plotly_dark")
                st.plotly_chart(fig_stress, use_container_width=True)

    with tabs[2]:
        st.subheader("Curva de Rendimiento (Yield Curve)")
        if not edited_bonds.empty:
            df_curve = edited_bonds.sort_values(by="Años a Venc.")
            fig = px.line(df_curve, x="Años a Venc.", y="YTM (%)", markers=True, text="Bono", title="Estructura Temporal de Tasas del Portafolio")
            fig.update_traces(textposition="top center")
            fig.update_layout(template="plotly_dark", yaxis_title="Yield to Maturity (%)", xaxis_title="Plazo (Años)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("💬 Asistente IA Institucional (Renta Fija)")
        if not st.session_state.get('preferred_ai'):
            st.warning("⚠️ Configura una API Key (OpenAI o Gemini) en la barra lateral para conversar con la IA.")
        else:
            if "fi_messages" not in st.session_state: st.session_state.fi_messages = []
            for msg in st.session_state.fi_messages: 
                st.chat_message(msg["role"]).write(msg["content"])
                
            if prompt := st.chat_input("Ej: ¿Estoy cubierto si sube la tasa?"):
                st.session_state.fi_messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                port_price_change = ((-port_mod_dur * 0.01) + (0.5 * port_convexity * (0.01**2))) * 100
                context = f"Contexto actual: Macaulay={port_mac_dur:.2f} años, Modificada={port_mod_dur:.2f}, Convexidad={port_convexity:.2f}. Si la tasa sube 100 bps (+1%), el portafolio caería un {port_price_change:.2f}%."
                
                try:
                    if st.session_state.preferred_ai == "OpenAI":
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        messages_for_api = [{"role": "system", "content": f"Eres experto en Bonos. Responde basándote en: {context}"}]
                        messages_for_api.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.fi_messages[-5:]])
                        response = client.chat.completions.create(model=st.session_state.get('openai_model', 'gpt-4o'), messages=messages_for_api)
                        reply = response.choices[0].message.content
                    elif st.session_state.preferred_ai == "Gemini":
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        model = genai.GenerativeModel(st.session_state.gemini_model)
                        hist = [{'role': ('user' if m['role']=='user' else 'model'), 'parts': [m['content']]} for m in st.session_state.fi_messages[-5:-1]]
                        chat = model.start_chat(history=hist)
                        full_prompt = f"[Contexto Matemático: {context}]\n\n{prompt}"
                        reply = chat.send_message(full_prompt).text

                    st.session_state.fi_messages.append({"role": "assistant", "content": reply})
                    st.chat_message("assistant").write(reply)
                except Exception as e:
                    st.error(f"Error de API: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  OTROS EXPLORADORES E IA GENERAL
# ═══════════════════════════════════════════════════════════════════════════
def page_yahoo_explorer():
    st.title("🌎 Explorador de Mercado (Yahoo Finance)")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    with c2: period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
    if not ticker: return
    with st.spinner(f"Descargando datos..."):
        try:
            search_ticker = ticker
            stock = yf.Ticker(search_ticker)
            hist = stock.history(period=period)
            if hist.empty:
                search_ticker = ticker + ".BA"
                stock = yf.Ticker(search_ticker)
                hist = stock.history(period=period)
            if hist.empty: st.error("No hay datos."); return
            info = stock.info
            st.subheader(f"{info.get('longName', ticker)} ({search_ticker})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precio", f"${info.get('currentPrice', hist['Close'].iloc[-1]):,.2f}")
            m2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            m3.metric("Beta", info.get('beta', 'N/A'))
            m4.metric("Sector", info.get('sector', 'N/A'))
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(title="Evolución", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")

def page_event_analyzer():
    st.header("📰 Analizador de Noticias con IA")
    if not st.session_state.get('preferred_ai'): 
        st.warning("⚠️ Configure una API Key (OpenAI o Gemini) en la barra lateral.")
        return
        
    news_text = st.text_area("Pega la noticia aquí:", height=150)
    if st.button("🤖 Analizar"):
        with st.spinner(f"Analizando con {st.session_state.preferred_ai}..."):
            try:
                prompt = f"Analiza financieramente la siguiente noticia y destaca los puntos clave, el impacto en los mercados y posibles estrategias:\n\n'{news_text}'"
                if st.session_state.preferred_ai == "OpenAI":
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    response = client.chat.completions.create(
                        model=st.session_state.openai_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
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
        st.warning("⚠️ Configure una API Key (OpenAI o Gemini) en la barra lateral.")
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
    st.markdown("""
    Describe tu objetivo de inversión en lenguaje natural. La IA actuará como un analista cuantitativo, 
    traduciendo tu idea en un conjunto de **restricciones y universos de activos (Acciones y Bonos)** 
    listos para ser usados en los módulos de optimización.
    """)
    
    if not st.session_state.get('preferred_ai'): 
        st.warning("⚠️ Configura una API Key (OpenAI o Gemini) en la barra lateral para usar el asistente.")
        return

    user_strategy_prompt = st.text_area(
        "Describe tu estrategia de inversión:",
        height=120,
        placeholder="Ej: 'Busco un portafolio conservador. Quiero un 60% en bonos soberanos argentinos (ley NY y local) y un 40% en acciones tecnológicas grandes de USA que no estén en burbuja (evitar P/E altísimos).'"
    )

    if st.button("Traducir Estrategia a Filtros", type="primary"):
        if not user_strategy_prompt:
            st.warning("Por favor, describe tu estrategia.")
        else:
            system_prompt = """
            Eres un experto en finanzas cuantitativas e institucionales. Tu tarea es traducir la estrategia de inversión de un usuario en un conjunto de restricciones JSON para un modelo de optimización. 
            Debes considerar tanto Renta Variable (Acciones) como Renta Fija (Bonos).
            
            Las claves exactas del JSON que debes generar son:
            - 'k_assets' (integer): Número total de activos en el portafolio.
            - 'asset_allocation' (dict): Pesos sugeridos. Ej: {"stocks": 0.40, "bonds": 0.60}.
            - 'beta_range' (array of two floats): [min_beta, max_beta] para el riesgo de las acciones.
            - 'pe_range' (array of two floats): [min_pe, max_pe] para valuación de acciones.
            - 'duration_range' (array of two floats): [min_duration, max_duration] en años para el riesgo de tasa de los bonos.
            - 'universe_stocks' (list of strings): Lista de tickers reales (ej. AAPL, MSFT) que encajen.
            - 'universe_bonds' (list of strings): Lista de tickers de bonos reales (ej. TLT, AL30, GD30) que encajen.

            Analiza la descripción del usuario y responde **SOLAMENTE CON UN BLOQUE DE CÓDIGO JSON VÁLIDO Y NADA MÁS**. No agregues texto introductorio.
            """
            
            with st.spinner(f"IA Quant ({st.session_state.preferred_ai}) analizando tu estrategia..."):
                try:
                    raw_response = ""
                    if st.session_state.preferred_ai == "OpenAI":
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        response = client.chat.completions.create(
                            model=st.session_state.openai_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_strategy_prompt}
                            ],
                            temperature=0.1
                        )
                        raw_response = response.choices[0].message.content
                        
                    elif st.session_state.preferred_ai == "Gemini":
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        model = genai.GenerativeModel(st.session_state.gemini_model)
                        full_prompt = f"{system_prompt}\n\n**Descripción del Usuario:**\n{user_strategy_prompt}"
                        response = model.generate_content(
                            full_prompt, 
                            generation_config=genai.types.GenerationConfig(temperature=0.1)
                        )
                        raw_response = response.text
                    
                    if raw_response:
                        # Limpiar la respuesta para asegurar que es un JSON (quita los bloques markdown de código si los hay)
                        json_text = re.search(r'\{.*\}', raw_response, re.DOTALL)
                        
                        if json_text:
                            suggested_params = json.loads(json_text.group(0))
                            
                            st.subheader("📊 Filtros Cuantitativos Sugeridos")
                            
                            # Mostrar de forma visual amigable
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Distribución de Activos (Allocation):**")
                                st.json(suggested_params.get("asset_allocation", {}))
                                st.write("**Restricciones Renta Variable:**")
                                st.write(f"- Beta: {suggested_params.get('beta_range')} (Riesgo Mercado)")
                                st.write(f"- P/E Ratio: {suggested_params.get('pe_range')} (Valuación)")
                                st.write("**Universo Acciones:**", ", ".join(suggested_params.get("universe_stocks", [])))
                            
                            with col2:
                                st.write("**Restricciones Renta Fija:**")
                                st.write(f"- Duración (Años): {suggested_params.get('duration_range')} (Riesgo Tasa)")
                                st.write("**Universo Bonos:**", ", ".join(suggested_params.get("universe_bonds", [])))
                            
                            st.markdown("---")
                            st.write("**JSON Crudo (Para copiar a motores de optimización):**")
                            st.code(json.dumps(suggested_params, indent=4), language="json")
                            
                            st.success("✅ ¡Listo! Puedes copiar estos tickers y usarlos en el **Dashboard Corporativo** (para optimizar) o en **Renta Fija**.")
                        else:
                            st.error("No se encontró un JSON válido en la respuesta.")
                            st.write(raw_response)
                            
                except Exception as e:
                    st.error(f"Error procesando la IA: {str(e)}")
                    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()

st.sidebar.title("Configuración y Accesos")

with st.sidebar.expander("🤖 IA (OpenAI / Copilot)", expanded=True):
    st.markdown("<small>Usado para Análisis Generales e Institucionales</small>", unsafe_allow_html=True)
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('openai_api_key', ''))
    st.session_state.openai_model = st.selectbox("Modelo OpenAI", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

with st.sidebar.expander("🧠 IA (Gemini)", expanded=False):
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get('gemini_api_key', ''))
    # Selección con los modelos más recientes de Gemini
    st.session_state.gemini_model = st.selectbox("Modelo Gemini", [
        "gemini-2.5-flash", 
        "gemini-2.0-flash", 
        "gemini-1.5-pro", 
        "gemini-1.5-flash", 
        "gemini-pro"
    ])

# Validación dinámica del motor de IA preferido según las llaves ingresadas
st.sidebar.markdown("---")
available_ais = []
if OPENAI_OK and st.session_state.get('openai_api_key'): available_ais.append("OpenAI")
if GEMINI_OK and st.session_state.get('gemini_api_key'): available_ais.append("Gemini")

if available_ais:
    st.session_state.preferred_ai = st.sidebar.radio("✨ Motor IA Activo", available_ais)
else:
    st.session_state.preferred_ai = None
    st.sidebar.warning("⚠️ Ingresa una API Key para usar IA.")

with st.sidebar.expander("🏦 IOL"):
    user_iol = st.text_input("Usuario IOL")
    pass_iol = st.text_input("Pass IOL", type="password")
    if st.button("Conectar"): st.session_state.iol_username, st.session_state.iol_password = user_iol, pass_iol

st.sidebar.markdown("---")
opciones = [
    "Inicio", 
    "📊 Dashboard Corporativo", 
    "🏛️ Renta Fija (Bonos y Curvas)", 
    "🧠 Asistente Quant (Estrategia IA)",  # <--- NUEVA OPCIÓN AÑADIDA AQUÍ
    "🏦 Explorador IOL API", 
    "🌎 Explorador Global (Yahoo)", 
    "🔭 Modelos Avanzados (Forecast)", 
    "📰 Analizador Eventos (IA)", 
    "💬 Chat IA General"
]

sel = st.sidebar.radio("Navegación", opciones, index=opciones.index(st.session_state.selected_page) if st.session_state.selected_page in opciones else 0)

if sel != st.session_state.selected_page: st.session_state.selected_page = sel; st.rerun()

if sel == "Inicio": st.title("BPNos - Finanzas Corporativas"); st.info("Seleccione un módulo en la barra lateral.")
elif sel == "📊 Dashboard Corporativo": page_corporate_dashboard()
elif sel == "🏛️ Renta Fija (Bonos y Curvas)": page_fixed_income()
elif sel == "🧠 Asistente Quant (Estrategia IA)": page_ai_strategy_assistant() # <--- NUEVO RUTEO AÑADIDO AQUÍ
elif sel == "🏦 Explorador IOL API": page_iol_explorer()
elif sel == "🌎 Explorador Global (Yahoo)": page_yahoo_explorer()
elif sel == "🔭 Modelos Avanzados (Forecast)": page_forecast()
elif sel == "📰 Analizador Eventos (IA)": page_event_analyzer()
elif sel == "💬 Chat IA General": page_chat_general()
