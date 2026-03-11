import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings("ignore")

# ── Intento de importar statsmodels (SARIMAX) ──────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

# ── Intento de importar Prophet ────────────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

# ── Intento de importar Tbats ──────────────────────────────────────────────
try:
    from tbats import TBATS
    TBATS_OK = True
except ImportError:
    TBATS_OK = False

# ═══════════════════════════════════════════════════════════════════════════
#  GOOGLE GEMINI  –  cliente liviano (sin SDK externo)
# ═══════════════════════════════════════════════════════════════════════════
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"

def gemini_generate(prompt: str, api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Llama a la API REST de Gemini y devuelve el texto generado."""
    url = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024}
    }
    try:
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Error Gemini: {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Descarga precios intentando primero con IOL API y, si no, usa Yahoo Finance."""
    if not tickers:
        return pd.DataFrame()

    all_prices = {}
    yf_tickers = []
    
    # 1. Intentar obtener el cliente autenticado de IOL
    client = None
    try:
        from iol_client import get_iol_client
        client = get_iol_client()
    except ImportError:
        pass

    # 2. Iterar por tickers y buscar en IOL
    for ticker in tickers:
        fetched = False
        if client:
            # Limpiamos el ticker por si el usuario le puso sufijos (ej: AL30.BA -> AL30)
            simbolo_iol = ticker.split(".")[0].upper()
            try:
                # Usamos ajustada="ajustada", que ya tiene el fallback a sinAjustar por dentro
                df_hist = client.get_serie_historica(simbolo_iol, start, end, ajustada="ajustada", mercado="bCBA")
                if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                    s = df_hist["ultimoPrecio"].rename(ticker)
                    # Remover zona horaria para compatibilidad estricta
                    if s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    all_prices[ticker] = s
                    fetched = True
            except Exception:
                pass
        
        # Si no se encontró en IOL, lo mandamos a la lista de Yahoo Finance
        if not fetched:
            yf_tickers.append(ticker)

    # 3. Fallback a Yahoo Finance para activos internacionales o si IOL falla
    if yf_tickers:
        try:
            raw = yf.download(yf_tickers, start=start, end=end, auto_adjust=True, progress=False)
            if not raw.empty:
                if "Close" in raw.columns:
                    prices = raw["Close"]
                else:
                    prices = raw
                    
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=yf_tickers[0])
                
                # Remover zona horaria para compatibilidad estricta
                if prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)
                
                for col in prices.columns:
                    all_prices[str(col)] = prices[col]
        except Exception as e:
            st.error(f"Error al descargar desde Yahoo Finance: {e}")

    # 4. Consolidar el DataFrame
    if not all_prices:
        return pd.DataFrame()

    prices_df = pd.concat(all_prices.values(), axis=1)
    prices_df.columns = list(all_prices.keys())
    prices_df.dropna(how="all", inplace=True)
    prices_df.ffill(inplace=True) # Rellena huecos por diferencias de feriados
    
    return prices_df


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller: devuelve dict con estadístico y p-valor."""
    clean = series.dropna()
    if len(clean) < 20:
        return {"statistic": None, "p_value": None, "stationary": None}
    result = adfuller(clean, autolag="AIC")
    return {"statistic": round(result[0], 4), "p_value": round(result[1], 4), "stationary": result[1] < 0.05}


def granger_test(endog: pd.Series, exog: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Prueba de causalidad de Granger entre dos series."""
    df_g = pd.concat([endog, exog], axis=1).dropna()
    df_g.columns = ["target", "exog"]
    try:
        gc = grangercausalitytests(df_g, maxlag=max_lag, verbose=False)
        rows = []
        for lag, res in gc.items():
            p_val = round(res[0]["ssr_ftest"][1], 4)
            rows.append({"Lag": lag, "p-valor (F-test)": p_val, "Significativo": "✅" if p_val < 0.05 else "❌"})
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# ═══════════════════════════════════════════════════════════════════════════
#  MODELOS DE PRONÓSTICO
# ═══════════════════════════════════════════════════════════════════════════

def run_sarimax(target: pd.Series, exog_df: pd.DataFrame | None,
                order=(1,1,1), seasonal_order=(0,0,0,0),
                horizon: int = 30) -> dict:
    """Ajusta SARIMAX y devuelve pronóstico + métricas."""
    target = target.dropna()

    # Alinear exógenas si las hay
    if exog_df is not None and not exog_df.empty:
        exog_df = exog_df.reindex(target.index).ffill().dropna(how="all")
        common_idx = target.index.intersection(exog_df.index)
        target = target.loc[common_idx]
        exog_df = exog_df.loc[common_idx]
        exog_fit = exog_df
        # Proyección futura de exógenas (últimos valores hacia adelante)
        last_exog = exog_df.iloc[[-1]]
        exog_future = pd.concat([last_exog] * horizon, ignore_index=True)
    else:
        exog_fit = None
        exog_future = None

    # Ajuste
    model = SARIMAX(target, exog=exog_fit,
                    order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    # Pronóstico
    fc = fit.get_forecast(steps=horizon, exog=exog_future)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)

    # Fechas futuras
    last_date = target.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)
    fc_mean.index = future_dates
    fc_ci.index   = future_dates

    # Métricas in-sample
    fitted = fit.fittedvalues
    resid  = target - fitted
    mae    = np.mean(np.abs(resid))
    rmse   = np.sqrt(np.mean(resid**2))
    aic    = fit.aic

    return {
        "model": "SARIMAX",
        "fitted": fitted,
        "forecast": fc_mean,
        "ci_lower": fc_ci.iloc[:, 0],
        "ci_upper": fc_ci.iloc[:, 1],
        "target": target,
        "metrics": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "AIC": round(aic, 2)},
        "summary": fit.summary().as_text()
    }


def run_prophet(target: pd.Series, exog_df: pd.DataFrame | None,
                horizon: int = 30, country_holidays: str | None = None) -> dict:
    """Ajusta Prophet (con regresores exógenos opcionales) y devuelve pronóstico."""
    df_p = pd.DataFrame({"ds": target.index, "y": target.values})

    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    if country_holidays:
        m.add_country_holidays(country_name=country_holidays)

    regressors = []
    if exog_df is not None and not exog_df.empty:
        exog_aligned = exog_df.reindex(target.index).ffill()
        for col in exog_aligned.columns:
            m.add_regressor(col)
            df_p[col] = exog_aligned[col].values
        regressors = list(exog_aligned.columns)

    m.fit(df_p)

    future = m.make_future_dataframe(periods=horizon, freq="B")
    if regressors:
        for col in regressors:
            all_vals = pd.concat([exog_df[col], pd.Series([exog_df[col].iloc[-1]] * horizon)])
            future[col] = all_vals.values[:len(future)]

    forecast = m.predict(future)
    hist_part = forecast[forecast["ds"] <= target.index[-1]]
    fut_part  = forecast[forecast["ds"] >  target.index[-1]]

    fitted = pd.Series(hist_part["yhat"].values, index=pd.to_datetime(hist_part["ds"].values))
    fc_mean = pd.Series(fut_part["yhat"].values, index=pd.to_datetime(fut_part["ds"].values))
    ci_lo   = pd.Series(fut_part["yhat_lower"].values, index=pd.to_datetime(fut_part["ds"].values))
    ci_hi   = pd.Series(fut_part["yhat_upper"].values, index=pd.to_datetime(fut_part["ds"].values))

    resid = target.values - fitted.reindex(target.index).values
    mae   = np.nanmean(np.abs(resid))
    rmse  = np.sqrt(np.nanmean(resid**2))

    return {
        "model": "Prophet",
        "fitted": fitted,
        "forecast": fc_mean,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "target": target,
        "metrics": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "AIC": "N/A"},
        "prophet_obj": m,
        "prophet_forecast": forecast
    }

def run_tbats(target: pd.Series, horizon: int = 30, seasonal_periods: list = None) -> dict:
    """Ajusta TBATS y devuelve pronóstico. (Modelo Univariado puro)."""
    target = target.dropna()

    if not seasonal_periods:
        seasonal_periods = None

    estimator = TBATS(
        seasonal_periods=seasonal_periods,
        use_arma_errors=True,
        use_box_cox=None,
        use_trend=None,
        use_damped_trend=None
    )
    model = estimator.fit(target.values)

    fc_mean, conf_int = model.forecast(steps=horizon, confidence_level=0.95)

    last_date = target.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)

    fc_series = pd.Series(fc_mean, index=future_dates)
    ci_lower = pd.Series(conf_int["lower_bound"], index=future_dates)
    ci_upper = pd.Series(conf_int["upper_bound"], index=future_dates)

    fitted = pd.Series(model.y_hat, index=target.index)

    resid = target.values - model.y_hat
    mae = np.nanmean(np.abs(resid))
    rmse = np.sqrt(np.nanmean(resid**2))
    aic = model.aic

    return {
        "model": "TBATS",
        "fitted": fitted,
        "forecast": fc_series,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "target": target,
        "metrics": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "AIC": round(aic, 2)},
        "summary": f"Modelo TBATS Ajustado: {str(model)}"
    }

# ═══════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = {
    "actual":    "#E8E8E8",
    "fitted":    "#64B5F6",
    "forecast":  "#FFD54F",
    "ci":        "rgba(255,213,79,0.15)",
    "exog":      ["#80CBC4", "#F48FB1", "#CE93D8", "#A5D6A7", "#FFAB91"],
    "bg":        "#0D1117",
    "grid":      "#1F2937",
    "text":      "#C9D1D9",
}

def build_forecast_chart(result: dict, target_label: str) -> go.Figure:
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=result["target"].index, y=result["target"].values,
        name="Histórico", line=dict(color=PALETTE["actual"], width=1.5), mode="lines"
    ))
    # Ajuste in-sample
    fitted = result["fitted"].reindex(result["target"].index)
    fig.add_trace(go.Scatter(
        x=fitted.index, y=fitted.values,
        name="Ajuste modelo", line=dict(color=PALETTE["fitted"], width=1, dash="dot"), mode="lines"
    ))
    # Banda de confianza
    fig.add_trace(go.Scatter(
        x=list(result["forecast"].index) + list(result["forecast"].index[::-1]),
        y=list(result["ci_upper"]) + list(result["ci_lower"][::-1]),
        fill="toself", fillcolor=PALETTE["ci"],
        line=dict(color="rgba(0,0,0,0)"), name="IC 95 %", showlegend=True
    ))
    # Pronóstico
    fig.add_trace(go.Scatter(
        x=result["forecast"].index, y=result["forecast"].values,
        name=f"Pronóstico ({result['model']})",
        line=dict(color=PALETTE["forecast"], width=2.5), mode="lines+markers",
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=dict(text=f"<b>{target_label}</b> – Pronóstico {result['model']}",
                   font=dict(size=18, color=PALETTE["text"])),
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=PALETTE["grid"], borderwidth=1),
        xaxis=dict(gridcolor=PALETTE["grid"], showgrid=True),
        yaxis=dict(gridcolor=PALETTE["grid"], showgrid=True, title=target_label),
        hovermode="x unified",
        height=480,
    )
    return fig


def build_exog_chart(prices: pd.DataFrame, target_col: str, exog_cols: list[str]) -> go.Figure:
    """Gráfico de correlaciones entre target y exógenas (retornos normalizados)."""
    returns = prices.pct_change().dropna()
    norm = (returns - returns.mean()) / returns.std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm.index, y=norm[target_col],
                             name=target_col,
                             line=dict(color=PALETTE["actual"], width=2)))
    for i, col in enumerate(exog_cols):
        if col in norm.columns:
            color = PALETTE["exog"][i % len(PALETTE["exog"])]
            fig.add_trace(go.Scatter(x=norm.index, y=norm[col],
                                     name=col,
                                     line=dict(color=color, width=1.2, dash="dash")))

    fig.update_layout(
        title="<b>Retornos normalizados</b> – Target vs Exógenas",
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        xaxis=dict(gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        height=340
    )
    return fig


def build_corr_heatmap(prices: pd.DataFrame, cols: list[str]) -> go.Figure:
    sub = prices[cols].pct_change().dropna()
    corr = sub.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="<b>Correlación de retornos</b>"
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]), height=380
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  ANÁLISIS GEMINI
# ═══════════════════════════════════════════════════════════════════════════

def build_gemini_prompt(result: dict, exog_tickers: list[str], target_label: str,
                        corr_info: str, granger_info: str) -> str:
    fc_vals  = result["forecast"].round(4).to_dict()
    metrics  = result["metrics"]
    last_val = float(result["target"].iloc[-1])
    fc_end   = float(result["forecast"].iloc[-1])
    chg      = (fc_end - last_val) / last_val * 100

    prompt = f"""Eres un analista financiero cuantitativo experto en mercados latinoamericanos.

## Contexto del pronóstico
- **Activo objetivo:** {target_label}
- **Modelo utilizado:** {result['model']}
- **Horizonte:** {len(result['forecast'])} días hábiles
- **Variables exógenas incluidas:** {', '.join(exog_tickers) if exog_tickers else 'Ninguna'}

## Métricas del modelo
- MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | AIC: {metrics['AIC']}

## Resultados
- Último precio histórico: {last_val:.4f}
- Precio pronosticado al final del horizonte: {fc_end:.4f}
- Cambio esperado: {chg:+.2f} %
- Primeras y últimas fechas del pronóstico: {list(fc_vals.keys())[0]} → {list(fc_vals.keys())[-1]}

## Correlaciones con exógenas
{corr_info}

## Causalidad de Granger (resumen)
{granger_info}

## Tarea
1. Interpreta la calidad del ajuste del modelo basándote en las métricas.
2. Analiza el pronóstico: ¿es alcista, bajista o neutro? ¿Con qué nivel de confianza?
3. Explica el rol de cada variable exógena y si su inclusión mejora el modelo. (Si el modelo es TBATS, ignora su inclusión directa e interpreta el contexto general del mercado).
4. Identifica riesgos clave para el pronóstico.
5. Proporciona una recomendación de posicionamiento (comprar / mantener / vender / esperar) con justificación breve.
6. Sé conciso: máximo 400 palabras. Usa viñetas cuando sea útil.
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA PRINCIPAL DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════

def page_forecast():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
    .fc-header { font-family:'Syne',sans-serif; font-weight:800; font-size:2rem;
                 background:linear-gradient(90deg,#FFD54F,#80CBC4); -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent; margin-bottom:.25rem; }
    .fc-sub    { font-family:'Space Mono',monospace; font-size:.78rem; color:#6B7280; letter-spacing:.06em; }
    .metric-box{ background:#161B22; border:1px solid #30363D; border-radius:8px;
                 padding:.75rem 1.1rem; text-align:center; }
    .metric-val{ font-family:'Space Mono',monospace; font-size:1.4rem; color:#FFD54F; font-weight:700; }
    .metric-lbl{ font-size:.7rem; color:#6B7280; letter-spacing:.05em; text-transform:uppercase; margin-top:.2rem; }
    .section   { background:#0D1117; border:1px solid #21262D; border-radius:10px; padding:1.2rem; margin-bottom:1rem; }
    .tag-ok    { background:#064E3B; color:#6EE7B7; border-radius:4px; padding:2px 8px; font-size:.72rem; }
    .tag-no    { background:#450A0A; color:#FCA5A5; border-radius:4px; padding:2px 8px; font-size:.72rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="fc-header">🔭 Módulo de Pronóstico con Variables Exógenas</p>', unsafe_allow_html=True)
    st.markdown('<p class="fc-sub">SARIMAX · PROPHET · TBATS · GEMINI AI · GRANGER CAUSALITY</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Verificación de dependencias ─────────────────────────────────────
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        label = '<span class="tag-ok">statsmodels ✓</span>' if STATSMODELS_OK else '<span class="tag-no">statsmodels ✗</span>'
        st.markdown(f"**SARIMAX:** {label}", unsafe_allow_html=True)
    with col_d2:
        label = '<span class="tag-ok">prophet ✓</span>' if PROPHET_OK else '<span class="tag-no">prophet ✗</span>'
        st.markdown(f"**Prophet:** {label}", unsafe_allow_html=True)
    with col_d3:
        label = '<span class="tag-ok">tbats ✓</span>' if TBATS_OK else '<span class="tag-no">tbats ✗</span>'
        st.markdown(f"**TBATS:** {label}", unsafe_allow_html=True)
    with col_d4:
        gemini_key = st.session_state.get("gemini_api_key", "")
        label = '<span class="tag-ok">key cargada ✓</span>' if gemini_key else '<span class="tag-no">sin key ✗</span>'
        st.markdown(f"**Gemini:** {label}", unsafe_allow_html=True)

    if not STATSMODELS_OK and not PROPHET_OK and not TBATS_OK:
        st.error("Instala al menos uno: `pip install statsmodels`, `pip install prophet` o `pip install tbats`")
        return

    st.markdown("---")

    # ── Panel de configuración ───────────────────────────────────────────
    with st.expander("⚙️ Configuración del pronóstico", expanded=True):

        c1, c2 = st.columns(2)
        with c1:
            target_ticker = st.text_input("🎯 Activo a pronosticar (Ticker IOL o Yahoo)", value="AL30",
                                          help="Ej: AL30, GD30, GGAL, BMA, AAPL")
        with c2:
            exog_input = st.text_input(
                "📎 Variables exógenas (tickers separados por coma)",
                value="GGAL",
                help="Ej: GGAL, DXY=F – dejar vacío para modelo univariado"
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            start_date = st.date_input("📅 Inicio histórico", value=datetime(2022, 1, 1))
        with c4:
            end_date   = st.date_input("📅 Fin histórico",   value=datetime.today())
        with c5:
            horizon    = st.number_input("🔮 Horizonte (días hábiles)", min_value=5, max_value=252, value=30)

        c6, c7 = st.columns(2)
        with c6:
            available_models = [m for m, ok in [("SARIMAX", STATSMODELS_OK), ("Prophet", PROPHET_OK), ("TBATS", TBATS_OK)] if ok]
            model_choice = st.selectbox("🤖 Modelo", available_models)
        with c7:
            use_returns = st.checkbox("Usar retornos (en lugar de precios)", value=False,
                                      help="Recomendado si las series no son estacionarias")

        # Inicializar variables por defecto
        p, d, q, P, D, Q, S = 1, 1, 1, 0, 0, 0, 0
        seasonal_periods = []

        # Parámetros específicos según modelo
        if model_choice == "SARIMAX":
            with st.expander("🔧 Parámetros SARIMAX (p,d,q) y estacionalidad (P,D,Q,s)"):
                sc1, sc2, sc3 = st.columns(3)
                with sc1: p = st.number_input("p (AR)", 0, 5, 1)
                with sc2: d = st.number_input("d (I)",  0, 2, 1)
                with sc3: q = st.number_input("q (MA)", 0, 5, 1)
                sc4, sc5, sc6, sc7 = st.columns(4)
                with sc4: P = st.number_input("P", 0, 2, 0)
                with sc5: D = st.number_input("D", 0, 1, 0)
                with sc6: Q = st.number_input("Q", 0, 2, 0)
                with sc7: S = st.number_input("s", 0, 52, 0)
        elif model_choice == "TBATS":
            with st.expander("🔧 Parámetros TBATS (Frecuencias Estacionales)"):
                st.info("ℹ️ TBATS es un modelo puramente univariado. Las variables exógenas serán ignoradas para el pronóstico estadístico.")
                ts1, ts2 = st.columns(2)
                with ts1: tbats_s1 = st.number_input("Estacionalidad 1 (ej: 5 días laborables)", 0, 365, 5)
                with ts2: tbats_s2 = st.number_input("Estacionalidad 2 (ej: 21 días al mes)", 0, 365, 21)
                seasonal_periods = [s for s in [tbats_s1, tbats_s2] if s > 0]

        run_granger  = st.checkbox("🧪 Ejecutar prueba de causalidad de Granger", value=True)
        max_lag_gr   = st.number_input("   Lags máximos Granger", 1, 10, 5) if run_granger else 5
        use_ai       = st.checkbox("🤖 Análisis cualitativo con Gemini AI", value=bool(gemini_key))

    run_btn = st.button("🚀 Ejecutar pronóstico", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configura los parámetros y presiona **Ejecutar pronóstico**.")
        return

    # ── Carga de datos ───────────────────────────────────────────────────
    exog_tickers = [t.strip().upper() for t in exog_input.split(",") if t.strip()] if exog_input.strip() else []
    all_tickers  = list(dict.fromkeys([target_ticker.upper()] + exog_tickers))  # sin duplicados

    with st.spinner("📡 Descargando datos desde IOL y Yahoo Finance..."):
        prices = load_prices(all_tickers, str(start_date), str(end_date))

    if prices.empty:
        st.error("No se pudieron descargar datos. Verifica que estés logueado en IOL y que los tickers sean correctos.")
        return

    target_col = target_ticker.upper()
    if target_col not in prices.columns:
        available = prices.columns.tolist()
        st.warning(f"Ticker '{target_col}' no encontrado. Columnas disponibles: {available}")
        target_col = available[0]

    target_series = prices[target_col]
    if use_returns:
        target_series = target_series.pct_change().dropna()
        prices = prices.pct_change().dropna()

    exog_available = [c for c in exog_tickers if c in prices.columns]
    exog_df = prices[exog_available] if exog_available else None

    # ── Estadísticas descriptivas ────────────────────────────────────────
    st.markdown("### 📊 Resumen de datos")
    meta_cols = st.columns(len(all_tickers))
    for i, col in enumerate(all_tickers):
        if col in prices.columns:
            s = prices[col].dropna()
            with meta_cols[i]:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val">{s.iloc[-1]:.2f}</div>
                  <div class="metric-lbl">{col}</div>
                  <div style="font-size:.7rem;color:#9CA3AF">
                    Ult. {len(s)} obs · vol {s.pct_change().std()*100:.2f}%
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Correlaciones y Granger ──────────────────────────────────────────
    corr_info    = ""
    granger_info = ""

    if exog_available:
        st.markdown("### 🔗 Relación target ↔ exógenas")
        tab_chart, tab_corr, tab_granger = st.tabs(["📈 Retornos normalizados", "🌡️ Correlación", "🧪 Granger"])

        with tab_chart:
            st.plotly_chart(build_exog_chart(prices if not use_returns else prices,
                                             target_col, exog_available),
                            use_container_width=True)

        with tab_corr:
            hmap_cols = [target_col] + exog_available
            hmap_cols = [c for c in hmap_cols if c in prices.columns]
            st.plotly_chart(build_corr_heatmap(prices, hmap_cols), use_container_width=True)
            corr_matrix = prices[hmap_cols].pct_change().dropna().corr()
            corr_info = corr_matrix.to_string()

        with tab_granger:
            if STATSMODELS_OK and run_granger:
                for exog_col in exog_available:
                    st.markdown(f"**{exog_col} → {target_col}**")
                    gc_df = granger_test(target_series, prices[exog_col], max_lag=max_lag_gr)
                    st.dataframe(gc_df, use_container_width=True, hide_index=True)
                    sig_lags = gc_df[gc_df.get("Significativo", "") == "✅"]["Lag"].tolist() if "Lag" in gc_df.columns else []
                    granger_info += f"\n{exog_col}: lags significativos {sig_lags}"
            else:
                st.info("Granger requiere statsmodels o fue desactivado.")

    # ── ADF test ─────────────────────────────────────────────────────────
    if STATSMODELS_OK:
        with st.expander("🔬 Prueba de estacionariedad (ADF)"):
            adf_cols = [target_col] + exog_available
            adf_rows = []
            for col in adf_cols:
                if col in prices.columns:
                    r = adf_test(prices[col].pct_change().dropna())
                    adf_rows.append({
                        "Serie": col,
                        "Estadístico": r["statistic"],
                        "p-valor": r["p_value"],
                        "Estacionaria": "✅ Sí" if r["stationary"] else "❌ No"
                    })
            st.dataframe(pd.DataFrame(adf_rows), use_container_width=True, hide_index=True)

    # ── Ajuste del modelo ────────────────────────────────────────────────
    st.markdown("### 🤖 Entrenamiento y pronóstico")

    with st.spinner(f"Ajustando {model_choice} (esto puede tardar unos segundos)..."):
        try:
            if model_choice == "SARIMAX":
                result = run_sarimax(
                    target_series, exog_df,
                    order=(p, d, q), seasonal_order=(P, D, Q, S),
                    horizon=horizon
                )
            elif model_choice == "TBATS":
                result = run_tbats(target_series, horizon=horizon, seasonal_periods=seasonal_periods)
            else:
                result = run_prophet(target_series, exog_df, horizon=horizon)
        except Exception as e:
            st.error(f"Error al ajustar el modelo: {e}")
            st.exception(e)
            return

    # ── Métricas ─────────────────────────────────────────────────────────
    m = result["metrics"]
    mc1, mc2, mc3 = st.columns(3)
    for box, lbl, val in zip([mc1, mc2, mc3],
                              ["MAE", "RMSE", "AIC"],
                              [m["MAE"], m["RMSE"], m["AIC"]]):
        with box:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Gráfico pronóstico ───────────────────────────────────────────────
    st.plotly_chart(build_forecast_chart(result, target_col), use_container_width=True)

    # ── Tabla de pronóstico ──────────────────────────────────────────────
    with st.expander("📋 Tabla de valores pronosticados"):
        fc_df = pd.DataFrame({
            "Fecha": result["forecast"].index,
            "Pronóstico": result["forecast"].values.round(4),
            "IC inferior (95%)": result["ci_lower"].values.round(4),
            "IC superior (95%)": result["ci_upper"].values.round(4),
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

        csv_fc = fc_df.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button("📥 Descargar pronóstico CSV", csv_fc,
                           file_name=f"forecast_{target_col}.csv", mime="text/csv")

    # ── Resumen Modelo ──────────────────────────────────────────────────
    if model_choice in ["SARIMAX", "TBATS"]:
        with st.expander(f"📜 Resumen estadístico {model_choice}"):
            st.text(result["summary"])

    # ── Análisis Gemini ──────────────────────────────────────────────────
    if use_ai:
        st.markdown("### 🧠 Análisis cualitativo – Gemini AI")
        if not gemini_key:
            st.warning("Ingresa tu Google Gemini API Key en la barra lateral para habilitar este análisis.")
        else:
            with st.spinner("Consultando Gemini..."):
                prompt_ai = build_gemini_prompt(result, exog_available,
                                                target_col, corr_info, granger_info)
                ai_response = gemini_generate(prompt_ai, gemini_key)

            st.markdown(f"""
            <div class="section">
            {ai_response.replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)

            with st.expander("🔍 Ver prompt enviado a Gemini"):
                st.text(prompt_ai)
