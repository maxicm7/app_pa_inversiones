# -*- coding: utf-8 -*-
"""
INVERSIONES PRO v2 — Plataforma integral de gestión de portafolios
-------------------------------------------------------------------
Autor: Maxi Morales

Cambios principales respecto de v1:
  • Corregidos: Risk Parity y Black-Litterman (llamaban métodos inexistentes y
    devolvían None en silencio), duration de bonos (faltaba el principal),
    dashboard que no permitía crear la primera cartera, `return` dentro de tabs
    que cortaba el render, botones anidados del Asistente Quant que nunca se
    ejecutaban, sufijo .BA forzado a todos los tickers, formatos que rompían
    con 'N/A', data_editor que perdía ediciones, tickers IOL sin fallback.
  • Nuevo: análisis de cartera vs benchmark, drawdown, correlaciones, volatilidad
    rolling, contribución al riesgo, frontera eficiente, Black-Litterman con
    views editables, restricciones de peso máximo, shrinkage Ledoit-Wolf,
    backtest con rebalanceo y costos, walk-forward fuera de muestra, Monte Carlo
    GBM / bootstrap, órdenes de rebalanceo con cantidades enteras y comisión,
    renta fija con convexidad y escenarios de tasa, chat con memoria y contexto.
"""
import os
import re
import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm as normal_dist
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════
#  IMPORTACIONES OPCIONALES
# ═══════════════════════════════════════════════════════════════════════════
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_OK = True
except ImportError:
    GSHEETS_OK = False

# Gemini: SDK nuevo (google-genai) con fallback al viejo (google-generativeai)
GEMINI_BACKEND = None
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    GEMINI_BACKEND = "new"
except ImportError:
    try:
        import google.generativeai as genai_old
        GEMINI_BACKEND = "old"
    except ImportError:
        pass
GEMINI_OK = GEMINI_BACKEND is not None

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from pypfopt import EfficientFrontier, risk_models, HRPOpt
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
    from iol_client import page_iol_explorer, get_iol_client
    IOL_MODULE_OK = True
except ImportError:
    IOL_MODULE_OK = False

    def page_iol_explorer():
        st.warning("📦 iol_client.py no encontrado.")

    def get_iol_client():
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="INVERSIONES PRO", page_icon="📈")

TD = 252  # días hábiles por año
PORTFOLIO_FILE = "portfolios_data1.json"
WORKSHEET_NAME = "portfolios"
TEMPLATE = "plotly_dark"
COLORS = {"opt": "#00CC96", "cur": "#EF553B", "ew": "#AB63FA", "bench": "#636EFA"}


def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


# `use_container_width` está deprecado en Streamlit >= 1.50 → usamos width="stretch"
W = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}


def secret(section, key=None, default=None):
    """Lee st.secrets sin romper cuando no existe secrets.toml."""
    try:
        if section not in st.secrets:
            return default
        sec = st.secrets[section]
        return sec if key is None else sec.get(key, default)
    except Exception:
        return default


SHEET_NAME = secret("google_sheets", "sheet_name", "Epre_Inversiones")
SHEET_ID = secret("google_sheets", "sheet_id", "")

# Tickers locales de BYMA (sin sufijo) → en modo "Auto" se les agrega .BA
AR_LOCAL = {
    "GGAL", "YPFD", "PAMP", "CEPU", "ALUA", "TXAR", "LOMA", "BMA", "SUPV", "TGSU2", "TGNO4",
    "CRES", "MIRG", "COME", "VALO", "EDN", "BYMA", "TRAN", "BBAR", "CVH", "HARG", "METR",
    "AL29", "AL30", "AL35", "AL41", "AE38", "GD29", "GD30", "GD35", "GD38", "GD41", "GD46",
    "AL30D", "GD30D", "AL35D", "GD35D", "TX26", "TX28", "TX31", "T2X5", "TZX26", "TZX27",
}
BOND_ETFS = {"AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "MUB", "VCIT", "BIL", "TIP", "SGOV"}
COMMODITY_ETFS = {"GLD", "SLV", "USO", "DBA", "PDBC", "DBB", "UGA", "IAU"}
AR_BOND_RE = re.compile(r"^(AL|GD|AE|TX|TZX|T\dX|S\d{2}[A-Z])\w*\d")

# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def safe_join_list(data, fallback="No disponible"):
    if isinstance(data, (list, tuple, set)):
        return ", ".join(str(x) for x in data)
    if data is None:
        return fallback
    return str(data)


def parse_csv_list(text, upper=False):
    items = [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]
    return [x.upper() for x in items] if upper else items


def normalize(ws):
    ws = np.asarray(ws, dtype=float)
    ws = np.clip(ws, 0, None)
    s = ws.sum()
    return ws / s if s > 0 else np.full(len(ws), 1 / max(len(ws), 1))


def fmt_num(x, pattern="{:,.2f}", na="N/A"):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return na
        return pattern.format(x)
    except (ValueError, TypeError):
        return na


def fmt_big(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "N/A"
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= div:
            return f"${x / div:,.2f}{suf}"
    return f"${x:,.0f}"


def classify_ticker(t):
    """Devuelve (moneda, clase de activo) por heurística."""
    t = t.upper()
    base = t.replace(".BA", "")
    if t.endswith(".BA") or base in AR_LOCAL:
        return "ARS", "Renta Fija" if AR_BOND_RE.match(base) else "Acciones AR"
    if "=X" in t:
        return "FX", "Divisas"
    if t.startswith("^"):
        return "Índice", "Índice"
    if t.endswith("-USD"):
        return "USD", "Cripto"
    if t in BOND_ETFS:
        return "USD", "Renta Fija"
    if t in COMMODITY_ETFS:
        return "USD", "Commodities"
    return "USD", "Acciones/ETF"


def extract_text_from_file(uploaded_file, max_chars=15000):
    if uploaded_file is None:
        return ""
    try:
        ftype = uploaded_file.type or ""
        content = ""
        if ftype == "application/pdf" and PDF_OK:
            reader = PdfReader(uploaded_file)
            for page in reader.pages[:10]:
                content += (page.extract_text() or "") + "\n"
        elif "wordprocessingml" in ftype and DOCX_OK:
            content = "\n".join(p.text for p in Document(uploaded_file).paragraphs)
        elif ftype == "text/csv":
            content = pd.read_csv(uploaded_file).head(50).to_string(index=False)
        else:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...contenido truncado...]"
        return content.strip()
    except Exception as e:
        return f"⚠️ Error al procesar el archivo: {e}"


def safe_int(x, default=2):
    try:
        v = int(float(x))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _color_signed(v):
    try:
        if v > 0:
            return "color: #00CC96"
        if v < 0:
            return "color: #EF553B"
    except TypeError:
        pass
    return ""


def style_signed(styler, subset=None):
    fn = getattr(styler, "map", None) or styler.applymap  # pandas >= 2.1 usa .map
    return fn(_color_signed, subset=subset)


def to_returns(prices):
    return prices.ffill().pct_change().iloc[1:].dropna(how="any")


# ═══════════════════════════════════════════════════════════════════════════
#  CAPA DE IA (OpenAI / Gemini) — una sola función para toda la app
# ═══════════════════════════════════════════════════════════════════════════

def ai_ready():
    return bool(st.session_state.get("preferred_ai"))


def call_ai(messages, system=None, temperature=0.4, json_mode=False):
    """messages: lista de {"role": "user"|"assistant", "content": str}"""
    engine = st.session_state.get("preferred_ai")
    if engine == "OpenAI":
        client = OpenAI(api_key=st.session_state.openai_api_key)
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs = {"model": st.session_state.openai_model, "messages": msgs}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            r = client.chat.completions.create(temperature=temperature, **kwargs)
        except Exception as e:
            # Algunos modelos de razonamiento no aceptan temperature
            if "temperature" in str(e).lower():
                r = client.chat.completions.create(**kwargs)
            else:
                raise
        return r.choices[0].message.content or ""

    if engine == "Gemini":
        model = st.session_state.gemini_model
        if GEMINI_BACKEND == "new":
            client = genai_new.Client(api_key=st.session_state.gemini_api_key)
            contents = [
                genai_types.Content(role="user" if m["role"] == "user" else "model",
                                    parts=[genai_types.Part(text=m["content"])])
                for m in messages
            ]
            cfg = genai_types.GenerateContentConfig(
                system_instruction=system, temperature=temperature,
                response_mime_type="application/json" if json_mode else None)
            return client.models.generate_content(model=model, contents=contents, config=cfg).text or ""
        genai_old.configure(api_key=st.session_state.gemini_api_key)
        gm = genai_old.GenerativeModel(model, system_instruction=system) if system else genai_old.GenerativeModel(model)
        hist = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages]
        cfg = {"temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        return gm.generate_content(hist, generation_config=cfg).text or ""

    raise RuntimeError("No hay motor de IA configurado.")


def extract_json(raw):
    if not raw:
        raise ValueError("Respuesta vacía del modelo")
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1 or e <= s:
        raise ValueError("No se encontró un objeto JSON en la respuesta")
    return json.loads(txt[s:e + 1])


# ═══════════════════════════════════════════════════════════════════════════
#  PERSISTENCIA: GOOGLE SHEETS + JSON LOCAL
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _gsheets_connect():
    """Devuelve (cliente, mensaje). No dibuja nada dentro del cache."""
    if not GSHEETS_OK:
        return None, "gspread / google-auth no instalados"
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        raw_json = secret("google_sheets", "service_account_json")
        if raw_json:
            creds = json.loads(raw_json) if isinstance(raw_json, str) else dict(raw_json)
        elif secret("gcp_service_account") is not None:
            creds = dict(secret("gcp_service_account"))
        else:
            return None, "Sin credenciales en Secrets"
        if "private_key" in creds:
            creds["private_key"] = creds["private_key"].replace("\\n", "\n")
        return gspread.authorize(Credentials.from_service_account_info(creds, scopes=scopes)), "OK"
    except json.JSONDecodeError:
        return None, "service_account_json no es un JSON válido"
    except Exception as e:
        return None, f"Error Google Sheets: {e}"


def get_gsheets_client():
    return _gsheets_connect()[0]


def _get_ws(client):
    try:
        ss = client.open_by_key(SHEET_ID) if SHEET_ID else client.open(SHEET_NAME)
        try:
            return ss.worksheet(WORKSHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = ss.add_worksheet(title=WORKSHEET_NAME, rows=200, cols=3)
            ws.append_row(["name", "tickers", "weights"])
            return ws
    except Exception as e:
        st.session_state["storage_error"] = str(e)
        return None


def load_portfolios():
    client = get_gsheets_client()
    if client:
        ws = _get_ws(client)
        if ws:
            try:
                out = {}
                for row in ws.get_all_records():
                    name = str(row.get("name", "")).strip()
                    tks = parse_csv_list(row.get("tickers", ""), upper=True)
                    if not name or not tks:
                        continue
                    try:
                        ws_ = [float(w) for w in parse_csv_list(row.get("weights", ""))]
                        if len(ws_) != len(tks):
                            raise ValueError
                    except ValueError:
                        ws_ = [1.0] * len(tks)
                    out[name] = {"tickers": tks, "weights": normalize(ws_).tolist()}
                return out
            except Exception as e:
                st.session_state["storage_error"] = str(e)
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_portfolios(pf):
    """Devuelve (ok, mensaje)."""
    msgs = []
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(pf, f, indent=2, ensure_ascii=False)
    except Exception as e:
        msgs.append(f"local: {e}")
    client = get_gsheets_client()
    if client:
        ws = _get_ws(client)
        if ws:
            try:
                rows = [["name", "tickers", "weights"]]
                rows += [[n, ", ".join(d["tickers"]), ", ".join(f"{w:.6f}" for w in d["weights"])]
                         for n, d in pf.items()]
                ws.clear()
                ws.update(values=rows, range_name="A1")
            except Exception as e:
                msgs.append(f"Sheets: {e}")
    return (len(msgs) == 0, "; ".join(msgs))


def persist(pf, ok_msg):
    st.session_state.portfolios = pf
    ok, msg = save_portfolios(pf)
    if ok:
        st.toast(ok_msg, icon="✅")
    else:
        st.warning(f"Guardado parcial ({msg})")


# ═══════════════════════════════════════════════════════════════════════════
#  DATOS DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════

def resolve_symbol(t, mode):
    t = t.strip().upper()
    if any(c in t for c in ".=^-"):
        return t
    if mode == "BYMA":
        return t + ".BA"
    if mode == "USA":
        return t
    return t + ".BA" if t in AR_LOCAL else t


def _extract_close(raw, symbols):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]].copy()
        close.columns = [symbols[0]]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    return close


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(tickers, start, end, mode="Auto", use_iol=False):
    """Devuelve (precios DataFrame | None, faltantes list, fuentes dict)."""
    tickers = list(dict.fromkeys(tickers))
    series, source, pending, missing = {}, {}, [], []

    client = get_iol_client() if use_iol else None
    for t in tickers:
        ok = False
        if client is not None and not any(c in t for c in "=^-"):
            try:
                h = client.get_serie_historica(t.split(".")[0].upper(),
                                               pd.to_datetime(start).strftime("%Y-%m-%d"),
                                               pd.to_datetime(end).strftime("%Y-%m-%d"))
                if h is not None and not h.empty and "ultimoPrecio" in h.columns:
                    s = h["ultimoPrecio"].astype(float)
                    s.index = pd.to_datetime(s.index)
                    if s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    series[t] = s.groupby(s.index.normalize()).last()
                    source[t] = "IOL"
                    ok = True
            except Exception:
                pass
        if not ok:
            pending.append(t)

    if pending:
        sym_map = {resolve_symbol(t, mode): t for t in pending}
        try:
            raw = yf.download(list(sym_map), start=pd.to_datetime(start),
                              end=pd.to_datetime(end) + pd.Timedelta(days=1),
                              auto_adjust=True, progress=False, threads=True)
            close = _extract_close(raw, list(sym_map))
        except Exception:
            close = pd.DataFrame()
        for sym, t in sym_map.items():
            if sym in close.columns and close[sym].notna().sum() > 5:
                s = close[sym].astype(float)
                s.index = pd.to_datetime(s.index)
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                series[t] = s
                source[t] = f"Yahoo ({sym})"
            else:
                missing.append(t)

    if not series:
        return None, missing, source
    df = pd.concat(series, axis=1).sort_index().ffill().dropna(how="any")
    return (df if len(df) > 1 else None), missing, source


# ═══════════════════════════════════════════════════════════════════════════
#  MOTOR CUANTITATIVO
# ═══════════════════════════════════════════════════════════════════════════

def get_cov(returns, shrink=True):
    if shrink and PYPFOPT_OK:
        try:
            return risk_models.CovarianceShrinkage(returns, returns_data=True, frequency=TD).ledoit_wolf()
        except Exception:
            pass
    return returns.cov() * TD


def package_result(w, returns, rf, method, cov=None, mu=None, extra=None):
    w = pd.Series(w, index=returns.columns) if not isinstance(w, pd.Series) else w
    w = pd.Series(normalize(w.reindex(returns.columns).fillna(0).values), index=returns.columns)
    mu = returns.mean() * TD if mu is None else mu.reindex(returns.columns)
    cov = returns.cov() * TD if cov is None else cov.reindex(index=returns.columns, columns=returns.columns)
    r = float(w @ mu)
    v = float(np.sqrt(max(w @ cov @ w, 0)))
    out = {"weights": w.values, "tickers": list(returns.columns), "expected_return": r,
           "volatility": v, "sharpe_ratio": (r - rf) / v if v > 0 else 0.0, "method": method}
    if extra:
        out.update(extra)
    return out


def _scipy_markowitz(mu, cov, rf, objective, max_w, target):
    n = len(mu)
    m, C = mu.values, cov.values
    x0 = np.full(n, 1 / n)
    bounds = [(0.0, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def vol(w):
        return np.sqrt(max(w @ C @ w, 1e-16))

    if objective == "Mínima Volatilidad":
        f = vol
    elif objective == "Retorno Máximo":
        f = lambda w: -(w @ m)
    elif objective == "Volatilidad Objetivo":
        f = lambda w: -(w @ m)
        cons.append({"type": "ineq", "fun": lambda w: target - vol(w)})
    elif objective == "Retorno Objetivo":
        f = vol
        cons.append({"type": "ineq", "fun": lambda w: w @ m - target})
    elif objective == "Utilidad Cuadrática":
        f = lambda w: -(w @ m - 0.5 * target * (w @ C @ w))
    else:  # Máximo Sharpe
        f = (lambda w: -((w @ m - rf) / vol(w))) if (m > rf).any() else vol
    res = minimize(f, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    w = res.x if res.success or np.isfinite(res.x).all() else x0
    return pd.Series(normalize(w), index=mu.index)


def opt_markowitz(returns, rf=0.04, objective="Máximo Sharpe", max_w=1.0, target=None, shrink=True):
    n = returns.shape[1]
    max_w = max(float(max_w), 1 / n + 1e-6)
    mu, cov = returns.mean() * TD, get_cov(returns, shrink)
    w, method = None, "Markowitz"
    if PYPFOPT_OK and objective != "Retorno Máximo":
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_w))
            if objective == "Máximo Sharpe":
                ef.max_sharpe(risk_free_rate=rf)
            elif objective == "Mínima Volatilidad":
                ef.min_volatility()
            elif objective == "Volatilidad Objetivo":
                ef.efficient_risk(target)
            elif objective == "Retorno Objetivo":
                ef.efficient_return(target)
            else:
                ef.max_quadratic_utility(risk_aversion=target or 1.0)
            w = pd.Series(ef.clean_weights())
            method = "Markowitz (PyPortfolioOpt)"
        except Exception:
            w = None
    if w is None:
        w = _scipy_markowitz(mu, cov, rf, objective, max_w, target)
        method = "Markowitz (SciPy)"
    return package_result(w, returns, rf, f"{method} · {objective}", cov=cov)


def opt_risk_parity(returns, rf=0.04, shrink=True):
    """Equal Risk Contribution resolviendo el problema convexo de Spinu (2013)."""
    cov = get_cov(returns, shrink)
    C = cov.values
    n = C.shape[0]
    b = np.full(n, 1 / n)
    f = lambda y: 0.5 * y @ C @ y - b @ np.log(y)
    g = lambda y: C @ y - b / y
    res = minimize(f, np.full(n, 1 / n), jac=g, method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    return package_result(pd.Series(normalize(res.x), index=cov.index), returns, rf, "Risk Parity (ERC)", cov=cov)


def opt_hrp(returns, rf=0.04):
    if PYPFOPT_OK:
        try:
            w = pd.Series(HRPOpt(returns=returns).optimize())
            return package_result(w, returns, rf, "Hierarchical Risk Parity")
        except Exception:
            pass
    iv = 1 / returns.var()
    return package_result(iv / iv.sum(), returns, rf, "Inversa de la varianza (fallback HRP)")


def opt_black_litterman(returns, rf=0.04, views=None, delta=2.5, tau=0.05, max_w=1.0, shrink=True):
    """
    Black-Litterman con prior de equilibrio implícito en pesos iguales
    (no hay capitalizaciones de mercado para todos los activos).
    views: dict {ticker: (retorno anual esperado, confianza 0-1)}
    """
    cov = get_cov(returns, shrink)
    tick = list(cov.index)
    S = cov.values
    n = len(tick)
    pi = delta * S @ np.full(n, 1 / n)
    views = {k: v for k, v in (views or {}).items() if k in tick}
    if views:
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        omega = np.zeros(len(views))
        for i, (t, (q, c)) in enumerate(views.items()):
            P[i, tick.index(t)] = 1
            Q[i] = q
            c = float(np.clip(c, 0.01, 0.99))
            omega[i] = tau * (P[i] @ S @ P[i]) * (1 - c) / c  # He-Litterman / Idzorek simplificado
        tS_inv = np.linalg.inv(tau * S)
        Om_inv = np.diag(1 / omega)
        M = np.linalg.inv(tS_inv + P.T @ Om_inv @ P)
        post_mu = M @ (tS_inv @ pi + P.T @ Om_inv @ Q)
        post_S = S + M
    else:
        post_mu, post_S = pi, S
    post_mu = pd.Series(post_mu, index=tick)
    post_cov = pd.DataFrame(post_S, index=tick, columns=tick)
    max_w = max(float(max_w), 1 / n + 1e-6)
    w = _scipy_markowitz(post_mu, post_cov, rf, "Máximo Sharpe", max_w, None)
    return package_result(w, returns, rf, "Black-Litterman", cov=post_cov, mu=post_mu,
                          extra={"bl_prior": pd.Series(pi, index=tick), "bl_posterior": post_mu})


def opt_equal(returns, rf=0.04):
    n = returns.shape[1]
    return package_result(np.full(n, 1 / n), returns, rf, "Pesos Iguales (1/N)")


def run_method(method, returns, rf, p):
    if method == "Markowitz":
        return opt_markowitz(returns, rf, p.get("objective", "Máximo Sharpe"), p.get("max_w", 1.0),
                             p.get("target"), p.get("shrink", True))
    if method == "Risk Parity":
        return opt_risk_parity(returns, rf, p.get("shrink", True))
    if method == "HRP":
        return opt_hrp(returns, rf)
    if method == "Black-Litterman":
        return opt_black_litterman(returns, rf, p.get("views"), p.get("delta", 2.5),
                                   max_w=p.get("max_w", 1.0), shrink=p.get("shrink", True))
    return opt_equal(returns, rf)


# ─── Métricas ──────────────────────────────────────────────────────────────

def port_returns(returns, weights):
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    return returns @ w


def perf_metrics(pr, rf=0.04, bench=None):
    pr = pr.dropna()
    if len(pr) < 2:
        return {}
    ann_r = pr.mean() * TD
    ann_v = pr.std() * np.sqrt(TD)
    cum = (1 + pr).cumprod()
    years = len(pr) / TD
    cagr = cum.iloc[-1] ** (1 / years) - 1 if cum.iloc[-1] > 0 else np.nan
    downside = np.sqrt((np.minimum(pr - rf / TD, 0) ** 2).mean()) * np.sqrt(TD)
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    var95 = np.percentile(pr, 5)
    tail = pr[pr <= var95]
    m = {
        "Retorno anual": ann_r, "CAGR": cagr, "Volatilidad": ann_v,
        "Sharpe": (ann_r - rf) / ann_v if ann_v > 0 else 0,
        "Sortino": (ann_r - rf) / downside if downside > 0 else 0,
        "Max Drawdown": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 and pd.notna(cagr) else 0,
        "VaR 95% (diario)": var95, "CVaR 95% (diario)": tail.mean() if len(tail) else var95,
        "VaR param. 95%": normal_dist.ppf(0.05, pr.mean(), pr.std()),
        "Skewness": pr.skew(), "Kurtosis": pr.kurtosis(),
        "Retorno total": cum.iloc[-1] - 1, "% días positivos": (pr > 0).mean(),
    }
    if bench is not None:
        b = bench.reindex(pr.index).dropna()
        a = pr.reindex(b.index)
        if len(b) > 20 and b.var() > 0:
            beta = a.cov(b) / b.var()
            te = (a - b).std() * np.sqrt(TD)
            m["Beta"] = beta
            m["Alpha (Jensen)"] = (a.mean() * TD - rf) - beta * (b.mean() * TD - rf)
            m["Tracking Error"] = te
            m["Information Ratio"] = ((a - b).mean() * TD) / te if te > 0 else 0
            m["Correlación c/ bench"] = a.corr(b)
    return m


PCT_KEYS = {"Retorno anual", "CAGR", "Volatilidad", "Max Drawdown", "VaR 95% (diario)", "CVaR 95% (diario)",
            "VaR param. 95%", "Retorno total", "% días positivos", "Alpha (Jensen)", "Tracking Error"}


def metrics_table(named):
    """named: {nombre: dict de métricas} → DataFrame formateado como strings."""
    df = pd.DataFrame(named)
    out = df.copy().astype(object)
    for k in df.index:
        for c in df.columns:
            v = df.loc[k, c]
            out.loc[k, c] = fmt_num(v, "{:.2%}" if k in PCT_KEYS else "{:.2f}")
    return out


def risk_contrib(weights, cov):
    w = np.asarray(weights, dtype=float)
    C = np.asarray(cov)
    pv = w @ C @ w
    return w * (C @ w) / pv if pv > 0 else np.zeros_like(w)


def frontier(returns, rf, max_w=1.0, shrink=True, n_points=30, n_random=2500, seed=7):
    mu, cov = returns.mean() * TD, get_cov(returns, shrink)
    n = len(mu)
    rng = np.random.default_rng(seed)
    Wr = rng.dirichlet(np.full(n, 0.7), n_random)
    cloud = pd.DataFrame({"ret": Wr @ mu.values,
                          "vol": np.sqrt(np.einsum("ij,jk,ik->i", Wr, cov.values, Wr))})
    cloud["sharpe"] = (cloud.ret - rf) / cloud.vol
    max_w = max(max_w, 1 / n + 1e-6)
    w_min = _scipy_markowitz(mu, cov, rf, "Mínima Volatilidad", max_w, None)
    r_lo, r_hi = float(w_min @ mu), float(_scipy_markowitz(mu, cov, rf, "Retorno Máximo", max_w, None) @ mu)
    pts = []
    for tgt in np.linspace(r_lo, r_hi, n_points):
        w = _scipy_markowitz(mu, cov, rf, "Retorno Objetivo", max_w, tgt)
        pts.append((float(np.sqrt(w @ cov @ w)), float(w @ mu)))
    curve = pd.DataFrame(pts, columns=["vol", "ret"]).sort_values("vol")
    return cloud, curve, mu, cov


# ─── Backtests ─────────────────────────────────────────────────────────────

FREQ = {"Nunca (buy & hold)": None, "Mensual": "M", "Trimestral": "Q", "Anual": "Y"}


def backtest(returns, weights, freq=None, cost_bps=0.0):
    """Simula la cartera con rebalanceo periódico. Devuelve retornos diarios."""
    R = returns.values
    tw = normalize(pd.Series(weights).reindex(returns.columns).fillna(0).values)
    keys = returns.index.to_period(freq) if freq else None
    c = cost_bps / 1e4
    h, v, vals = tw.copy(), 1.0, np.empty(len(R))
    for i in range(len(R)):
        if keys is not None and i > 0 and keys[i] != keys[i - 1]:
            new_h = v * tw
            v -= np.abs(new_h - h).sum() * c
            h = v * tw
        h = h * (1 + R[i])
        v = h.sum()
        vals[i] = v
    eq = pd.Series(vals, index=returns.index)
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def walk_forward(returns, method, rf, params, lookback=252, freq="Q", cost_bps=10.0):
    """Re-optimiza cada período usando SOLO datos pasados (fuera de muestra)."""
    R = returns.values
    n = R.shape[1]
    keys = returns.index.to_period(freq)
    c = cost_bps / 1e4
    h, v = None, 1.0
    vals, dates, w_hist = [], [], []
    last_w = np.full(n, 1 / n)
    for i in range(lookback, len(R)):
        if h is None or keys[i] != keys[i - 1]:
            try:
                tw = np.asarray(run_method(method, returns.iloc[i - lookback:i], rf, params)["weights"], float)
                if not np.isfinite(tw).all():
                    raise ValueError
            except Exception:
                tw = last_w
            last_w = tw
            prev = h if h is not None else np.zeros(n)
            v -= np.abs(v * tw - prev).sum() * c
            h = v * tw
            w_hist.append(pd.Series(tw, index=returns.columns, name=returns.index[i]))
        h = h * (1 + R[i])
        v = h.sum()
        vals.append(v)
        dates.append(returns.index[i])
    if not vals:
        return None, None
    eq = pd.Series(vals, index=dates)
    return eq.pct_change().fillna(eq.iloc[0] - 1), pd.DataFrame(w_hist)


def monte_carlo(pr, days, sims, method="GBM", seed=None):
    rng = np.random.default_rng(seed)
    if method == "GBM":
        mu, sig = pr.mean(), pr.std()
        log_r = (mu - 0.5 * sig ** 2) + sig * rng.standard_normal((days, sims))
        return np.exp(np.cumsum(log_r, axis=0))
    sample = rng.choice(pr.values, size=(days, sims), replace=True)  # bootstrap histórico
    return np.cumprod(1 + sample, axis=0)


# ─── Renta fija ────────────────────────────────────────────────────────────

def bond_analytics(coupon_pct, ytm_pct, years, freq=2):
    """Bono bullet: precio (% VN), duration Macaulay/modificada (años), convexidad."""
    freq = max(int(freq), 1)
    c, y = coupon_pct / 100 / freq, ytm_pct / 100 / freq
    n = max(1, int(round(years * freq)))
    t = np.arange(1, n + 1)
    cf = np.full(n, c)
    cf[-1] += 1.0
    pv = cf / (1 + y) ** t
    price = pv.sum()
    mac = (t * pv).sum() / price / freq
    mod = mac / (1 + y)
    conv = (pv * t * (t + 1)).sum() / (price * (1 + y) ** 2) / freq ** 2
    return price * 100, mac, mod, conv


# ═══════════════════════════════════════════════════════════════════════════
#  CONTEXTO PARA IA
# ═══════════════════════════════════════════════════════════════════════════

def build_portfolio_context(res, prices=None, name="Portafolio", rf=0.04, bench=None):
    L = [f"ANÁLISIS: {name}", f"Fecha: {date.today():%Y-%m-%d}", "",
         "MÉTRICAS GLOBALES (anualizadas, históricas):",
         f"- Retorno esperado: {res['expected_return']:.2%}",
         f"- Volatilidad: {res['volatility']:.2%}",
         f"- Sharpe (rf={rf:.1%}): {res['sharpe_ratio']:.2f}",
         f"- Método: {res.get('method', 'N/A')}", "", "COMPOSICIÓN:"]
    active = sorted([(t, w) for t, w in zip(res["tickers"], res["weights"]) if w > 0.001],
                    key=lambda x: -x[1])
    for t, w in active:
        cur, cls = classify_ticker(t)
        L.append(f"- {t:<10} {w:6.1%}  [{cur} · {cls}]")
    if active and active[0][1] > 0.4:
        L.append(f"⚠️ Concentración alta en {active[0][0]}")
    hhi = sum(w ** 2 for _, w in active)
    L.append(f"- N efectivo de activos (1/HHI): {1 / hhi:.1f}" if hhi > 0 else "")

    exp_cur, exp_cls = {}, {}
    for t, w in active:
        cur, cls = classify_ticker(t)
        exp_cur[cur] = exp_cur.get(cur, 0) + w
        exp_cls[cls] = exp_cls.get(cls, 0) + w
    L += ["", "EXPOSICIÓN POR MONEDA: " + ", ".join(f"{k} {v:.0%}" for k, v in exp_cur.items()),
          "EXPOSICIÓN POR CLASE: " + ", ".join(f"{k} {v:.0%}" for k, v in exp_cls.items())]

    if prices is not None and len(prices) > 30:
        rets = to_returns(prices)
        L += ["", "MÉTRICAS INDIVIDUALES:"]
        for t, _ in active:
            if t in rets:
                r, v = rets[t].mean() * TD, rets[t].std() * np.sqrt(TD)
                L.append(f"- {t}: Ret {r:6.1%} | Vol {v:5.1%} | Sharpe {(r - rf) / v if v else 0:5.2f}")
        cov = rets.cov() * TD
        rc = risk_contrib(pd.Series(res["weights"], index=res["tickers"]).reindex(cov.index).fillna(0).values, cov.values)
        L += ["", "CONTRIBUCIÓN AL RIESGO:"] + [f"- {t}: {c:.1%}" for t, c in zip(cov.index, rc) if c > 0.005]
        names = [t for t, _ in active]
        corr = rets[names].corr() if len(names) > 1 else None
        if corr is not None:
            pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(names) for b in names[i + 1:]
                     if abs(corr.loc[a, b]) > 0.65]
            if pairs:
                L += ["", "CORRELACIONES ALTAS:"] + [f"- {a} ↔ {b}: {c:+.2f}" for a, b, c in pairs]
        pm = perf_metrics(port_returns(rets, pd.Series(res["weights"], index=res["tickers"])), rf, bench)
        L += ["", "RIESGO HISTÓRICO:",
              f"- Max Drawdown: {pm.get('Max Drawdown', 0):.2%}",
              f"- VaR 95% diario: {pm.get('VaR 95% (diario)', 0):.2%} | CVaR: {pm.get('CVaR 95% (diario)', 0):.2%}",
              f"- Sortino: {pm.get('Sortino', 0):.2f}"]
        if "Beta" in pm:
            L.append(f"- Beta vs benchmark: {pm['Beta']:.2f}")
        L.append(f"\nDATOS: {len(prices)} observaciones ({prices.index[0]:%Y-%m-%d} a {prices.index[-1]:%Y-%m-%d})")
    return "\n".join(x for x in L if x is not None)


# ═══════════════════════════════════════════════════════════════════════════
#  COMPONENTES DE GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

def chart_equity(curves, title="Evolución (base 100)", log=False):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    palette = [COLORS["opt"], COLORS["cur"], COLORS["ew"], COLORS["bench"], "#FFA15A", "#19D3F3"]
    for i, (name, r) in enumerate(curves.items()):
        eq = (1 + r).cumprod() * 100
        dd = eq / eq.cummax() - 1
        col = palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=dict(color=col, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"DD {name}", line=dict(color=col, width=1),
                                 showlegend=False, fill="tozeroy" if i == 0 else None), 2, 1)
    fig.update_yaxes(title_text="Valor", type="log" if log else "linear", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=520, title=title, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    return fig


def chart_weights(df_w):
    """df_w: DataFrame index=ticker, columnas = carteras."""
    fig = go.Figure()
    palette = [COLORS["cur"], COLORS["opt"], COLORS["ew"]]
    for i, c in enumerate(df_w.columns):
        fig.add_trace(go.Bar(x=df_w.index, y=df_w[c], name=c, marker_color=palette[i % 3]))
    fig.update_layout(barmode="group", template=TEMPLATE, height=360, yaxis_tickformat=".0%",
                      title="Pesos por activo")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def section_manage(portfolios):
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("Administrar carteras")
        action = st.radio("Acción", ["✨ Crear", "✏️ Editar / 🗑️ Eliminar", "📥 Importar de IA"],
                          horizontal=True, key="dash_action")
        if action == "✨ Crear":
            name = st.text_input("Nombre", key="new_name")
            tks = st.text_area("Tickers (separados por coma)", "SPY, QQQ, TLT, GLD", key="new_tks",
                               help="Usá sufijo .BA para BYMA (ej. GGAL.BA). Sin sufijo se aplica el modo de mercado de la barra lateral.")
            wts = st.text_area("Pesos (coma, se normalizan)", "0.4, 0.2, 0.3, 0.1", key="new_ws")
            if st.button("💾 Guardar", type="primary", key="save_new"):
                try:
                    ts, ws = parse_csv_list(tks, upper=True), [float(x) for x in parse_csv_list(wts)]
                    if not name.strip():
                        st.error("Poné un nombre.")
                    elif len(ts) != len(ws):
                        st.error(f"Hay {len(ts)} tickers y {len(ws)} pesos.")
                    elif len(set(ts)) != len(ts):
                        st.error("Hay tickers repetidos.")
                    elif name in portfolios:
                        st.error("Ya existe una cartera con ese nombre.")
                    else:
                        portfolios[name.strip()] = {"tickers": ts, "weights": normalize(ws).tolist()}
                        persist(portfolios, "Cartera guardada")
                        st.rerun()
                except ValueError:
                    st.error("Los pesos deben ser números (usá punto decimal).")
        elif action.startswith("✏️"):
            if not portfolios:
                st.info("No hay carteras todavía.")
            else:
                sel = st.selectbox("Seleccionar", list(portfolios), key="edit_sel")
                d = portfolios[sel]
                nn = st.text_input("Renombrar", value=sel, key=f"ren_{sel}")
                nt = st.text_area("Tickers", value=", ".join(d["tickers"]), key=f"et_{sel}")
                nw = st.text_area("Pesos", value=", ".join(f"{w:.4f}" for w in d["weights"]), key=f"ew_{sel}")
                b1, b2 = st.columns(2)
                if b1.button("🔄 Actualizar", key="upd", **W):
                    try:
                        ts, ws = parse_csv_list(nt, upper=True), [float(x) for x in parse_csv_list(nw)]
                        if len(ts) != len(ws):
                            st.error(f"Hay {len(ts)} tickers y {len(ws)} pesos.")
                        elif nn != sel and nn in portfolios:
                            st.error("Ya existe una cartera con ese nombre.")
                        else:
                            if nn != sel:
                                del portfolios[sel]
                            portfolios[nn] = {"tickers": ts, "weights": normalize(ws).tolist()}
                            persist(portfolios, "Cartera actualizada")
                            st.rerun()
                    except ValueError:
                        st.error("Pesos inválidos.")
                confirm = b2.checkbox("Confirmar borrado", key="del_confirm")
                if b2.button("🗑️ Eliminar", key="del", disabled=not confirm, **W):
                    del portfolios[sel]
                    persist(portfolios, "Cartera eliminada")
                    st.rerun()
        else:
            q = st.session_state.get("quant_pf")
            if not q:
                st.info("Generá una estrategia en 🧠 Asistente Quant y tocá 'Enviar al Dashboard'.")
            else:
                st.write(f"**{q['name']}** · {len(q['tickers'])} activos")
                st.dataframe(pd.DataFrame({"Ticker": q["tickers"], "Peso": q["weights"]})
                             .style.format({"Peso": "{:.1%}"}), hide_index=True, **W)
                name = st.text_input("Guardar como", value=q["name"], key="imp_name")
                if st.button("📥 Importar", type="primary"):
                    portfolios[name] = {"tickers": q["tickers"], "weights": normalize(q["weights"]).tolist()}
                    persist(portfolios, "Estrategia importada")
                    st.session_state.pop("quant_pf", None)
                    st.rerun()
    with c2:
        st.subheader("📋 Carteras guardadas")
        if portfolios:
            df = pd.DataFrame([{"Nombre": k, "Activos": len(v["tickers"]), "Tickers": ", ".join(v["tickers"]),
                                "Peso mayor": max(v["weights"])} for k, v in portfolios.items()])
            st.dataframe(df.style.format({"Peso mayor": "{:.1%}"}), hide_index=True, **W)
            st.download_button("📥 Exportar JSON", json.dumps(portfolios, indent=2, ensure_ascii=False),
                               "carteras.json", "application/json")
        else:
            st.info("Todavía no hay carteras.")


def load_dashboard_data(portfolios):
    """Selector común a todas las pestañas. Devuelve dict de contexto o None."""
    c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, 0.8])
    pf_name = c1.selectbox("📦 Cartera", list(portfolios), key="dash_pf")
    ds = c2.date_input("Desde", date(2023, 1, 1), key="dash_ds")
    de = c3.date_input("Hasta", date.today(), key="dash_de")
    bench = c4.text_input("Benchmark", "SPY", key="dash_bench",
                          help="Ej. SPY, QQQ, ^MERV, ^GSPC. Vacío = sin benchmark.").strip().upper()
    rf = c5.number_input("Tasa libre (rf)", 0.0, 2.0, 0.04, 0.005, format="%.3f", key="dash_rf")
    if ds >= de:
        st.error("La fecha 'Desde' debe ser anterior a 'Hasta'.")
        return None

    pf = portfolios[pf_name]
    mode = st.session_state.get("market_mode", "Auto")
    use_iol = bool(st.session_state.get("iol_connected"))
    with st.spinner("Descargando precios..."):
        prices, missing, source = fetch_prices(tuple(pf["tickers"]), ds, de, mode, use_iol)
    if prices is None or len(prices) < 40:
        st.error("No hay datos suficientes para esta cartera y rango de fechas. "
                 f"Sin datos: {', '.join(missing) or '—'}")
        return None
    if missing:
        st.warning(f"Sin datos para: {', '.join(missing)}. Se excluyen y los pesos se renormalizan.")

    tickers = list(prices.columns)
    w_cur = pd.Series(pf["weights"], index=pf["tickers"]).reindex(tickers).fillna(0)
    w_cur = pd.Series(normalize(w_cur.values), index=tickers)
    rets = to_returns(prices)

    bench_r = None
    if bench:
        bp, _, _ = fetch_prices((bench,), ds, de, "USA", False)
        if bp is not None:
            bench_r = to_returns(bp).iloc[:, 0].reindex(rets.index).dropna()
        else:
            st.caption(f"⚠️ No se pudo descargar el benchmark {bench}.")

    currencies = {classify_ticker(t)[0] for t in tickers} - {"Índice"}
    if len(currencies) > 1:
        st.caption(f"⚠️ La cartera mezcla monedas ({', '.join(sorted(currencies))}); los retornos se calculan en la "
                   "moneda de cada activo, sin convertir. Para comparar en USD usá CEDEARs/bonos en D o ajustá por CCL.")

    with st.expander("ℹ️ Fuentes de datos"):
        st.write({t: source.get(t, "—") for t in tickers})
        st.caption(f"{len(prices)} observaciones · {prices.index[0]:%d/%m/%Y} → {prices.index[-1]:%d/%m/%Y}")

    return {"name": pf_name, "prices": prices, "rets": rets, "w_cur": w_cur, "bench": bench,
            "bench_r": bench_r, "rf": rf, "key": (pf_name, str(ds), str(de), tuple(tickers))}


def current_opt(ctx):
    o = st.session_state.get("opt")
    if o and o.get("key") == ctx["key"]:
        return o["res"]
    return None


def section_analysis(ctx):
    rets, w, rf = ctx["rets"], ctx["w_cur"], ctx["rf"]
    pr = port_returns(rets, w)
    ew = port_returns(rets, pd.Series(1 / len(w), index=w.index))
    m = perf_metrics(pr, rf, ctx["bench_r"])

    k = st.columns(6)
    k[0].metric("CAGR", fmt_num(m.get("CAGR"), "{:.1%}"))
    k[1].metric("Volatilidad", fmt_num(m.get("Volatilidad"), "{:.1%}"))
    k[2].metric("Sharpe", fmt_num(m.get("Sharpe"), "{:.2f}"))
    k[3].metric("Sortino", fmt_num(m.get("Sortino"), "{:.2f}"))
    k[4].metric("Max DD", fmt_num(m.get("Max Drawdown"), "{:.1%}"))
    k[5].metric("Beta", fmt_num(m.get("Beta"), "{:.2f}"))

    curves = {ctx["name"]: pr, "Pesos iguales": ew}
    if ctx["bench_r"] is not None:
        curves[ctx["bench"]] = ctx["bench_r"].reindex(pr.index).fillna(0)
    st.plotly_chart(chart_equity(curves), **W)

    c1, c2 = st.columns(2)
    with c1:
        named = {ctx["name"]: m, "Pesos iguales": perf_metrics(ew, rf, ctx["bench_r"])}
        if ctx["bench_r"] is not None:
            named[ctx["bench"]] = perf_metrics(ctx["bench_r"], rf)
        st.markdown("**Métricas comparadas**")
        st.dataframe(metrics_table(named), **W)
    with c2:
        corr = rets.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        template=TEMPLATE, title="Matriz de correlación")
        fig.update_layout(height=420)
        st.plotly_chart(fig, **W)

    c3, c4 = st.columns(2)
    with c3:
        cov = rets.cov() * TD
        rc = risk_contrib(w.values, cov.values)
        dfr = pd.DataFrame({"Activo": w.index, "Peso": w.values, "Contribución al riesgo": rc})
        fig = go.Figure([go.Bar(x=dfr.Activo, y=dfr.Peso, name="Peso", marker_color=COLORS["cur"]),
                         go.Bar(x=dfr.Activo, y=dfr["Contribución al riesgo"], name="Riesgo",
                                marker_color=COLORS["bench"])])
        fig.update_layout(barmode="group", template=TEMPLATE, height=380, yaxis_tickformat=".0%",
                          title="Peso vs contribución al riesgo")
        st.plotly_chart(fig, **W)
    with c4:
        win = st.select_slider("Ventana rolling (días)", [21, 63, 126, 252], 63, key="roll_win")
        rv = pr.rolling(win).std() * np.sqrt(TD)
        rs = (pr.rolling(win).mean() * TD - rf) / rv
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=rv.index, y=rv, name="Vol rolling", line=dict(color=COLORS["cur"])), secondary_y=False)
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name="Sharpe rolling", line=dict(color=COLORS["opt"])), secondary_y=True)
        fig.update_yaxes(tickformat=".0%", secondary_y=False)
        fig.update_layout(template=TEMPLATE, height=330, title=f"Volatilidad y Sharpe rolling ({win}d)")
        st.plotly_chart(fig, **W)

    st.markdown("**Métricas por activo**")
    rows = []
    for t in rets.columns:
        mt = perf_metrics(rets[t], rf, ctx["bench_r"])
        cur, cls = classify_ticker(t)
        rows.append({"Activo": t, "Moneda": cur, "Clase": cls, "Peso": w[t], "Ret. anual": mt["Retorno anual"],
                     "Vol": mt["Volatilidad"], "Sharpe": mt["Sharpe"], "Max DD": mt["Max Drawdown"],
                     "Beta": mt.get("Beta", np.nan)})
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Peso": "{:.1%}", "Ret. anual": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}",
         "Max DD": "{:.1%}", "Beta": "{:.2f}"}, na_rep="—"), hide_index=True, **W)


def section_optimize(ctx, portfolios):
    rets, rf = ctx["rets"], ctx["rf"]
    n = rets.shape[1]
    if n < 2:
        st.info("Se necesitan al menos 2 activos con datos para optimizar.")
        return

    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Método", ["Markowitz", "Risk Parity", "HRP", "Black-Litterman", "Pesos Iguales"], key="opt_method")
    max_w = c2.slider("Peso máximo por activo", max(0.05, round(1 / n + 0.01, 2)), 1.0, 1.0, 0.05, key="opt_maxw")
    shrink = c3.checkbox("Covarianza Ledoit-Wolf (más robusta)", True, key="opt_shrink",
                         disabled=not PYPFOPT_OK, help="Requiere PyPortfolioOpt")
    params = {"max_w": max_w, "shrink": shrink and PYPFOPT_OK}

    if method == "Markowitz":
        a, b = st.columns(2)
        obj = a.selectbox("Objetivo", ["Máximo Sharpe", "Mínima Volatilidad", "Volatilidad Objetivo",
                                       "Retorno Objetivo", "Utilidad Cuadrática", "Retorno Máximo"], key="opt_obj")
        params["objective"] = obj
        if obj == "Volatilidad Objetivo":
            params["target"] = b.number_input("Volatilidad anual objetivo", 0.01, 2.0, 0.15, 0.01, key="opt_tv")
        elif obj == "Retorno Objetivo":
            params["target"] = b.number_input("Retorno anual objetivo", -0.5, 5.0, 0.12, 0.01, key="opt_tr")
        elif obj == "Utilidad Cuadrática":
            params["target"] = b.number_input("Aversión al riesgo (λ)", 0.1, 50.0, 2.0, 0.5, key="opt_ra")
    elif method == "Black-Litterman":
        st.caption("Prior de equilibrio implícito en pesos iguales (δ·Σ·w). Cargá tus views absolutas: "
                   "retorno anual esperado y confianza (0-1). Sin views = sólo el prior.")
        params["delta"] = st.number_input("Aversión al riesgo de mercado (δ)", 0.5, 10.0, 2.5, 0.5, key="bl_delta")
        if "bl_views" not in st.session_state or set(st.session_state.bl_views_tickers) != set(rets.columns):
            st.session_state.bl_views = pd.DataFrame({"Ticker": list(rets.columns), "Usar": False,
                                                      "Retorno esperado": 0.10, "Confianza": 0.5})
            st.session_state.bl_views_tickers = list(rets.columns)
        ed = st.data_editor(st.session_state.bl_views, hide_index=True,
                            key="bl_editor_" + "_".join(st.session_state.bl_views_tickers), **W,
                            column_config={"Ticker": st.column_config.TextColumn(disabled=True),
                                           "Retorno esperado": st.column_config.NumberColumn(format="%.3f"),
                                           "Confianza": st.column_config.NumberColumn(min_value=0.01, max_value=0.99)})
        params["views"] = {r.Ticker: (float(r["Retorno esperado"]), float(r.Confianza))
                           for _, r in ed.iterrows() if r.Usar}

    b1, b2 = st.columns([1, 1])
    if b1.button("🚀 Optimizar", type="primary", key="run_opt", **W):
        with st.spinner("Optimizando..."):
            try:
                res = run_method(method, rets, rf, params)
                st.session_state.opt = {"key": ctx["key"], "res": res, "method": method, "params": params}
            except Exception as e:
                st.error(f"Error al optimizar: {type(e).__name__}: {e}")
    if b2.button("🔄 Comparar todos los métodos", key="cmp_all", **W):
        rows = []
        for mname in ["Markowitz", "Risk Parity", "HRP", "Black-Litterman", "Pesos Iguales"]:
            try:
                r = run_method(mname, rets, rf, {"max_w": max_w, "shrink": params["shrink"],
                                                 "objective": "Máximo Sharpe"})
                pm = perf_metrics(port_returns(rets, pd.Series(r["weights"], index=r["tickers"])), rf)
                rows.append({"Método": r["method"], "Retorno": r["expected_return"], "Vol": r["volatility"],
                             "Sharpe": r["sharpe_ratio"], "Max DD": pm["Max Drawdown"],
                             "N efectivo": 1 / np.sum(np.square(r["weights"]))})
            except Exception as e:
                rows.append({"Método": f"{mname} (error: {e})"})
        st.session_state.cmp_table = pd.DataFrame(rows)
    if st.session_state.get("cmp_table") is not None:
        st.dataframe(st.session_state.cmp_table.style.format(
            {"Retorno": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "Max DD": "{:.1%}", "N efectivo": "{:.1f}"},
            na_rep="—"), hide_index=True, **W)
        st.caption("⚠️ Métricas dentro de muestra: optimizar y medir con los mismos datos sobreestima el resultado. "
                   "Mirá la pestaña Backtest → walk-forward para una evaluación honesta.")

    res = current_opt(ctx)
    if res is None:
        return
    st.markdown("---")
    w_opt = pd.Series(res["weights"], index=res["tickers"])
    w_cur = ctx["w_cur"]
    m_cur = perf_metrics(port_returns(rets, w_cur), rf, ctx["bench_r"])
    m_opt = perf_metrics(port_returns(rets, w_opt), rf, ctx["bench_r"])

    st.subheader(f"Resultado · {res['method']}")
    k = st.columns(5)
    k[0].metric("Retorno esperado", f"{res['expected_return']:.1%}",
                f"{res['expected_return'] - m_cur['Retorno anual']:+.1%} vs actual")
    k[1].metric("Volatilidad", f"{res['volatility']:.1%}",
                f"{res['volatility'] - m_cur['Volatilidad']:+.1%}", delta_color="inverse")
    k[2].metric("Sharpe", f"{res['sharpe_ratio']:.2f}", f"{m_opt['Sharpe'] - m_cur['Sharpe']:+.2f}")
    k[3].metric("Max DD (hist.)", f"{m_opt['Max Drawdown']:.1%}",
                f"{m_opt['Max Drawdown'] - m_cur['Max Drawdown']:+.1%}")
    k[4].metric("N efectivo", f"{1 / np.sum(w_opt.values ** 2):.1f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_weights(pd.DataFrame({"Actual": w_cur, "Óptima": w_opt})), **W)
    with c2:
        with st.spinner("Calculando frontera eficiente..."):
            cloud, curve, mu, cov = frontier(rets, rf, max_w, params["shrink"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cloud.vol, y=cloud.ret, mode="markers", name="Carteras aleatorias",
                                 marker=dict(size=4, color=cloud.sharpe, colorscale="Viridis", showscale=True,
                                             colorbar=dict(title="Sharpe")), opacity=0.5))
        fig.add_trace(go.Scatter(x=curve.vol, y=curve.ret, mode="lines", name="Frontera",
                                 line=dict(color="white", width=2)))
        for lbl, ww, col in [("Actual", w_cur, COLORS["cur"]), ("Óptima", w_opt, COLORS["opt"])]:
            wv = ww.reindex(mu.index).fillna(0).values
            fig.add_trace(go.Scatter(x=[np.sqrt(wv @ cov.values @ wv)], y=[wv @ mu.values], mode="markers",
                                     name=lbl, marker=dict(size=14, color=col, symbol="star")))
        vols = np.sqrt(np.diag(cov.values))
        fig.add_trace(go.Scatter(x=vols, y=mu.values, mode="markers+text", text=list(mu.index),
                                 textposition="top center", name="Activos", marker=dict(size=7, color="#bbb")))
        fig.update_layout(template=TEMPLATE, height=380, title="Frontera eficiente",
                          xaxis_title="Volatilidad", yaxis_title="Retorno", xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, **W)

    if "bl_posterior" in res:
        st.markdown("**Black-Litterman: retornos prior vs posterior**")
        st.dataframe(pd.DataFrame({"Prior (equilibrio)": res["bl_prior"], "Posterior": res["bl_posterior"],
                                   "Histórico": rets.mean() * TD}).style.format("{:.2%}"), **W)

    cov_s = rets.cov() * TD
    detail = pd.DataFrame({"Peso actual": w_cur, "Peso óptimo": w_opt,
                           "Cambio": w_opt - w_cur,
                           "Riesgo actual": risk_contrib(w_cur.values, cov_s.values),
                           "Riesgo óptimo": risk_contrib(w_opt.reindex(cov_s.index).values, cov_s.values)})
    st.dataframe(style_signed(detail.style.format("{:.1%}"), subset=["Cambio"]), **W)

    with st.expander("📊 Métricas avanzadas: actual vs óptima"):
        st.dataframe(metrics_table({"Actual": m_cur, "Óptima": m_opt}), **W)
        pr = port_returns(rets, w_opt)
        fig = go.Figure(go.Histogram(x=pr, nbinsx=60, marker_color="rgba(0,204,150,0.7)", name="Retornos"))
        fig.add_vline(x=m_opt["VaR 95% (diario)"], line_dash="dash", line_color="red", annotation_text="VaR 95%")
        fig.add_vline(x=m_opt["CVaR 95% (diario)"], line_dash="dot", line_color="orange", annotation_text="CVaR")
        fig.update_layout(template=TEMPLATE, title="Distribución de retornos diarios (óptima)",
                          xaxis_tickformat=".1%", height=330)
        st.plotly_chart(fig, **W)

    a, b = st.columns([2, 1])
    new_name = a.text_input("Guardar la cartera óptima como", f"{ctx['name']} · {res['method'].split(' ')[0]}",
                            key="opt_save_name")
    if b.button("💾 Guardar como cartera", key="opt_save", **W):
        keep = w_opt[w_opt > 0.001]
        portfolios[new_name] = {"tickers": list(keep.index), "weights": normalize(keep.values).tolist()}
        persist(portfolios, f"Guardada '{new_name}'")


def section_backtest(ctx):
    rets, rf = ctx["rets"], ctx["rf"]
    res = current_opt(ctx)
    st.subheader("Backtest dentro de muestra")
    c1, c2, c3 = st.columns(3)
    freq_lbl = c1.selectbox("Rebalanceo", list(FREQ), index=2, key="bt_freq")
    cost = c2.number_input("Costo por operación (bps)", 0.0, 200.0, 10.0, 5.0, key="bt_cost")
    log = c3.checkbox("Escala logarítmica", False, key="bt_log")
    freq = FREQ[freq_lbl]

    curves = {f"{ctx['name']} (actual)": backtest(rets, ctx["w_cur"], freq, cost)}
    if res is not None:
        curves["Óptima"] = backtest(rets, pd.Series(res["weights"], index=res["tickers"]), freq, cost)
    curves["Pesos iguales"] = backtest(rets, pd.Series(1.0, index=rets.columns), freq, cost)
    if ctx["bench_r"] is not None:
        curves[ctx["bench"]] = ctx["bench_r"].reindex(rets.index).fillna(0)
    st.plotly_chart(chart_equity(curves, log=log), **W)
    st.dataframe(metrics_table({k: perf_metrics(v, rf, ctx["bench_r"]) for k, v in curves.items()}), **W)
    if res is not None:
        st.caption("⚠️ La cartera 'Óptima' fue calculada con estos mismos datos: su backtest está sesgado a favor.")

    st.markdown("---")
    st.subheader("🧪 Walk-forward (fuera de muestra)")
    st.caption("En cada rebalanceo se optimiza usando sólo la ventana pasada y se mantiene hasta el siguiente. "
               "Es la forma honesta de saber si el método agrega valor.")
    a, b, c = st.columns(3)
    wf_method = a.selectbox("Método", ["Markowitz", "Risk Parity", "HRP", "Pesos Iguales"], key="wf_method")
    wf_obj = b.selectbox("Objetivo (Markowitz)", ["Máximo Sharpe", "Mínima Volatilidad"], key="wf_obj",
                         disabled=wf_method != "Markowitz")
    lookback = c.select_slider("Ventana de estimación (días)", [63, 126, 252, 504], 252, key="wf_lb")
    d, e, f = st.columns(3)
    wf_freq_lbl = d.selectbox("Re-optimizar", ["Mensual", "Trimestral", "Anual"], index=1, key="wf_freq")
    wf_maxw = e.slider("Peso máximo", 0.1, 1.0, 1.0, 0.05, key="wf_maxw")
    wf_cost = f.number_input("Costo (bps)", 0.0, 200.0, 10.0, 5.0, key="wf_cost")

    if len(rets) <= lookback + 21:
        st.info(f"Se necesitan más de {lookback + 21} días de datos (hay {len(rets)}). Ampliá el rango de fechas.")
        return
    if st.button("▶️ Correr walk-forward", type="primary", key="wf_run"):
        with st.spinner("Re-optimizando período a período..."):
            p = {"objective": wf_obj, "max_w": wf_maxw, "shrink": PYPFOPT_OK}
            wf_r, wf_w = walk_forward(rets, wf_method, rf, p, lookback, FREQ[wf_freq_lbl], wf_cost)
            st.session_state.wf = {"key": ctx["key"], "r": wf_r, "w": wf_w, "label": f"WF {wf_method}",
                                   "freq": FREQ[wf_freq_lbl], "cost": wf_cost}
    wf = st.session_state.get("wf")
    if wf and wf["key"] == ctx["key"] and wf["r"] is not None:
        idx = wf["r"].index
        sub = rets.loc[idx]
        comp = {wf["label"]: wf["r"],
                "Actual (mismo período)": backtest(sub, ctx["w_cur"], wf["freq"], wf["cost"]),
                "Pesos iguales": backtest(sub, pd.Series(1.0, index=sub.columns), wf["freq"], wf["cost"])}
        if ctx["bench_r"] is not None:
            comp[ctx["bench"]] = ctx["bench_r"].reindex(idx).fillna(0)
        st.plotly_chart(chart_equity(comp, title="Walk-forward fuera de muestra"), **W)
        st.dataframe(metrics_table({k: perf_metrics(v, rf, ctx["bench_r"]) for k, v in comp.items()}), **W)
        if wf["w"] is not None and not wf["w"].empty:
            fig = px.area(wf["w"], template=TEMPLATE, title="Evolución de pesos en cada re-optimización")
            fig.update_layout(height=330, yaxis_tickformat=".0%", legend_title=None)
            st.plotly_chart(fig, **W)


def section_rebalance(ctx):
    res = current_opt(ctx)
    if res is None:
        st.info("Primero optimizá la cartera en la pestaña 🚀 Optimización.")
        return
    prices = ctx["prices"]
    last = prices.iloc[-1]
    tgt = pd.Series(res["weights"], index=res["tickers"])
    cur_w = ctx["w_cur"]

    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Umbral de drift", 0.01, 0.20, 0.05, 0.01, key="rb_thr")
    drift = (tgt - cur_w.reindex(tgt.index).fillna(0)).abs()
    c2.metric("Drift máximo", f"{drift.max():.1%}")
    c3.metric("Estado", "⚠️ REBALANCEAR" if drift.max() > thr else "✅ Dentro del umbral")

    st.markdown("**Tenencias actuales** (editá las cantidades reales si las tenés)")
    a, b, c = st.columns(3)
    capital = a.number_input("Capital de referencia", 100.0, 1e10, 1_000_000.0, 10000.0, key="rb_cap",
                             help="Se usa para precargar cantidades según los pesos actuales.")
    comm = b.number_input("Comisión (%)", 0.0, 5.0, 0.5, 0.05, key="rb_comm") / 100
    lots = c.checkbox("Cantidades enteras", True, key="rb_int")

    base_key = (ctx["key"], capital)
    if st.session_state.get("rb_base_key") != base_key:
        q0 = (capital * cur_w / last).reindex(tgt.index).fillna(0)
        st.session_state.rb_hold = pd.DataFrame({"Activo": tgt.index, "Cantidad actual": np.floor(q0.values) if lots else q0.values})
        st.session_state.rb_base_key = base_key
    hold = st.data_editor(st.session_state.rb_hold, hide_index=True, key=f"rb_editor_{abs(hash(base_key))}", **W,
                          column_config={"Activo": st.column_config.TextColumn(disabled=True)})

    q_cur = pd.Series(pd.to_numeric(hold["Cantidad actual"], errors="coerce").fillna(0).values, index=hold["Activo"])
    px_last = last.reindex(q_cur.index)
    val_cur = q_cur * px_last
    total = val_cur.sum()
    if total <= 0:
        st.warning("El valor total de las tenencias es 0.")
        return
    q_tgt = total * tgt.reindex(q_cur.index) / px_last
    if lots:
        q_tgt = np.floor(q_tgt)
    diff = q_tgt - q_cur
    orders = pd.DataFrame({
        "Activo": q_cur.index, "Precio": px_last.values, "Cant. actual": q_cur.values,
        "Cant. objetivo": q_tgt.values, "Operar": diff.values, "Monto": (diff * px_last).values,
        "Peso actual": (val_cur / total).values, "Peso objetivo": tgt.reindex(q_cur.index).values,
    })
    orders["Acción"] = np.where(orders.Operar > 0, "🟢 COMPRAR", np.where(orders.Operar < 0, "🔴 VENDER", "—"))
    orders["Comisión"] = orders.Monto.abs() * comm
    orders = orders[orders.Operar.abs() > 1e-9]
    if orders.empty:
        st.success("No hay operaciones necesarias.")
        return
    st.dataframe(orders.style.format({"Precio": "{:,.2f}", "Cant. actual": "{:,.2f}", "Cant. objetivo": "{:,.2f}",
                                      "Operar": "{:+,.2f}", "Monto": "{:+,.0f}", "Peso actual": "{:.1%}",
                                      "Peso objetivo": "{:.1%}", "Comisión": "{:,.0f}"}), hide_index=True, **W)
    k = st.columns(4)
    k[0].metric("Valor cartera", f"{total:,.0f}")
    k[1].metric("Compras", f"{orders.loc[orders.Monto > 0, 'Monto'].sum():,.0f}")
    k[2].metric("Ventas", f"{-orders.loc[orders.Monto < 0, 'Monto'].sum():,.0f}")
    k[3].metric("Comisiones", f"{orders['Comisión'].sum():,.0f}")
    st.caption(f"Precios de cierre al {prices.index[-1]:%d/%m/%Y}, en la moneda de cada activo.")
    st.download_button("📥 Descargar órdenes (CSV)", orders.to_csv(index=False).encode("utf-8"), "ordenes.csv", "text/csv")


def section_simulation(ctx):
    rets = ctx["rets"]
    res = current_opt(ctx)
    options = ["Actual"] + (["Óptima"] if res is not None else [])
    c1, c2, c3, c4 = st.columns(4)
    which = c1.selectbox("Cartera", options, key="mc_which")
    method = c2.selectbox("Modelo", ["Bootstrap histórico", "GBM"], key="mc_model",
                          help="Bootstrap remuestrea retornos reales (respeta colas gordas). GBM asume normalidad.")
    days = c3.slider("Horizonte (días hábiles)", 21, 756, 252, 21, key="mc_days")
    sims = c4.selectbox("Simulaciones", [500, 1000, 5000, 10000], 2, key="mc_sims")
    capital = st.number_input("Capital inicial", 100.0, 1e10, 100_000.0, 1000.0, key="mc_cap")

    w = ctx["w_cur"] if which == "Actual" else pd.Series(res["weights"], index=res["tickers"])
    pr = port_returns(rets, w)
    if st.button("🔮 Simular", type="primary", key="mc_run"):
        paths = monte_carlo(pr, days, sims, "GBM" if method == "GBM" else "boot") * capital
        st.session_state.mc = {"key": (ctx["key"], which, method, days, sims, capital), "paths": paths}
    mc = st.session_state.get("mc")
    if not mc or mc["key"] != (ctx["key"], which, method, days, sims, capital):
        return
    paths = mc["paths"]
    x = np.arange(1, days + 1)
    p = {q: np.percentile(paths, q, axis=1) for q in [5, 25, 50, 75, 95]}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[p[95], p[5][::-1]], fill="toself",
                             fillcolor="rgba(0,204,150,0.12)", line=dict(width=0), name="P5–P95"))
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[p[75], p[25][::-1]], fill="toself",
                             fillcolor="rgba(0,204,150,0.25)", line=dict(width=0), name="P25–P75"))
    fig.add_trace(go.Scatter(x=x, y=p[50], line=dict(color=COLORS["opt"], width=2), name="Mediana"))
    for i in range(min(30, paths.shape[1])):
        fig.add_trace(go.Scatter(x=x, y=paths[:, i], line=dict(width=0.5, color="rgba(200,200,200,0.15)"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=capital, line_dash="dot", line_color="gray")
    fig.update_layout(template=TEMPLATE, height=430, title=f"Monte Carlo · {method}",
                      xaxis_title="Días", yaxis_title="Valor")
    st.plotly_chart(fig, **W)
    final = paths[-1]
    mdd = (paths / np.maximum.accumulate(paths, axis=0) - 1).min(axis=0)
    k = st.columns(6)
    k[0].metric("Mediana final", f"{np.median(final):,.0f}")
    k[1].metric("P5", f"{np.percentile(final, 5):,.0f}")
    k[2].metric("P95", f"{np.percentile(final, 95):,.0f}")
    k[3].metric("Prob. de pérdida", f"{np.mean(final < capital):.1%}")
    k[4].metric("Prob. caída > 20%", f"{np.mean(final < 0.8 * capital):.1%}")
    k[5].metric("DD mediano en camino", f"{np.median(mdd):.1%}")


def section_ai(ctx):
    if not ai_ready():
        st.warning("⚠️ Configurá una API key en la barra lateral.")
        return
    res = current_opt(ctx)
    options = ["Actual"] + (["Óptima"] if res is not None else [])
    which = st.radio("Analizar", options, horizontal=True, key="ai_which")
    focus = st.text_input("Foco opcional", placeholder="Ej: perfil moderado en pesos, horizonte 2 años, evitar bonos largos",
                          key="ai_focus")
    if which == "Actual":
        w = ctx["w_cur"]
        mu, cov = ctx["rets"].mean() * TD, ctx["rets"].cov() * TD
        r, v = float(w @ mu), float(np.sqrt(w @ cov @ w))
        target = {"weights": w.values, "tickers": list(w.index), "expected_return": r, "volatility": v,
                  "sharpe_ratio": (r - ctx["rf"]) / v if v else 0, "method": "Pesos actuales"}
    else:
        target = res
    if st.button("🧠 Generar informe", type="primary", key="ai_run"):
        context = build_portfolio_context(target, ctx["prices"], ctx["name"], ctx["rf"], ctx["bench_r"])
        system = ("Sos el CIO de una gestora. Respondé en español rioplatense, con criterio profesional y concreto. "
                  "Basate en los datos provistos; si algo no se puede concluir con ellos, decilo. "
                  "No inventes cifras. Recordá que las métricas son históricas.")
        prompt = (f"{context}\n\n{('Foco del cliente: ' + focus) if focus else ''}\n\n"
                  "Estructura: 1) Diagnóstico de diversificación (moneda, clase, correlaciones, concentración del riesgo). "
                  "2) Riesgos principales y escenarios adversos. 3) Ajustes concretos con pesos sugeridos. "
                  "4) Alertas y qué monitorear. Cerrá con un resumen de 3 líneas.")
        with st.spinner("La IA está analizando..."):
            try:
                st.session_state.ai_report = {"key": ctx["key"], "text": call_ai([{"role": "user", "content": prompt}], system)}
            except Exception as e:
                st.error(f"Error IA: {e}")
    rep = st.session_state.get("ai_report")
    if rep and rep["key"] == ctx["key"]:
        st.markdown(rep["text"])
        st.download_button("📥 Descargar informe", rep["text"], f"informe_{ctx['name']}.md", "text/markdown")


def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo")
    portfolios = st.session_state.portfolios
    if not portfolios:
        st.info("👋 Todavía no hay carteras. Creá la primera acá abajo.")
        section_manage(portfolios)
        return

    ctx = load_dashboard_data(portfolios)
    tabs = st.tabs(["💼 Gestión", "📈 Análisis", "🚀 Optimización", "🧪 Backtest",
                    "🔄 Rebalanceo", "🔮 Simulación", "🧠 Informe IA"])
    with tabs[0]:
        section_manage(portfolios)
    sections = [section_analysis, lambda c: section_optimize(c, portfolios), section_backtest,
                section_rebalance, section_simulation, section_ai]
    for tab, fn in zip(tabs[1:], sections):
        with tab:
            if ctx is None:
                st.info("Elegí una cartera y un rango de fechas con datos.")
            else:
                fn(ctx)


# ═══════════════════════════════════════════════════════════════════════════
#  RENTA FIJA
# ═══════════════════════════════════════════════════════════════════════════

def page_fixed_income():
    st.title("🏛️ Renta Fija")
    st.caption("Modelo de bono bullet (sin amortización). AL30/GD30 y otros soberanos amortizan: "
               "tomá estos números como aproximación o cargá la vida promedio en 'Años'.")
    if "bonds_init" not in st.session_state:
        st.session_state.bonds_init = pd.DataFrame({
            "Bono": ["AL30", "GD30", "TX26"], "Cupón (%)": [0.75, 0.75, 2.0], "YTM (%)": [15.0, 13.0, 8.0],
            "Años": [3.5, 3.5, 1.2], "Pagos/año": [2, 2, 2], "Nominal": [100000, 150000, 50000]})
    df = st.data_editor(st.session_state.bonds_init, num_rows="dynamic", key="bond_editor", **W,
                        column_config={"Pagos/año": st.column_config.NumberColumn(min_value=1, max_value=12, step=1)})

    rows = []
    for _, r in df.iterrows():
        try:
            if pd.isna(r["Bono"]) or float(r["Años"]) <= 0:
                continue
            p, mac, mod, conv = bond_analytics(float(r["Cupón (%)"]), float(r["YTM (%)"]),
                                               float(r["Años"]), safe_int(r.get("Pagos/año", 2)))
            nom = float(r["Nominal"])
            rows.append({"Bono": r["Bono"], "Precio (% VN)": p, "Dur. Macaulay": mac, "Dur. Modificada": mod,
                         "Convexidad": conv, "Nominal": nom, "Valor mercado": nom * p / 100,
                         "DV01": nom * p / 100 * mod * 1e-4, "YTM": float(r["YTM (%)"]) / 100})
        except (TypeError, ValueError):
            continue
    if not rows:
        st.info("Cargá al menos un bono válido.")
        return
    dr = pd.DataFrame(rows)
    mv = dr["Valor mercado"].sum()
    dr["Peso"] = dr["Valor mercado"] / mv
    k = st.columns(5)
    k[0].metric("Valor de mercado", f"${mv:,.0f}")
    k[1].metric("Duration mod.", f"{(dr['Dur. Modificada'] * dr.Peso).sum():.2f}")
    k[2].metric("Convexidad", f"{(dr.Convexidad * dr.Peso).sum():.2f}")
    k[3].metric("YTM ponderada", f"{(dr.YTM * dr.Peso).sum():.2%}")
    k[4].metric("DV01 cartera", f"${dr.DV01.sum():,.0f}")
    st.dataframe(dr.style.format({"Precio (% VN)": "{:.2f}", "Dur. Macaulay": "{:.2f}", "Dur. Modificada": "{:.2f}",
                                  "Convexidad": "{:.2f}", "Nominal": "{:,.0f}", "Valor mercado": "{:,.0f}",
                                  "DV01": "{:,.0f}", "YTM": "{:.2%}", "Peso": "{:.1%}"}), hide_index=True, **W)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Escenarios de tasa (shock paralelo)**")
        shocks = [-300, -200, -100, -50, 50, 100, 200, 300]
        sc = {}
        for s in shocks:
            dy = s / 1e4
            sc[f"{s:+d} bps"] = [(-row["Dur. Modificada"] * dy + 0.5 * row.Convexidad * dy ** 2) * row["Valor mercado"]
                                 for _, row in dr.iterrows()]
        scen = pd.DataFrame(sc, index=dr.Bono)
        scen.loc["TOTAL"] = scen.sum()
        st.dataframe(style_signed(scen.style.format("{:+,.0f}")), **W)
    with c2:
        sel = st.selectbox("Curva precio-tasa", dr.Bono, key="fi_sel")
        row = df[df.Bono == sel].iloc[0]
        ys = np.linspace(max(0.1, float(row["YTM (%)"]) - 10), float(row["YTM (%)"]) + 10, 80)
        prices_ = [bond_analytics(float(row["Cupón (%)"]), y, float(row["Años"]), safe_int(row["Pagos/año"]))[0] for y in ys]
        fig = go.Figure(go.Scatter(x=ys, y=prices_, line=dict(color=COLORS["opt"])))
        fig.add_vline(x=float(row["YTM (%)"]), line_dash="dot")
        fig.update_layout(template=TEMPLATE, height=330, xaxis_title="YTM (%)", yaxis_title="Precio (% VN)",
                          title=f"{sel}: convexidad precio-rendimiento")
        st.plotly_chart(fig, **W)


# ═══════════════════════════════════════════════════════════════════════════
#  EXPLORADOR YAHOO
# ═══════════════════════════════════════════════════════════════════════════

def page_yahoo_explorer():
    st.title("🌎 Explorador Yahoo Finance")
    c1, c2 = st.columns([2, 1])
    t = c1.text_input("Ticker", "AAPL", key="yx_t").strip().upper()
    period = c2.selectbox("Período", ["6mo", "1y", "2y", "5y", "max"], 1, key="yx_p")
    if not t:
        return
    try:
        s = yf.Ticker(t)
        h = s.history(period=period, auto_adjust=False)
        if h.empty:
            st.error("Sin datos para ese ticker.")
            return
        try:
            info = s.info or {}
        except Exception:
            info = {}
        last = info.get("currentPrice") or info.get("regularMarketPrice") or h["Close"].iloc[-1]
        prev = h["Close"].iloc[-2] if len(h) > 1 else last
        k = st.columns(6)
        k[0].metric("Precio", fmt_num(last), f"{(last / prev - 1):+.2%}" if prev else None)
        k[1].metric("Market cap", fmt_big(info.get("marketCap")))
        k[2].metric("P/E", fmt_num(info.get("trailingPE")))
        k[3].metric("Beta", fmt_num(info.get("beta")))
        k[4].metric("Div. yield", fmt_num(info.get("dividendYield"), "{:.2f}%") if info.get("dividendYield") else "N/A")
        k[5].metric("Sector", str(info.get("sector") or info.get("quoteType") or "N/A"))

        h["SMA50"] = h["Close"].rolling(50).mean()
        h["SMA200"] = h["Close"].rolling(200).mean()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=h.index, open=h.Open, high=h.High, low=h.Low, close=h.Close, name=t), 1, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h.SMA50, name="SMA 50", line=dict(width=1.2, color="#FFA15A")), 1, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h.SMA200, name="SMA 200", line=dict(width=1.2, color="#19D3F3")), 1, 1)
        fig.add_trace(go.Bar(x=h.index, y=h.Volume, name="Volumen", marker_color="rgba(150,150,150,0.5)"), 2, 1)
        fig.update_layout(template=TEMPLATE, height=600, xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, **W)

        r = h["Close"].pct_change().dropna()
        m = perf_metrics(r, 0.0)
        st.dataframe(metrics_table({t: m}).T, **W)
        if info.get("longBusinessSummary"):
            with st.expander("Descripción de la empresa"):
                st.write(info["longBusinessSummary"])
    except Exception as e:
        st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  IA: EVENTOS Y CHAT
# ═══════════════════════════════════════════════════════════════════════════

def page_event_analyzer():
    st.header("📰 Analizador de Noticias y Documentos")
    if not ai_ready():
        st.warning("⚠️ Configurá una API key en la barra lateral.")
        return
    txt = st.text_area("Pegá la noticia o análisis", height=160, key="ev_txt")
    up = st.file_uploader("…o subí un archivo (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "md", "csv"], key="ev_file")
    pf_names = list(st.session_state.portfolios)
    pf_sel = st.selectbox("Evaluar impacto sobre cartera (opcional)", ["—"] + pf_names, key="ev_pf")
    if st.button("Analizar", type="primary", key="ev_run"):
        body = (txt or "") + ("\n\n" + extract_text_from_file(up) if up else "")
        if not body.strip():
            st.warning("Pegá un texto o subí un archivo.")
            return
        extra = ""
        if pf_sel != "—":
            p = st.session_state.portfolios[pf_sel]
            extra = "\n\nCartera del usuario:\n" + "\n".join(f"- {t}: {w:.1%}" for t, w in zip(p["tickers"], p["weights"]))
        prompt = (f"Analizá como estratega financiero:\n\n{body}{extra}\n\n"
                  "1) Resumen en 3 líneas 2) Impacto por mercado/sector (corto y mediano plazo) "
                  "3) Activos más afectados (+/-) 4) Impacto en la cartera si se proveyó 5) Qué vigilar.")
        with st.spinner("Analizando..."):
            try:
                st.session_state.ev_out = call_ai([{"role": "user", "content": prompt}],
                                                  "Respondé en español, conciso y sin inventar datos.")
            except Exception as e:
                st.error(f"Error: {e}")
    if st.session_state.get("ev_out"):
        st.markdown(st.session_state.ev_out)


def page_chat_general():
    st.header("💬 Chat IA")
    if not ai_ready():
        st.warning("⚠️ Configurá una API key en la barra lateral.")
        return
    c1, c2 = st.columns([3, 1])
    pf_names = list(st.session_state.portfolios)
    ctx_pf = c1.selectbox("Contexto de cartera", ["Ninguno"] + pf_names, key="chat_ctx")
    if c2.button("🧹 Nueva conversación", **W):
        st.session_state.msgs = []
        st.rerun()
    st.session_state.setdefault("msgs", [])
    for m in st.session_state.msgs:
        st.chat_message(m["role"]).markdown(m["content"])
    if q := st.chat_input("Consultá lo que quieras..."):
        st.session_state.msgs.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        system = "Sos un analista financiero senior. Respondé en español, claro y con fundamento."
        if ctx_pf != "Ninguno":
            p = st.session_state.portfolios[ctx_pf]
            system += f"\nCartera del usuario '{ctx_pf}': " + ", ".join(
                f"{t} {w:.1%}" for t, w in zip(p["tickers"], p["weights"]))
        with st.chat_message("assistant"), st.spinner("Pensando..."):
            try:
                ans = call_ai(st.session_state.msgs[-20:], system)
                st.markdown(ans)
                st.session_state.msgs.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.session_state.msgs.pop()
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  ASISTENTE QUANT
# ═══════════════════════════════════════════════════════════════════════════

IOL_FALLBACK = {
    "Acciones": ["GGAL", "YPFD", "PAMP", "CEPU", "ALUA", "TXAR", "BMA", "SUPV", "LOMA", "TGSU2", "MIRG", "COME", "VALO", "BYMA", "TRAN", "EDN"],
    "CEDEARs": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "KO", "MCD", "V", "SPY", "QQQ", "BRKB", "MELI", "VIST"],
    "Bonos": ["AL30", "GD30", "AL35", "GD35", "AL41", "GD41", "GD38", "GD46", "AE38", "TX26", "TX28"],
    "ONs": ["YMCXO", "PNDCO", "TLC5O", "MGCHO", "IRCFO"],
}


@st.cache_data(ttl=1800, show_spinner=False)
def get_iol_tickers(connected):
    client = get_iol_client() if connected else None
    cats = {k: list(v) for k, v in IOL_FALLBACK.items()}
    if client is None:
        return cats, False
    live = False
    for c in cats:
        try:
            df = client.get_instruments(category=c.lower())
            if df is not None and not df.empty and "simbolo" in df.columns:
                cats[c] = df["simbolo"].dropna().unique().tolist()[:40]
                live = True
        except Exception:
            pass
    return cats, live


YAHOO_UNIVERSE = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "CRM", "ADBE"],
    "S&P 500 ETFs": ["SPY", "VOO", "IVV", "SPLG", "RSP"],
    "Nasdaq ETFs": ["QQQ", "QQQM"],
    "Sectoriales": ["XLE", "XLF", "XLV", "XLK", "XLP", "XLU", "XLI"],
    "Bonos": ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "BIL", "TIP"],
    "Commodities": ["GLD", "IAU", "SLV", "USO", "DBA", "PDBC"],
    "Emergentes": ["EEM", "VWO", "IEMG", "FXI", "EWZ", "INDA", "ARGT"],
    "Dividendos": ["SCHD", "VYM", "DGRO", "DVY", "HDV", "NOBL"],
}


@st.cache_data(ttl=900, show_spinner=False)
def compute_movers(symbols):
    raw = yf.download(list(symbols), period="1mo", auto_adjust=True, progress=False, threads=True)
    close = _extract_close(raw, list(symbols)).ffill()
    if close.empty or len(close) < 6:
        return pd.DataFrame()
    out = pd.DataFrame({"Último": close.iloc[-1], "1D": close.iloc[-1] / close.iloc[-2] - 1,
                        "5D": close.iloc[-1] / close.iloc[-6] - 1, "1M": close.iloc[-1] / close.iloc[0] - 1})
    return out.dropna()


def page_ai_strategy_assistant():
    st.header("🧠 Asistente Quant")
    iol, iol_live = get_iol_tickers(bool(st.session_state.get("iol_connected")))
    tabs = st.tabs(["🎯 Generador IA", "🏦 Tickers IOL", "🌎 Tickers Yahoo", "🔥 Movers"])

    with tabs[0]:
        if not ai_ready():
            st.warning("⚠️ Configurá una API key en la barra lateral.")
        else:
            strat = st.text_area("Describí tu estrategia", height=120, key="q_strat",
                                 placeholder="Ej: Cartera moderada, 40% renta fija en dólares, algo de acciones argentinas y dividendos USA")
            c1, c2, c3 = st.columns(3)
            risk = c1.select_slider("Riesgo", ["Conservador", "Moderado", "Agresivo", "Muy agresivo"], "Moderado", key="q_risk")
            hor = c2.selectbox("Horizonte", ["Corto (<1 año)", "Mediano (1-3 años)", "Largo (>3 años)"], 1, key="q_hor")
            mkts = c3.multiselect("Mercados", ["Argentina", "USA"], ["Argentina", "USA"], key="q_mkt")
            if st.button("✨ Generar estrategia", type="primary", key="q_run"):
                if not strat.strip():
                    st.warning("Describí tu estrategia.")
                elif not mkts:
                    st.warning("Elegí al menos un mercado.")
                else:
                    allowed = {}
                    if "Argentina" in mkts:
                        allowed["argentina"] = sorted({t for v in iol.values() for t in v})
                    if "USA" in mkts:
                        allowed["usa"] = sorted({t for v in YAHOO_UNIVERSE.values() for t in v})
                    lists = "\n".join(f"- {k}: {safe_join_list(v)}" for k, v in allowed.items())
                    system = ("Sos un gestor cuantitativo senior. Respondé EXCLUSIVAMENTE con un objeto JSON válido, "
                              "sin markdown.")
                    prompt = f"""Diseñá una estrategia de inversión.
Perfil: {risk} | Horizonte: {hor} | Mercados: {', '.join(mkts)}
Descripción del usuario: {strat}

Tickers permitidos por mercado (usá SOLO estos, cada ticker en su mercado):
{lists}

Formato:
{{"strategy_name": str, "risk_profile": str,
  "asset_allocation": {{"acciones": float, "renta_fija": float, "etfs": float, "liquidez": float}},
  "portfolios": {{{', '.join(f'"{k}": [{{"ticker": str, "weight": float, "reason": str}}]' for k in allowed)}}},
  "expected_metrics": {{"expected_return": str, "volatility": str, "sharpe_target": str}},
  "rebalancing_frequency": str, "risks": [str], "notes": str}}
Los pesos de CADA mercado deben sumar 1.0. asset_allocation suma 1.0."""
                    with st.spinner("🤖 Diseñando estrategia..."):
                        try:
                            raw = call_ai([{"role": "user", "content": prompt}], system, temperature=0.3, json_mode=True)
                            data = extract_json(raw)
                            warnings = []
                            for mk, assets in list(data.get("portfolios", {}).items()):
                                ok_set = set(allowed.get(mk, []))
                                clean = []
                                for a in assets or []:
                                    t = str(a.get("ticker", "")).upper().strip()
                                    if ok_set and t not in ok_set:
                                        warnings.append(f"{t} ({mk}) no está en la lista permitida y se descartó")
                                        continue
                                    try:
                                        wv = float(a.get("weight", 0))
                                    except (TypeError, ValueError):
                                        wv = 0.0
                                    clean.append({"ticker": t, "weight": wv, "reason": a.get("reason", "")})
                                if clean:
                                    ws = normalize([c["weight"] for c in clean])
                                    for c, wv in zip(clean, ws):
                                        c["weight"] = float(wv)
                                data["portfolios"][mk] = clean
                            st.session_state.last_quant = {"data": data, "warnings": warnings}
                        except Exception as e:
                            st.error(f"❌ {type(e).__name__}: {e}")
                            if "raw" in locals():
                                st.code(raw[:3000])

        lq = st.session_state.get("last_quant")
        if lq:
            data = lq["data"]
            for w_ in lq["warnings"]:
                st.caption(f"⚠️ {w_}")
            rt = st.tabs(["📊 Resumen", "🥧 Asignación", "📋 Tickers", "💾 Exportar"])
            with rt[0]:
                c1, c2, c3 = st.columns(3)
                c1.metric("Estrategia", str(data.get("strategy_name", "N/A")))
                c2.metric("Riesgo", str(data.get("risk_profile", "N/A")))
                c3.metric("Rebalanceo", str(data.get("rebalancing_frequency", "N/A")))
                em = data.get("expected_metrics", {}) or {}
                st.write(f"**Retorno esperado:** {em.get('expected_return', '—')} · **Volatilidad:** "
                         f"{em.get('volatility', '—')} · **Sharpe objetivo:** {em.get('sharpe_target', '—')}")
                if data.get("risks"):
                    st.markdown("**Riesgos:**\n" + "\n".join(f"- {r}" for r in data["risks"]))
                if data.get("notes"):
                    st.info(data["notes"])
                st.caption("Las métricas esperadas las estima el modelo de lenguaje: validalas en el Dashboard.")
            with rt[1]:
                alloc = {k: v for k, v in (data.get("asset_allocation") or {}).items() if isinstance(v, (int, float)) and v > 0}
                if alloc:
                    st.plotly_chart(px.pie(names=list(alloc), values=list(alloc.values()), hole=0.45,
                                           template=TEMPLATE), **W)
            with rt[2]:
                for mk, assets in data.get("portfolios", {}).items():
                    if not assets:
                        continue
                    st.markdown(f"**{mk.upper()}**")
                    st.dataframe(pd.DataFrame(assets).style.format({"weight": "{:.1%}"}), hide_index=True, **W)
                    if st.button(f"📤 Enviar al Dashboard ({mk})", key=f"send_{mk}"):
                        tks = [a["ticker"] for a in assets]
                        if mk == "argentina":  # explícito: precios locales en BYMA
                            tks = [t if "." in t else t + ".BA" for t in tks]
                        st.session_state.quant_pf = {"tickers": tks, "weights": [a["weight"] for a in assets],
                                                     "name": f"{data.get('strategy_name', 'IA')} · {mk}"}
                        st.success("Listo. Andá a Dashboard → Gestión → 📥 Importar de IA.")
            with rt[3]:
                name = re.sub(r"\W+", "_", str(data.get("strategy_name", "quant")))
                st.download_button("📥 JSON", json.dumps(data, indent=2, ensure_ascii=False), f"strategy_{name}.json")
                rows = [{"Mercado": m, "Ticker": a["ticker"], "Peso": a["weight"], "Razón": a.get("reason", "")}
                        for m, aa in data.get("portfolios", {}).items() for a in aa]
                if rows:
                    st.download_button("📥 CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), f"tickers_{name}.csv")

    with tabs[1]:
        st.subheader("🏦 Tickers IOL")
        st.caption("🟢 Listado en vivo desde IOL" if iol_live else "📋 Lista de referencia (conectá IOL para el listado en vivo)")
        for k, v in iol.items():
            with st.expander(f"{k} ({len(v)})"):
                st.write(" · ".join(f"`{t}`" for t in sorted(v)))

    with tabs[2]:
        st.subheader("🌎 Tickers Yahoo")
        for k, v in YAHOO_UNIVERSE.items():
            with st.expander(f"{k} ({len(v)})"):
                st.write(" · ".join(f"`{t}`" for t in v))

    with tabs[3]:
        st.subheader("🔥 Movers")
        univ = st.radio("Universo", ["USA", "Argentina (BYMA)"], horizontal=True, key="mv_univ")
        syms = (tuple(sorted({t for v in YAHOO_UNIVERSE.values() for t in v})) if univ == "USA"
                else tuple(t + ".BA" for t in IOL_FALLBACK["Acciones"]))
        if st.button("🔄 Actualizar", key="mv_ref"):
            compute_movers.clear()
        with st.spinner("Descargando cotizaciones..."):
            try:
                mv = compute_movers(syms)
            except Exception as e:
                mv = pd.DataFrame()
                st.error(f"Error: {e}")
        if mv.empty:
            st.info("No se pudieron obtener cotizaciones.")
        else:
            h = st.radio("Horizonte", ["1D", "5D", "1M"], horizontal=True, key="mv_h")
            c1, c2 = st.columns(2)
            fmt = {"Último": "{:,.2f}", "1D": "{:+.2%}", "5D": "{:+.2%}", "1M": "{:+.2%}"}
            c1.markdown("**🟢 Mayores subas**")
            c1.dataframe(mv.sort_values(h, ascending=False).head(10).style.format(fmt), **W)
            c2.markdown("**🔴 Mayores bajas**")
            c2.dataframe(mv.sort_values(h).head(10).style.format(fmt), **W)
            fig = px.bar(mv.sort_values(h), x=mv.sort_values(h).index, y=h, color=h,
                         color_continuous_scale="RdYlGn", template=TEMPLATE)
            fig.update_layout(height=360, yaxis_tickformat=".1%", xaxis_title=None, coloraxis_showscale=False)
            st.plotly_chart(fig, **W)


# ═══════════════════════════════════════════════════════════════════════════
#  INICIO
# ═══════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("📈 INVERSIONES PRO")
    st.markdown("#### Plataforma integral de gestión de portafolios")
    pf = st.session_state.portfolios
    c = st.columns(4)
    c[0].metric("Carteras", len(pf))
    c[1].metric("Activos únicos", len({t for p in pf.values() for t in p["tickers"]}))
    c[2].metric("Motor IA", st.session_state.get("preferred_ai") or "—")
    c[3].metric("Almacenamiento", "Google Sheets" if get_gsheets_client() else "Local")
    st.markdown("""
| Módulo | Qué hace |
|---|---|
| 📊 **Dashboard** | Análisis vs benchmark, optimización (Markowitz, Risk Parity, HRP, Black-Litterman), frontera eficiente, backtest y walk-forward, rebalanceo con órdenes, Monte Carlo, informe IA |
| 🏛️ **Renta Fija** | Precio, duration, convexidad, DV01 y escenarios de tasa |
| 🧠 **Asistente Quant** | La IA propone una cartera con tickers reales y la envía al Dashboard |
| 🌎 **Yahoo / 🏦 IOL** | Exploración de activos |
| 📰 / 💬 | Análisis de noticias y documentos, chat con contexto de cartera |
""")
    if not PYPFOPT_OK:
        st.info("💡 Instalá `PyPortfolioOpt` para habilitar HRP y covarianza Ledoit-Wolf (el resto funciona igual).")
    st.caption("Herramienta de análisis. Las métricas son históricas y no garantizan resultados futuros.")


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR & ROUTER
# ═══════════════════════════════════════════════════════════════════════════

def sidebar():
    sb = st.sidebar
    sb.title("⚙️ Configuración")
    client, msg = _gsheets_connect()
    if client:
        sb.success("🟢 Google Sheets conectado")
    else:
        sb.info(f"📁 Almacenamiento local · {msg}")
    if st.session_state.get("storage_error"):
        sb.caption(f"⚠️ {st.session_state.storage_error}")

    sb.selectbox("Tickers sin sufijo", ["Auto", "USA", "BYMA"], key="market_mode",
                 help="Auto: los tickers locales conocidos (GGAL, AL30…) van a BYMA (.BA) y el resto a USA. "
                      "BYMA: todos como .BA (ej. AAPL → CEDEAR en pesos). USA: ninguno.")

    st.session_state.setdefault("openai_api_key", secret("openai", "api_key", "") or "")
    st.session_state.setdefault("gemini_api_key", secret("gemini", "api_key", "") or "")
    with sb.expander("🤖 IA (OpenAI)", expanded=False):
        st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        m = st.selectbox("Modelo", ["gpt-4o", "gpt-4o-mini", "Otro…"], key="openai_model_sel")
        st.session_state.openai_model = (st.text_input("Nombre del modelo", "gpt-4.1", key="openai_model_custom")
                                         if m == "Otro…" else m)
    with sb.expander("🧠 IA (Gemini)", expanded=False):
        st.text_input("Gemini API Key", type="password", key="gemini_api_key")
        m = st.selectbox("Modelo", ["gemini-2.5-flash", "gemini-2.5-pro", "Otro…"], key="gemini_model_sel")
        st.session_state.gemini_model = (st.text_input("Nombre del modelo", "gemini-2.5-flash-lite", key="gemini_model_custom")
                                         if m == "Otro…" else m)

    ais = []
    if OPENAI_OK and st.session_state.get("openai_api_key"):
        ais.append("OpenAI")
    if GEMINI_OK and st.session_state.get("gemini_api_key"):
        ais.append("Gemini")
    if ais:
        st.session_state.preferred_ai = sb.radio("✨ Motor IA activo", ais, horizontal=True)
    else:
        st.session_state.preferred_ai = None
        sb.caption("⚠️ Ingresá una API key para habilitar la IA")

    with sb.expander("🏦 Conexión IOL", expanded=False):
        if not IOL_MODULE_OK:
            st.caption("iol_client.py no encontrado")
        u = st.text_input("Usuario IOL", value=st.session_state.get("iol_username", ""))
        p = st.text_input("Contraseña IOL", type="password", value=st.session_state.get("iol_password", ""))
        if st.button("Conectar", **W):
            st.session_state.iol_username, st.session_state.iol_password = u, p
            try:
                st.session_state.iol_connected = get_iol_client() is not None
            except Exception:
                st.session_state.iol_connected = False
            fetch_prices.clear()
            get_iol_tickers.clear()
        if st.session_state.get("iol_connected"):
            st.success(f"🟢 Conectado: {st.session_state.get('iol_username')}")
        else:
            st.caption("🔴 Desconectado")

    if sb.button("🔄 Recargar carteras y datos", **W):
        st.cache_data.clear()
        st.session_state.portfolios = load_portfolios()
        st.rerun()
    sb.markdown("---")


PAGES = {
    "🏠 Inicio": page_home,
    "📊 Dashboard Corporativo": page_corporate_dashboard,
    "🏛️ Renta Fija": page_fixed_income,
    "🧠 Asistente Quant": page_ai_strategy_assistant,
    "🏦 Explorador IOL": page_iol_explorer,
    "🌎 Yahoo Finance": page_yahoo_explorer,
    "📰 Analizador Eventos": page_event_analyzer,
    "💬 Chat IA General": page_chat_general,
}


def main():
    if "portfolios" not in st.session_state:
        st.session_state.portfolios = load_portfolios()
    sidebar()
    page = st.sidebar.radio("Navegación", list(PAGES), key="nav")
    PAGES[page]()


main()
