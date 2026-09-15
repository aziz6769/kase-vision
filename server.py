from __future__ import annotations

from pathlib import Path
import asyncio
import io
import os
import re
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

BASE = Path(__file__).resolve().parent
REAL_DATA = BASE / "data" / "kase_monthly_real.csv"
DEMO = BASE / "data" / "demo_prices.csv"

N_PORTFOLIOS = 60000
SEED = 42
MAX_WEIGHT = 0.35
RF_RATE = 0.08
RETURN_SHRINKAGE = 0.35
COV_SHRINKAGE = 0.15
BOOTSTRAP_SIMS = 4000

NAMES = {
    "HSBK": "Halyk Bank",
    "KSPI": "Kaspi.kz",
    "KZAP": "Kazatomprom",
    "KMGZ": "KazMunayGas",
    "KCEL": "Kcell",
    "KEGC": "KEGOC",
}

PROFILE_LABELS = {
    "min_risk": "Консервативный",
    "balanced": "Сбалансированный",
    "max_sharpe": "Рост / max Sharpe",
    "equal": "Равные веса",
}

app = FastAPI(title="KASE Vision", version="7.0")

CORS_ORIGINS = [
    x.strip()
    for x in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "source": "",
    "result": None,
    "weights": None,
    "portfolio_metrics": None,
    "research_df": None,
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        low = {str(c).lower(): c for c in df.columns}
        for alias in ("datetime", "timestamp", "trade_date"):
            if alias in low:
                df = df.rename(columns={low[alias]: "date"})
                break
    if "date" not in df.columns:
        raise ValueError("Нужна колонка date.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

    assets = [t for t in NAMES if t in df.columns]
    if len(assets) < 2:
        raise ValueError(
            "Нужно минимум два столбца с тикерами: "
            "HSBK, KSPI, KZAP, KMGZ, KCEL, KEGC."
        )
    for ticker in assets:
        df[ticker] = pd.to_numeric(df[ticker], errors="coerce")

    return df[["date"] + assets].dropna(subset=assets, how="all")


def metrics(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = RF_RATE):
    ret = float(w @ mu)
    risk = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / risk if risk > 1e-12 else 0.0
    return ret, risk, sharpe


def feasible_cloud(n_assets: int, count: int, rng: np.random.Generator) -> np.ndarray:
    chunks = []
    total = 0
    while total < count:
        batch = rng.dirichlet(np.ones(n_assets), size=max(10000, count - total))
        batch = batch[np.max(batch, axis=1) <= MAX_WEIGHT + 1e-12]
        if len(batch):
            chunks.append(batch)
            total += len(batch)
    return np.vstack(chunks)[:count]


def bootstrap_scenario(
    returns: pd.DataFrame,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    # Sample entire historical months, preserving the co-movement of all assets.
    rp = returns.values @ weights
    if len(rp) == 0:
        return {"p10": None, "median": None, "p90": None}
    idx = rng.integers(0, len(rp), size=(BOOTSTRAP_SIMS, 12))
    sampled = rp[idx]
    annual = np.prod(1.0 + sampled, axis=1) - 1.0
    p10, p50, p90 = np.percentile(annual, [10, 50, 90])
    return {
        "p10": round(float(p10), 6),
        "median": round(float(p50), 6),
        "p90": round(float(p90), 6),
    }


def profile_payload(
    key: str,
    weights: np.ndarray,
    assets: list[str],
    mu_model: np.ndarray,
    cov_model: np.ndarray,
    returns: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    ret, risk, sharpe = metrics(weights, mu_model, cov_model)
    trailing = returns.tail(min(12, len(returns))).values @ weights
    trailing_12m = float(np.prod(1.0 + trailing) - 1.0)
    whole = returns.values @ weights
    total_period = float(np.prod(1.0 + whole) - 1.0)

    order = np.argsort(weights)[::-1]
    leaders = [
        {"ticker": assets[i], "weight": round(float(weights[i]), 6)}
        for i in order
        if weights[i] >= 0.03
    ][:4]

    return {
        "key": key,
        "label": PROFILE_LABELS[key],
        "weights": {
            ticker: round(float(value), 6)
            for ticker, value in zip(assets, weights)
        },
        "return": round(ret, 6),
        "risk": round(risk, 6),
        "sharpe": round(sharpe, 6),
        "trailing_12m": round(trailing_12m, 6),
        "sample_total_return": round(total_period, 6),
        "scenario_12m": bootstrap_scenario(returns, weights, rng),
        "leaders": leaders,
    }


def calculate(df: pd.DataFrame) -> dict:
    clean = normalize(df)
    monthly = clean.set_index("date").resample("ME").last().dropna(how="any")
    if len(monthly) < 12:
        raise ValueError("Для исследовательской модели нужно минимум 12 общих месячных наблюдений.")

    assets = list(monthly.columns)
    returns = monthly.pct_change().dropna()

    raw_mu = returns.mean().values * 12.0
    raw_cov = returns.cov().values * 12.0

    # Stabilize noisy estimates: shrink returns toward the cross-sectional mean
    # and covariance toward its diagonal. This reduces extreme Markowitz weights
    # on a short monthly sample without inventing new price observations.
    grand_mu = float(np.mean(raw_mu))
    mu_model = (1.0 - RETURN_SHRINKAGE) * raw_mu + RETURN_SHRINKAGE * grand_mu
    cov_model = (
        (1.0 - COV_SHRINKAGE) * raw_cov
        + COV_SHRINKAGE * np.diag(np.diag(raw_cov))
    )

    rng = np.random.default_rng(SEED)
    W = feasible_cloud(len(assets), N_PORTFOLIOS, rng)
    p_ret = W @ mu_model
    p_var = np.einsum("ij,jk,ik->i", W, cov_model, W)
    p_risk = np.sqrt(np.maximum(p_var, 0.0))
    p_sharpe = np.divide(
        p_ret - RF_RATE,
        p_risk,
        out=np.zeros_like(p_ret),
        where=p_risk > 1e-12,
    )

    w_equal = np.ones(len(assets)) / len(assets)
    bounds = [(0.0, MAX_WEIGHT)] * len(assets)
    cons_sum = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def min_var(w):
        return float(w @ cov_model @ w)

    def neg_sharpe(w):
        return -metrics(w, mu_model, cov_model)[2]

    if SCIPY_OK:
        a = minimize(
            min_var,
            w_equal,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_sum,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        b = minimize(
            neg_sharpe,
            w_equal,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_sum,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w_min = a.x if a.success else W[np.argmin(p_risk)]
        w_max = b.x if b.success else W[np.argmax(p_sharpe)]
    else:
        w_min = W[np.argmin(p_risk)]
        w_max = W[np.argmax(p_sharpe)]

    min_ret = metrics(w_min, mu_model, cov_model)[0]
    max_ret = metrics(w_max, mu_model, cov_model)[0]
    target_balanced = (min_ret + max_ret) / 2.0

    if SCIPY_OK:
        cons_balanced = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, target=target_balanced: float(w @ mu_model - target),
            },
        ]
        c = minimize(
            min_var,
            (w_min + w_max) / 2.0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_balanced,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if c.success:
            w_balanced = c.x
        else:
            idx = np.argmin(np.abs(p_ret - target_balanced) + 0.25 * p_risk)
            w_balanced = W[idx]
    else:
        idx = np.argmin(np.abs(p_ret - target_balanced) + 0.25 * p_risk)
        w_balanced = W[idx]

    profile_rng = np.random.default_rng(SEED + 77)
    profiles = {
        "min_risk": profile_payload(
            "min_risk", w_min, assets, mu_model, cov_model, returns, profile_rng
        ),
        "balanced": profile_payload(
            "balanced", w_balanced, assets, mu_model, cov_model, returns, profile_rng
        ),
        "max_sharpe": profile_payload(
            "max_sharpe", w_max, assets, mu_model, cov_model, returns, profile_rng
        ),
        "equal": profile_payload(
            "equal", w_equal, assets, mu_model, cov_model, returns, profile_rng
        ),
    }

    asset_stats = {}
    cumulative = monthly.iloc[-1] / monthly.iloc[0] - 1.0
    raw_risk = np.sqrt(np.diag(raw_cov))
    for i, ticker in enumerate(assets):
        asset_stats[ticker] = {
            "name": NAMES[ticker],
            "annual_return_raw": round(float(raw_mu[i]), 6),
            "annual_return_model": round(float(mu_model[i]), 6),
            "annual_risk": round(float(raw_risk[i]), 6),
            "sample_return": round(float(cumulative[ticker]), 6),
            "last_research_price": round(float(monthly[ticker].iloc[-1]), 4),
        }

    order = np.argsort(p_risk)
    frontier_idx = []
    best_return = -np.inf
    for i in order:
        if p_ret[i] > best_return:
            frontier_idx.append(i)
            best_return = p_ret[i]

    display_idx = np.linspace(0, N_PORTFOLIOS - 1, 2200, dtype=int)
    normalized = []
    for ticker in assets:
        values = (monthly[ticker] / monthly[ticker].iloc[0]).round(6).tolist()
        normalized.append(
            {"ticker": ticker, "name": NAMES[ticker], "values": values}
        )

    state["weights"] = W
    state["portfolio_metrics"] = {
        "return": p_ret,
        "risk": p_risk,
        "sharpe": p_sharpe,
        "assets": assets,
    }
    state["research_df"] = monthly.reset_index()

    return {
        "period": {
            "start": monthly.index.min().strftime("%Y-%m"),
            "end": monthly.index.max().strftime("%Y-%m"),
            "months": int(len(monthly)),
            "return_observations": int(len(returns)),
            "assets": assets,
        },
        "portfolio_count": N_PORTFOLIOS,
        "seed": SEED,
        "max_weight": MAX_WEIGHT,
        "risk_free_rate": RF_RATE,
        "profiles": profiles,
        "portfolios": profiles,
        "asset_stats": asset_stats,
        "normalized": normalized,
        "monthly_returns": [
            {
                "date": d.strftime("%Y-%m"),
                **{ticker: round(float(row[ticker]), 6) for ticker in assets},
            }
            for d, row in returns.iterrows()
        ],
        "cloud": [
            {
                "risk": round(float(p_risk[i]), 6),
                "return": round(float(p_ret[i]), 6),
                "sharpe": round(float(p_sharpe[i]), 6),
            }
            for i in display_idx
        ],
        "frontier": [
            {
                "risk": round(float(p_risk[i]), 6),
                "return": round(float(p_ret[i]), 6),
            }
            for i in frontier_idx
        ],
        "method": {
            "returns": "monthly simple returns from official KASE month-end last-deal prices",
            "expected_return": "annualized monthly mean, shrunk 35% toward cross-sectional mean",
            "variance": "annualized covariance, 15% diagonal shrinkage",
            "risk_free_rate": RF_RATE,
            "constraints": f"long-only; sum(weights)=1; each weight <= {MAX_WEIGHT:.0%}",
            "optimizer": "SLSQP" if SCIPY_OK else "feasible random search",
            "bootstrap": f"{BOOTSTRAP_SIMS} resamples of 12 historical months",
            "limitations": [
                "Historical estimates do not guarantee future returns.",
                "The official free sample used here is short (19 month-end observations).",
                "Dividends, commissions, taxes, bid/ask spread and liquidity are not included.",
                "Current cards use the latest published public KASE price, not a licensed tick-by-tick feed.",
            ],
        },
    }


LIVE_TTL = 25
live_cache = {}
_NUM_TOKEN_RE = re.compile(r"^[+\-−]?\d[\d\s\xa0]*(?:[.,]\d+)?$")


def _parse_num(value):
    if value is None:
        return None
    x = (
        str(value)
        .replace("−", "-")
        .replace("\xa0", " ")
        .replace(" ", "")
        .replace(",", ".")
    )
    x = re.sub(r"[^0-9.\-+]", "", x)
    try:
        return float(x)
    except Exception:
        return None


def _value_before_label(strings, labels):
    labels = [re.sub(r"\s+", " ", x).strip().lower() for x in labels]
    for i, raw in enumerate(strings):
        current = re.sub(r"\s+", " ", raw).strip()
        low = current.lower()
        for label in labels:
            if label not in low:
                continue

            before = low.split(label, 1)[0].strip()
            if before:
                m = re.search(
                    r"([+\-−]?\d[\d\s\xa0]*(?:[.,]\d+)?)\s*$",
                    before,
                )
                if m:
                    val = _parse_num(m.group(1))
                    if val is not None:
                        return val

            for j in range(i - 1, max(-1, i - 5), -1):
                candidate = strings[j].strip()
                if _NUM_TOKEN_RE.fullmatch(candidate):
                    val = _parse_num(candidate)
                    if val is not None:
                        return val
    return None


async def fetch_kase_public(ticker: str):
    ticker = ticker.upper().strip()
    if ticker not in NAMES:
        raise HTTPException(status_code=404, detail="Тикер не поддерживается.")

    now = datetime.now(timezone.utc)
    cached = live_cache.get(ticker)
    if cached and (now - cached["fetched_at"]).total_seconds() < LIVE_TTL:
        return cached["data"]

    url = f"https://kase.kz/ru/investors/shares/{ticker}"
    try:
        async with httpx.AsyncClient(
            timeout=12,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 KASE-Vision/7.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        strings = list(soup.stripped_strings)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить публичные данные KASE: {exc}",
        )

    price = _value_before_label(
        strings,
        ["цена последней сделки", "price of the last deal", "last trade price"],
    )
    change = _value_before_label(strings, ["тренд, KZT", "trend, KZT"])
    change_pct = _value_before_label(strings, ["тренд, %", "trend, %"])

    if price is None or price <= 0:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось распознать последнюю цену KASE для {ticker}.",
        )

    data = {
        "ticker": ticker,
        "name": NAMES[ticker],
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "source": "KASE PUBLIC PAGE",
        "fetched_at": now.isoformat(),
        "url": url,
        "note": (
            "Последняя опубликованная цена сделки на публичной странице KASE; "
            "это не лицензированный биржевой real-time feed."
        ),
    }
    live_cache[ticker] = {"fetched_at": now, "data": data}
    return data


def load_research_data():
    if REAL_DATA.exists():
        return pd.read_csv(REAL_DATA, parse_dates=["date"])
    return pd.read_csv(DEMO, parse_dates=["date"])


def load_demo():
    return pd.read_csv(DEMO, parse_dates=["date"])


_research = load_research_data()
state["result"] = calculate(_research)
state["source"] = (
    "KASE • официальные месячные обзоры • цены последних сделок"
    if REAL_DATA.exists()
    else "DEMO • 24 месяца • seed=42"
)


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/api/live/{ticker}")
async def live(ticker: str):
    return await fetch_kase_public(ticker)


@app.get("/api/live")
async def live_many(tickers: str = "HSBK,KSPI,KZAP,KMGZ,KCEL,KEGC"):
    requested = [x.strip().upper() for x in tickers.split(",") if x.strip()]

    async def one(ticker):
        try:
            return await fetch_kase_public(ticker)
        except HTTPException as exc:
            return {"ticker": ticker, "error": exc.detail}

    items = await asyncio.gather(*(one(t) for t in requested))
    return {
        "items": items,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "refresh_seconds": LIVE_TTL,
        "mode": "KASE_PUBLIC_AUTO_REFRESH",
    }


@app.get("/api/analysis")
def analysis():
    return {"source": state["source"], "result": state["result"]}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "source": state["source"],
        "portfolios_generated": N_PORTFOLIOS,
        "version": "7.0",
    }


@app.get("/api/portfolios")
def portfolios(
    page: int = 1,
    limit: int = 25,
    sort: str = "sharpe",
    direction: str = "desc",
):
    if state["weights"] is None:
        raise HTTPException(status_code=503, detail="Портфели ещё не рассчитаны.")

    page = max(1, int(page))
    limit = min(100, max(1, int(limit)))
    metrics_data = state["portfolio_metrics"]
    key = sort if sort in ("return", "risk", "sharpe") else "sharpe"
    order = np.argsort(metrics_data[key])
    if direction != "asc":
        order = order[::-1]

    start = (page - 1) * limit
    idxs = order[start : start + limit]
    rows = []
    for idx in idxs:
        rows.append(
            {
                "id": int(idx) + 1,
                "return": float(metrics_data["return"][idx]),
                "risk": float(metrics_data["risk"][idx]),
                "sharpe": float(metrics_data["sharpe"][idx]),
                "weights": {
                    ticker: float(value)
                    for ticker, value in zip(
                        metrics_data["assets"],
                        state["weights"][idx],
                    )
                },
            }
        )

    return {
        "page": page,
        "limit": limit,
        "total": len(metrics_data["return"]),
        "assets": metrics_data["assets"],
        "rows": rows,
    }


@app.get("/api/recommendations")
async def recommendations(budget: float = 1_000_000, profile: str = "balanced"):
    if not np.isfinite(budget) or budget <= 0:
        raise HTTPException(status_code=400, detail="Бюджет должен быть больше нуля.")

    aliases = {
        "conservative": "min_risk",
        "min_risk": "min_risk",
        "balanced": "balanced",
        "growth": "max_sharpe",
        "max_sharpe": "max_sharpe",
    }
    profile_key = aliases.get(profile, profile)
    profiles = state["result"]["profiles"]
    if profile_key not in profiles:
        raise HTTPException(status_code=400, detail="Неизвестный профиль.")

    selected = profiles[profile_key]
    assets = state["result"]["period"]["assets"]
    last_research = {
        ticker: state["result"]["asset_stats"][ticker]["last_research_price"]
        for ticker in assets
    }

    async def price_for(ticker):
        try:
            live_quote = await fetch_kase_public(ticker)
            return ticker, float(live_quote["price"]), "KASE public latest"
        except Exception:
            return ticker, float(last_research[ticker]), "KASE month-end fallback"

    price_results = await asyncio.gather(*(price_for(t) for t in assets))
    prices = {ticker: (price, source) for ticker, price, source in price_results}

    rows = []
    invested = 0.0
    for ticker in assets:
        weight = float(selected["weights"][ticker])
        target_kzt = float(budget * weight)
        price, price_source = prices[ticker]
        shares = int(np.floor(target_kzt / price)) if price > 0 else 0
        actual_kzt = float(shares * price)
        invested += actual_kzt
        rows.append(
            {
                "ticker": ticker,
                "name": NAMES[ticker],
                "weight": round(weight, 6),
                "price": round(price, 4),
                "price_source": price_source,
                "target_kzt": round(target_kzt, 2),
                "shares": shares,
                "actual_kzt": round(actual_kzt, 2),
            }
        )

    return {
        "profile": profile_key,
        "label": selected["label"],
        "budget": round(float(budget), 2),
        "invested": round(invested, 2),
        "cash_residual": round(float(budget - invested), 2),
        "expected_return": selected["return"],
        "risk": selected["risk"],
        "sharpe": selected["sharpe"],
        "scenario_12m": selected["scenario_12m"],
        "rows": rows,
        "note": (
            "Исследовательская модель, а не персональная инвестиционная рекомендация. "
            "Количество акций округлено до целых; лоты, комиссии, налоги, спред и "
            "ликвидность не учтены."
        ),
    }


@app.post("/api/reset")
def reset():
    df = load_research_data()
    state["result"] = calculate(df)
    state["source"] = (
        "KASE • официальные месячные обзоры • цены последних сделок"
        if REAL_DATA.exists()
        else "DEMO • 24 месяца • seed=42"
    )
    return {"source": state["source"], "result": state["result"]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        name = (file.filename or "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            raise ValueError("Поддерживаются CSV и XLSX.")

        state["result"] = calculate(normalize(df))
        state["source"] = f"USER FILE • {file.filename}"
        return {"source": state["source"], "result": state["result"]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
