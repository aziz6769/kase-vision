from __future__ import annotations
from pathlib import Path
import io
import os
import numpy as np
import pandas as pd
import httpx
import re
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

BASE = Path(__file__).resolve().parent
DEMO = BASE / "data" / "demo_prices.csv"
N_PORTFOLIOS = 60000
SEED = 42
NAMES = {"HSBK":"Halyk Bank","KSPI":"Kaspi.kz","KZAP":"Kazatomprom","KMGZ":"KazMunayGas","KCEL":"Kcell","KEGC":"KEGC"}
app = FastAPI(title="KASE Vision", version="6.0")

# CORS is configurable through the environment. Keep the local defaults for development;
# production hosts should set CORS_ORIGINS to their real frontend origin(s).
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
state = {"source":"DEMO • 24 месяца • seed=42","result":None,"weights":None,"portfolio_metrics":None}

def normalize(df):
    df=df.copy()
    if "date" not in df.columns:
        low={str(c).lower():c for c in df.columns}
        for a in ("datetime","timestamp","trade_date"):
            if a in low: df=df.rename(columns={low[a]:"date"}); break
    if "date" not in df.columns: raise ValueError("Нужна колонка date.")
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df=df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    assets=[t for t in NAMES if t in df.columns]
    if len(assets)<2: raise ValueError("Нужно минимум два столбца с тикерами: HSBK, KSPI, KZAP, KMGZ, KCEL, KEGC.")
    for t in assets: df[t]=pd.to_numeric(df[t],errors="coerce")
    return df[["date"]+assets].dropna(subset=assets,how="all")

def metrics(w,mu,cov,rf=0.0):
    ret=float(w@mu); risk=float(np.sqrt(max(w@cov@w,0.0))); sharpe=(ret-rf)/risk if risk>1e-12 else 0.0
    return ret,risk,sharpe

def calculate(df):
    monthly=df.set_index("date").resample("ME").last().dropna(how="any")
    if len(monthly)<6: raise ValueError("Нужно минимум 6 общих месячных наблюдений.")
    assets=list(monthly.columns)
    R=monthly.pct_change().dropna(); mean_m=R.mean(); cov_m=R.cov(); mu=mean_m.values*12.0; cov=cov_m.values*12.0
    rng=np.random.default_rng(SEED); W=rng.dirichlet(np.ones(len(assets)),size=N_PORTFOLIOS)
    p_ret=W@mu; p_var=np.einsum("ij,jk,ik->i",W,cov,W); p_risk=np.sqrt(np.maximum(p_var,0)); p_sharpe=np.divide(p_ret,p_risk,out=np.zeros_like(p_ret),where=p_risk>1e-12)
    w_eq=np.ones(len(assets))/len(assets); cons=[{"type":"eq","fun":lambda w:np.sum(w)-1}]; bounds=[(0,1)]*len(assets)
    def min_obj(w): return w@cov@w
    def neg_sh(w):
        ret,risk,_=metrics(w,mu,cov); return -ret/risk if risk>1e-12 else 1e6
    if SCIPY_OK:
        a=minimize(min_obj,w_eq,method="SLSQP",bounds=bounds,constraints=cons); b=minimize(neg_sh,w_eq,method="SLSQP",bounds=bounds,constraints=cons)
        w_min=a.x if a.success else W[np.argmin(p_risk)]; w_maxsh=b.x if b.success else W[np.argmax(p_sharpe)]
    else:
        w_min=W[np.argmin(p_risk)]; w_maxsh=W[np.argmax(p_sharpe)]
    portfolios={}
    for key,w in [("equal",w_eq),("min_risk",w_min),("max_sharpe",w_maxsh)]:
        ret,risk,sh=metrics(w,mu,cov); portfolios[key]={"weights":{t:round(float(x),6) for t,x in zip(assets,w)},"return":round(ret,6),"risk":round(risk,6),"sharpe":round(sh,6)}
    order=np.argsort(p_risk); frontier=[]; best=-np.inf
    for i in order:
        if p_ret[i]>best: frontier.append(i); best=p_ret[i]
    state["weights"]=W; state["portfolio_metrics"]={"return":p_ret,"risk":p_risk,"sharpe":p_sharpe,"assets":assets}
    display_idx=np.linspace(0,N_PORTFOLIOS-1,2400,dtype=int)
    norm=[]
    for t in assets:
        vals=(monthly[t]/monthly[t].iloc[0]).round(6).tolist(); norm.append({"ticker":t,"name":NAMES[t],"values":[1.0]+vals})
    return {"period":{"start":monthly.index.min().strftime("%Y-%m"),"end":monthly.index.max().strftime("%Y-%m"),"months":int(len(monthly)),"assets":assets},"portfolio_count":N_PORTFOLIOS,"seed":SEED,"normalized":norm,"monthly_returns":[{"date":d.strftime("%Y-%m"),**{t:round(float(row[t]),6) for t in assets}} for d,row in R.iterrows()],"mean_monthly":{t:round(float(mean_m[t]),6) for t in assets},"annual_mean":{t:round(float(mu[i]),6) for i,t in enumerate(assets)},"cumulative":{t:round(float(((1+R[t]).prod()-1)),6) for t in assets},"covariance":[[round(float(x),8) for x in row] for row in cov_m.values],"cloud":[{"risk":round(float(p_risk[i]),6),"return":round(float(p_ret[i]),6),"sharpe":round(float(p_sharpe[i]),6)} for i in display_idx],"frontier":[{"risk":round(float(p_risk[i]),6),"return":round(float(p_ret[i]),6)} for i in frontier],"portfolios":portfolios,"method":{"returns":"R_t = P_t / P_(t-1) - 1","expected_return":"E(Rp) = w^T μ","variance":"σ²p = w^T Σw","sharpe":"Sharpe = (E(Rp) - Rf) / σp","rf":0.0,"constraints":"wi ≥ 0; Σwi = 1","optimizer":"SLSQP" if SCIPY_OK else "feasible random search"}}

LIVE_TTL=25; live_cache={}
def _parse_num(value):
    if value is None:return None
    x=str(value).replace("\xa0"," ").replace(" ","").replace(",","."); x=re.sub(r"[^0-9.\-+]","",x)
    try:return float(x)
    except:return None

async def fetch_kase_public(ticker:str):
    ticker=ticker.upper().strip()
    if ticker not in NAMES: raise HTTPException(status_code=404,detail="Тикер не поддерживается.")
    now=datetime.now(timezone.utc); cached=live_cache.get(ticker)
    if cached and (now-cached["fetched_at"]).total_seconds()<LIVE_TTL:return cached["data"]
    url=f"https://kase.kz/ru/investors/shares/{ticker}"
    try:
        async with httpx.AsyncClient(timeout=10,follow_redirects=True,headers={"User-Agent":"KASE-Vision-Educational/6.0"}) as client:
            r=await client.get(url); r.raise_for_status(); text=re.sub(r"\s+"," ",r.text)
    except Exception as e: raise HTTPException(status_code=502,detail=f"Не удалось получить публичные данные KASE: {e}")
    patterns=[rf"{re.escape(ticker)}.*?([0-9][0-9\s\xa0]*[.,][0-9]+).*?цена последней сделки.*?([+-]?[0-9][0-9\s\xa0]*[.,][0-9]+).*?тренд, KZT.*?([+-]?[0-9][0-9\s\xa0]*[.,][0-9]+).*?тренд, %",rf"{re.escape(ticker)}.*?([0-9][0-9\s\xa0]*[.,][0-9]+).*?last trade price.*?([+-]?[0-9][0-9\s\xa0]*[.,][0-9]+).*?trend, KZT.*?([+-]?[0-9][0-9\s\xa0]*[.,][0-9]+).*?trend, %"]
    price=change=change_pct=None
    for pat in patterns:
        m=re.search(pat,text,flags=re.I)
        if m: price=_parse_num(m.group(1)); change=_parse_num(m.group(2)); change_pct=_parse_num(m.group(3)); break
    if price is None:
        m=re.search(rf"{re.escape(ticker)}.*?([0-9][0-9\s\xa0]*[.,][0-9]+)",text,flags=re.I)
        if m: price=_parse_num(m.group(1))
    if price is None: raise HTTPException(status_code=502,detail="Публичная страница KASE не вернула цену.")
    data={"ticker":ticker,"name":NAMES[ticker],"price":price,"change":change,"change_pct":change_pct,"source":"KASE PUBLIC PAGE","fetched_at":now.isoformat(),"url":url,"note":"Публичные данные KASE; это не лицензированный биржевой real-time feed."}
    live_cache[ticker]={"fetched_at":now,"data":data}; return data

def load_demo(): return pd.read_csv(DEMO,parse_dates=["date"])
state["result"]=calculate(load_demo())
@app.get("/")
def home(): return FileResponse(BASE/"index.html")
@app.get("/api/live/{ticker}")
async def live(ticker:str): return await fetch_kase_public(ticker)
@app.get("/api/live")
async def live_many(tickers:str="HSBK,KSPI,KZAP,KMGZ,KCEL,KEGC"):
    items=[]
    for ticker in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        try: items.append(await fetch_kase_public(ticker))
        except HTTPException as e: items.append({"ticker":ticker,"error":e.detail})
    return {"items":items,"fetched_at":datetime.now(timezone.utc).isoformat(),"refresh_seconds":LIVE_TTL,"mode":"KASE_PUBLIC_AUTO_REFRESH"}
@app.get("/api/analysis")
def analysis(): return {"source":state["source"],"result":state["result"]}
@app.get("/api/health")
def health(): return {"ok":True,"source":state["source"],"portfolios_generated":N_PORTFOLIOS}
@app.get("/api/portfolios")
def portfolios(page:int=1,limit:int=25,sort:str="sharpe",direction:str="desc"):
    if state["weights"] is None: raise HTTPException(status_code=503,detail="Портфели ещё не рассчитаны.")
    page=max(1,int(page)); limit=min(100,max(1,int(limit))); m=state["portfolio_metrics"]; key=sort if sort in ("return","risk","sharpe") else "sharpe"; order=np.argsort(m[key]); order=order if direction=="asc" else order[::-1]; start=(page-1)*limit; idxs=order[start:start+limit]
    rows=[]
    for idx in idxs: rows.append({"id":int(idx)+1,"return":float(m["return"][idx]),"risk":float(m["risk"][idx]),"sharpe":float(m["sharpe"][idx]),"weights":{t:float(x) for t,x in zip(m["assets"],state["weights"][idx])}})
    return {"page":page,"limit":limit,"total":len(m["return"]),"assets":m["assets"],"rows":rows}
@app.post("/api/reset")
def reset(): state["result"]=calculate(load_demo()); state["source"]="DEMO • 24 месяца • seed=42"; return {"source":state["source"],"result":state["result"]}
@app.post("/api/upload")
async def upload(file:UploadFile=File(...)):
    raw=await file.read()
    try:
        name=(file.filename or "").lower()
        if name.endswith(".csv"): df=pd.read_csv(io.BytesIO(raw))
        elif name.endswith((".xlsx",".xls")): df=pd.read_excel(io.BytesIO(raw))
        else: raise ValueError("Поддерживаются CSV и XLSX.")
        state["result"]=calculate(normalize(df)); state["source"]=f"USER FILE • {file.filename}"; return {"source":state["source"],"result":state["result"]}
    except Exception as e: raise HTTPException(status_code=400,detail=str(e))
