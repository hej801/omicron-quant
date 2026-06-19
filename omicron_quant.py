#!/usr/bin/env python3
"""
omicron_quant.py  —  Cross-Asset + Multi-Factor Stock Signal Engine
====================================================================
A survival-first quantitative ADVISORY engine. It does not place orders.
It tells you: what environment we're in, which asset classes have edge,
and — NEW in v2.0 — which individual STOCKS to buy, ranked by a
regime-aware multi-factor alpha model.

Layers (overrides flow downward):
    Layer 5  EjectorSeat        hard-coded cash triggers, bypasses everything
    Layer 4  CorrelationMonitor systemic stress / correlation-to-1 detection
    Layer 3  RegimeClassifier   what market state are we in? (HMM / GMM)
    Layer 2a SignalGenerator    regime-scoped ETF / asset-class signals
    Layer 2b StockFactorEngine  multi-factor alpha on individual names  <-- NEW
    Layer 1  SizingEngine       Kelly-capped conviction sizing

Stock factors (each z-scored cross-sectionally, then weighted by regime):
    momentum    12-1 month price momentum (skips last month)
    short_mom   1-month momentum
    rel_str     63-day return minus SPY (relative strength)
    low_vol     inverse annualized volatility (defensive)
    quality     ROE + profit margin - leverage      (fundamentals)
    value       inverse P/E + inverse P/B            (fundamentals)
    meanrev     inverse 10-day z-score (buy the dip)

Regime factor tilts:
    Calm      -> momentum / relative-strength heavy
    Normal    -> balanced
    Stressed  -> quality / low-vol / value heavy (defensive)

Data: 100% free (yfinance). Fundamentals fetched per-name with graceful
fallback to price-only factors if unavailable. Cached locally to speed reruns.

Usage:
    python omicron_quant.py                 # live: ETFs + stocks
    python omicron_quant.py --demo          # synthetic, no network
    python omicron_quant.py --no-fundamentals   # price-only factors (faster)
    python omicron_quant.py --stock-top 20      # surface 20 stock picks
    python omicron_quant.py --etf-only          # skip stocks (v1 behavior)

Author: Xclaymation  |  v2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMM = True
except ImportError:
    HAVE_HMM = False

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _con = Console()
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False
    _con = None

CACHE_FILE = ".omicron_fundamentals_cache.json"


# ===========================================================================
# CONFIG
# ===========================================================================
@dataclass
class Config:
    universe: dict = field(default_factory=lambda: {
        "SPY": "US Large Cap", "QQQ": "US Tech", "IWM": "US Small Cap",
        "EFA": "Intl Developed", "EEM": "Emerging Mkts",
        "GLD": "Gold", "SLV": "Silver", "DBC": "Commodities", "USO": "Oil",
        "TLT": "Long Treasuries", "IEF": "7-10y Treasuries",
        "LQD": "IG Credit", "HYG": "High Yield", "UUP": "US Dollar",
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    })

    stock_universe: dict = field(default_factory=lambda: {
        "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
        "META": "Tech", "AVGO": "Tech", "ORCL": "Tech", "CRM": "Tech",
        "AMD": "Tech", "ADBE": "Tech",
        "AMZN": "Cons Disc", "TSLA": "Cons Disc", "HD": "Cons Disc",
        "NKE": "Cons Disc", "MCD": "Cons Disc", "SBUX": "Cons Disc",
        "PG": "Staples", "KO": "Staples", "PEP": "Staples",
        "COST": "Staples", "WMT": "Staples", "CL": "Staples",
        "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
        "ABBV": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "V": "Financials",
        "MA": "Financials", "BRK-B": "Financials",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
        "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
        "HON": "Industrials", "UPS": "Industrials",
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        "DIS": "Comms", "NFLX": "Comms", "T": "Comms", "VZ": "Comms",
    })
    defensive_sectors: tuple = ("Staples", "Healthcare", "Utilities")

    signal_only: list = field(default_factory=lambda: ["^VIX"])

    lookback: int = 504
    regime_states: int = 3
    mom_fast: int = 20
    mom_slow: int = 100
    mr_window: int = 10
    vol_window: int = 20
    corr_window: int = 30
    var_window: int = 252
    var_conf: float = 0.95
    top: int = 6
    stock_top: int = 15

    corr_panic: float = 0.80
    var_ceiling: float = 0.035
    vix_panic: float = 32.0
    max_total_risk: float = 0.60


# ===========================================================================
# DATA LAYER
# ===========================================================================
class DataFeed:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch_live(self, include_stocks: bool) -> pd.DataFrame:
        if not HAVE_YF:
            raise RuntimeError(
                "yfinance not installed. Run: pip install yfinance\n"
                "Or use --demo for a no-network synthetic run.")
        tickers = list(self.cfg.universe) + self.cfg.signal_only
        if include_stocks:
            tickers += list(self.cfg.stock_universe)
        raw = yf.download(tickers, period=f"{self.cfg.lookback + 50}d",
                          interval="1d", auto_adjust=True, progress=False)
        px = raw["Close"] if "Close" in raw else raw
        px = px.dropna(how="all").ffill()
        return px.tail(self.cfg.lookback)

    def fetch_fundamentals(self, use_cache: bool = True) -> dict:
        if not HAVE_YF:
            return {}
        cache = {}
        if use_cache and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE) as f:
                    blob = json.load(f)
                if time.time() - blob.get("_ts", 0) < 86400:
                    cache = blob.get("data", {})
            except Exception:
                cache = {}
        out = dict(cache)
        missing = [t for t in self.cfg.stock_universe if t not in out]
        if missing:
            print(f"  fetching fundamentals for {len(missing)} names "
                  f"(cached for 24h)...")
        for i, tk in enumerate(missing):
            try:
                info = yf.Ticker(tk).info
                out[tk] = {
                    "roe": info.get("returnOnEquity"),
                    "margin": info.get("profitMargins"),
                    "d2e": info.get("debtToEquity"),
                    "pe": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                }
            except Exception:
                out[tk] = {}
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(missing)}...")
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"_ts": time.time(), "data": out}, f)
        except Exception:
            pass
        return out

    def fetch_demo(self, include_stocks: bool, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = self.cfg.lookback
        tickers = list(self.cfg.universe) + self.cfg.signal_only
        if include_stocks:
            tickers += list(self.cfg.stock_universe)

        regime = np.zeros(n, dtype=int)
        state = 1
        for t in range(n):
            roll = rng.random()
            if state == 1 and roll < 0.01: state = 2
            elif state == 2 and roll < 0.06: state = 1
            elif state == 1 and roll < 0.03: state = 0
            elif state == 0 and roll < 0.05: state = 1
            regime[t] = state
        regime[int(n*0.70):int(n*0.74)] = 2

        vol_map = {0: 0.006, 1: 0.011, 2: 0.040}
        drift = {0: 0.0006, 1: 0.0004, 2: -0.0035}
        px = {}
        for tk in tickers:
            safe = tk in ("TLT", "IEF", "GLD", "UUP", "LQD")
            defensive = tk in self.cfg.stock_universe and \
                self.cfg.stock_universe[tk] in self.cfg.defensive_sectors
            is_vix = tk == "^VIX"
            beta = rng.uniform(0.6, 1.6) if tk in self.cfg.stock_universe else 1.0
            series = [100.0 * rng.uniform(0.5, 3)]
            for t in range(1, n):
                r = regime[t]
                if is_vix:
                    series.append({0: 13, 1: 17, 2: 38}[r] + rng.normal(0, 2))
                    continue
                vmul = 0.6 if (safe or defensive) else 1.0
                vol = vol_map[r] * vmul * beta
                mu = drift[r]
                if (safe or defensive) and r == 2:
                    mu = 0.0015
                series.append(series[-1] * (1 + rng.normal(mu, vol)))
            px[tk] = series
        df = pd.DataFrame(px)
        df.index = pd.bdate_range(end=datetime.today(), periods=n)
        return df

    def demo_fundamentals(self, seed: int = 11) -> dict:
        rng = np.random.default_rng(seed)
        out = {}
        for tk, sec in self.cfg.stock_universe.items():
            defensive = sec in self.cfg.defensive_sectors
            out[tk] = {
                "roe": float(rng.uniform(0.05, 0.45)),
                "margin": float(rng.uniform(0.20, 0.40) if defensive
                                else rng.uniform(0.02, 0.30)),
                "d2e": float(rng.uniform(20, 200)),
                "pe": float(rng.uniform(10, 22) if defensive
                            else rng.uniform(15, 55)),
                "pb": float(rng.uniform(1.5, 12)),
            }
        return out


# ===========================================================================
# LAYER 3 — REGIME CLASSIFIER
# ===========================================================================
class RegimeClassifier:
    LABELS = {0: "Calm / Low-Vol", 1: "Normal", 2: "Stressed / High-Vol"}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def _features(self, px):
        risk = [c for c in ("SPY", "QQQ", "IWM", "EEM", "HYG") if c in px]
        mkt = px[risk].pct_change().mean(axis=1)
        f = pd.DataFrame(index=px.index)
        f["ret"] = mkt
        f["vol"] = mkt.rolling(self.cfg.vol_window).std()
        f["mom"] = mkt.rolling(self.cfg.mom_fast).mean()
        if "^VIX" in px:
            f["vix"] = px["^VIX"].pct_change()
        return f.dropna()

    def fit_predict(self, px):
        feats = self._features(px)
        X = self.scaler.fit_transform(feats.values)
        if HAVE_HMM:
            m = GaussianHMM(n_components=self.cfg.regime_states,
                            covariance_type="full", n_iter=200, random_state=42)
            m.fit(X); states = m.predict(X)
        else:
            m = GaussianMixture(n_components=self.cfg.regime_states,
                                covariance_type="full", random_state=42, n_init=5)
            states = m.fit_predict(X)
        vol_by = {s: feats["vol"].values[states == s].mean()
                  for s in np.unique(states)}
        order = sorted(vol_by, key=vol_by.get)
        remap = {raw: rank for rank, raw in enumerate(order)}
        self._feats = feats
        return pd.Series([remap[s] for s in states], index=feats.index)

    def summary(self, labeled):
        cur = int(labeled.iloc[-1])
        streak = 1
        for v in labeled.iloc[::-1][1:]:
            if int(v) == cur: streak += 1
            else: break
        conf = (labeled.tail(5) == cur).mean()
        return {"state": cur, "label": self.LABELS.get(cur, f"S{cur}"),
                "confidence": float(conf), "streak": streak,
                "transitioning": streak <= 2}


# ===========================================================================
# LAYER 4 — CORRELATION MONITOR
# ===========================================================================
class CorrelationMonitor:
    def __init__(self, cfg): self.cfg = cfg

    def assess(self, px):
        risk = [c for c in ("SPY", "QQQ", "IWM", "EEM", "HYG", "DBC", "BTC-USD")
                if c in px]
        rets = px[risk].pct_change().dropna()
        recent = rets.tail(self.cfg.corr_window).corr().values
        iu = np.triu_indices_from(recent, k=1)
        avg = float(np.nanmean(recent[iu]))
        base = rets.tail(self.cfg.var_window).corr().values
        basec = float(np.nanmedian(base[np.triu_indices_from(base, k=1)]))
        spike = avg - basec
        if avg >= self.cfg.corr_panic: lvl = "RED"
        elif avg >= self.cfg.corr_panic - 0.15 or spike > 0.20: lvl = "YELLOW"
        else: lvl = "GREEN"
        return {"avg_corr": avg, "baseline": basec, "spike": spike, "level": lvl}


# ===========================================================================
# LAYER 2a — ETF / ASSET-CLASS SIGNAL GENERATOR
# ===========================================================================
class SignalGenerator:
    def __init__(self, cfg): self.cfg = cfg

    def _scores(self, px):
        out = {}
        for tk in self.cfg.universe:
            if tk not in px: continue
            s = px[tk].dropna()
            if len(s) < self.cfg.mom_slow + 5: continue
            ret = s.pct_change()
            mf = s.iloc[-1]/s.iloc[-self.cfg.mom_fast]-1
            ms = s.iloc[-1]/s.iloc[-self.cfg.mom_slow]-1
            trend = float(np.sign(ms)*(0.6*mf+0.4*ms))
            z = (s.iloc[-1]-s.tail(self.cfg.mr_window).mean())/ \
                (s.tail(self.cfg.mr_window).std()+1e-9)
            out[tk] = {"trend": trend, "meanrev": float(-z),
                       "ann_vol": float(ret.tail(self.cfg.vol_window).std()*np.sqrt(252))}
        return pd.DataFrame(out).T

    def generate(self, px, regime):
        sc = self._scores(px)
        if sc.empty: return sc
        st = regime["state"]
        wt, wm = {0:(0.80,0.20),1:(0.55,0.45),2:(0.25,0.75)}[st]
        def zc(c):
            v = sc[c]; return (v-v.mean())/(v.std()+1e-9)
        sc["raw"] = wt*zc("trend")+wm*zc("meanrev")
        if st == 2:
            for tk,b in {"TLT":0.5,"IEF":0.4,"GLD":0.5,"UUP":0.4,"LQD":0.2}.items():
                if tk in sc.index: sc.loc[tk,"raw"] += b
        return sc.sort_values("raw", ascending=False)


# ===========================================================================
# LAYER 2b — MULTI-FACTOR STOCK ALPHA ENGINE
# ===========================================================================
class StockFactorEngine:
    WEIGHTS = {
        0: {"momentum": 0.30, "short_mom": 0.10, "rel_str": 0.20,
            "low_vol": 0.05, "meanrev": 0.05, "quality": 0.15, "value": 0.15},
        1: {"momentum": 0.22, "short_mom": 0.08, "rel_str": 0.18,
            "low_vol": 0.10, "meanrev": 0.10, "quality": 0.18, "value": 0.14},
        2: {"momentum": 0.05, "short_mom": 0.03, "rel_str": 0.07,
            "low_vol": 0.25, "meanrev": 0.15, "quality": 0.28, "value": 0.17},
    }

    def __init__(self, cfg): self.cfg = cfg

    @staticmethod
    def _winsor_z(s: pd.Series) -> pd.Series:
        z = (s - s.mean()) / (s.std() + 1e-9)
        return z.clip(-3, 3)

    def _price_factors(self, px):
        rows = {}
        spy = px["SPY"].dropna() if "SPY" in px else None
        spy_63 = (spy.iloc[-1]/spy.iloc[-63]-1) if spy is not None and len(spy) > 63 else 0.0
        for tk in self.cfg.stock_universe:
            if tk not in px: continue
            s = px[tk].dropna()
            if len(s) < 260: continue
            ret = s.pct_change()
            mom = s.iloc[-21]/s.iloc[-252]-1
            short = s.iloc[-1]/s.iloc[-21]-1
            rel = (s.iloc[-1]/s.iloc[-63]-1) - spy_63
            vol = float(ret.tail(self.cfg.vol_window).std()*np.sqrt(252))
            z = (s.iloc[-1]-s.tail(self.cfg.mr_window).mean())/ \
                (s.tail(self.cfg.mr_window).std()+1e-9)
            rows[tk] = {"momentum": float(mom), "short_mom": float(short),
                        "rel_str": float(rel), "low_vol": float(-vol),
                        "meanrev": float(-z), "ann_vol": vol,
                        "sector": self.cfg.stock_universe[tk]}
        return pd.DataFrame(rows).T

    def _fundamental_factors(self, df, fundamentals):
        roe, margin, d2e, pe, pb = ({} for _ in range(5))
        for tk in df.index:
            f = fundamentals.get(tk, {}) or {}
            roe[tk] = f.get("roe"); margin[tk] = f.get("margin")
            d2e[tk] = f.get("d2e"); pe[tk] = f.get("pe"); pb[tk] = f.get("pb")
        def col(d, invert=False, posonly=False):
            s = pd.Series(d, dtype="float64")
            if posonly: s = s.where(s > 0)
            s = s.fillna(s.median())
            if invert: s = -s
            return s
        q = (self._winsor_z(col(roe)) + self._winsor_z(col(margin))
             + self._winsor_z(col(d2e, invert=True))) / 3
        v = (self._winsor_z(col(pe, invert=True, posonly=True))
             + self._winsor_z(col(pb, invert=True, posonly=True))) / 2
        df["quality"] = q.reindex(df.index).fillna(0.0)
        df["value"] = v.reindex(df.index).fillna(0.0)
        return df

    def generate(self, px, regime, fundamentals=None, use_fund=True):
        df = self._price_factors(px)
        if df.empty: return df
        for c in ("momentum", "short_mom", "rel_str", "low_vol", "meanrev"):
            df[c + "_z"] = self._winsor_z(df[c].astype(float))
        if use_fund and fundamentals:
            df = self._fundamental_factors(df, fundamentals)
        else:
            df["quality"] = 0.0; df["value"] = 0.0
        df["quality_z"] = df["quality"]
        df["value_z"] = df["value"]

        w = self.WEIGHTS[regime["state"]]
        df["alpha"] = (
            w["momentum"]*df["momentum_z"] + w["short_mom"]*df["short_mom_z"]
            + w["rel_str"]*df["rel_str_z"] + w["low_vol"]*df["low_vol_z"]
            + w["meanrev"]*df["meanrev_z"] + w["quality"]*df["quality_z"]
            + w["value"]*df["value_z"]
        )
        if regime["state"] == 2:
            mask = df["sector"].isin(self.cfg.defensive_sectors)
            df.loc[mask, "alpha"] += 0.25
        return df.sort_values("alpha", ascending=False)

    def size(self, scored, regime, ejector):
        if scored.empty: return scored
        df = scored.copy()
        df["conv"] = df["alpha"].clip(lower=0)
        if df["conv"].sum() == 0:
            df["weight"] = 0.0; return df
        inv_vol = 1.0/(df["ann_vol"]+1e-6)
        raw = (df["conv"]*inv_vol)
        top = df.nlargest(self.cfg.stock_top, "alpha").index
        raw = raw.where(df.index.isin(top), 0.0)
        weights = raw/raw.sum() if raw.sum() > 0 else raw
        deploy = {0:0.60,1:0.45,2:0.25}[regime["state"]]
        if regime["transitioning"]: deploy *= 0.5
        if ejector["status"] == "YELLOW": deploy *= 0.5
        elif ejector["status"] == "RED": deploy = 0.0
        deploy = min(deploy, self.cfg.max_total_risk)
        df["weight"] = (weights*deploy).round(4)
        return df.sort_values("weight", ascending=False)


# ===========================================================================
# LAYER 1 — SIZING ENGINE (ETFs)
# ===========================================================================
class SizingEngine:
    def __init__(self, cfg): self.cfg = cfg
    def size(self, scored, regime, ejector):
        if scored.empty: return scored
        df = scored.copy()
        df["conv"] = df["raw"].clip(lower=0)
        if df["conv"].sum() == 0:
            df["weight"] = 0.0; return df
        inv = 1.0/(df["ann_vol"]+1e-6)
        w = (df["conv"]*inv); w = w/w.sum()
        deploy = {0:0.60,1:0.45,2:0.25}[regime["state"]]
        if regime["transitioning"]: deploy *= 0.5
        if ejector["status"] == "YELLOW": deploy *= 0.5
        elif ejector["status"] == "RED": deploy = 0.0
        deploy = min(deploy, self.cfg.max_total_risk)
        df["weight"] = (w*deploy).round(4)
        return df.sort_values("weight", ascending=False)


# ===========================================================================
# LAYER 5 — EJECTOR SEAT
# ===========================================================================
class EjectorSeat:
    def __init__(self, cfg): self.cfg = cfg
    def _var(self, px):
        risk = [c for c in ("SPY","QQQ","IWM","EEM","HYG") if c in px]
        r = px[risk].pct_change().dropna().tail(self.cfg.var_window).mean(axis=1)
        return float(-np.percentile(r, (1-self.cfg.var_conf)*100))
    def assess(self, px, corr):
        trig = []; var = self._var(px)
        vix = float(px["^VIX"].iloc[-1]) if "^VIX" in px else np.nan
        if corr["level"] == "RED":
            trig.append(f"Cross-asset corr {corr['avg_corr']:.2f} >= {self.cfg.corr_panic:.2f}")
        if var >= self.cfg.var_ceiling:
            trig.append(f"VaR {var:.2%} >= ceiling {self.cfg.var_ceiling:.2%}")
        if not np.isnan(vix) and vix >= self.cfg.vix_panic:
            trig.append(f"VIX {vix:.1f} >= panic {self.cfg.vix_panic:.0f}")
        status = "RED" if len(trig) >= 2 else "YELLOW" if len(trig) == 1 else "GREEN"
        return {"status": status, "triggers": trig, "var": var, "vix": vix}


# ===========================================================================
# ORCHESTRATION
# ===========================================================================
class OmicronQuant:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feed = DataFeed(cfg)
        self.regime = RegimeClassifier(cfg)
        self.corr = CorrelationMonitor(cfg)
        self.signals = SignalGenerator(cfg)
        self.stocks = StockFactorEngine(cfg)
        self.sizer = SizingEngine(cfg)
        self.ejector = EjectorSeat(cfg)

    def run(self, demo=False, etf_only=False, use_fund=True):
        if demo:
            px = self.feed.fetch_demo(include_stocks=not etf_only)
            fund = {} if etf_only else self.feed.demo_fundamentals()
        else:
            px = self.feed.fetch_live(include_stocks=not etf_only)
            fund = {} if (etf_only or not use_fund) else self.feed.fetch_fundamentals()

        labeled = self.regime.fit_predict(px)
        reg = self.regime.summary(labeled)
        corr = self.corr.assess(px)
        ejc = self.ejector.assess(px, corr)

        etf_sized = self.sizer.size(self.signals.generate(px, reg), reg, ejc)
        stk_sized = None
        if not etf_only:
            stk_scored = self.stocks.generate(px, reg, fund, use_fund)
            stk_sized = self.stocks.size(stk_scored, reg, ejc)

        return {"regime": reg, "corr": corr, "ejector": ejc,
                "etf": etf_sized, "stocks": stk_sized}


# ===========================================================================
# OUTPUT
# ===========================================================================
def _verdict(w, raw, ejector_status):
    if ejector_status == "RED": return "CASH"
    if w >= 0.06: return "BUY"
    if w >= 0.025: return "ACCUMULATE"
    if raw > 0: return "WATCH"
    return "AVOID"

_VC = {"BUY":"bold green","ACCUMULATE":"green","WATCH":"yellow",
       "AVOID":"dim","CASH":"red"}


def render(cfg, res, demo, etf_only, use_fund):
    reg, corr, ejc = res["regime"], res["corr"], res["ejector"]
    etf, stk = res["etf"], res["stocks"]
    ejcol = {"GREEN":"green","YELLOW":"yellow","RED":"red"}[ejc["status"]]
    crcol = {"GREEN":"green","YELLOW":"yellow","RED":"red"}[corr["level"]]

    if not HAVE_RICH:
        _render_plain(cfg, res, demo, etf_only); return

    _con.print()
    _con.print(Panel.fit(
        "[bold]OMICRON QUANT[/bold]  ·  Cross-Asset + Multi-Factor Stocks  v2.0"
        + ("  [dim](DEMO)[/dim]" if demo else ""),
        box=box.DOUBLE, border_style="cyan"))
    eng = "HMM" if HAVE_HMM else "GaussianMixture"
    trans = " [yellow](TRANSITIONING — sizes halved)[/yellow]" if reg["transitioning"] else ""
    _con.print(f"  Regime engine : [cyan]{eng}[/cyan]   "
               f"Factors: {'price+fundamentals' if (use_fund and not etf_only) else 'price-only'}")
    _con.print(f"  Current regime: [bold]{reg['label']}[/bold]  "
               f"conf {reg['confidence']:.0%}  ({reg['streak']}d){trans}")
    _con.print(f"  Cross-asset ρ : {corr['avg_corr']:.2f} "
               f"(base {corr['baseline']:.2f}, spike {corr['spike']:+.2f}) "
               f"→ [{crcol}]{corr['level']}[/{crcol}]")
    vixs = f"{ejc['vix']:.1f}" if not np.isnan(ejc['vix']) else "n/a"
    _con.print(f"  Portfolio VaR : {ejc['var']:.2%} (95%)   VIX: {vixs}")
    _con.print(f"  [bold {ejcol}]EJECTOR SEAT: {ejc['status']}[/bold {ejcol}]")
    for t in ejc["triggers"]:
        _con.print(f"     ⚠  {t}", style=ejcol)
    _con.print()

    if ejc["status"] == "RED":
        _con.print(Panel.fit(
            "[bold red]ALL POSITIONS → CASH[/bold red]\n"
            "Multiple systemic stress triggers fired. Ejector seat overrides "
            "all signals. Park capital.", border_style="red"))
        return

    t1 = Table(box=box.SIMPLE_HEAVY, title="ASSET-CLASS SIGNALS (ETF)",
               title_style="bold")
    for c in ("Asset","Class","Action","Size","Trend","Vol"): t1.add_column(c)
    for tk,row in etf.head(cfg.top + 3).iterrows():
        v = _verdict(row["weight"], row["raw"], ejc["status"])
        t1.add_row(tk, cfg.universe.get(tk,""), f"[{_VC[v]}]{v}[/{_VC[v]}]",
                   f"{row['weight']:.1%}" if row["weight"]>0 else "—",
                   f"{row['trend']:+.2%}", f"{row['ann_vol']:.0%}")
    _con.print(t1)
    _con.print(f"  ETF deployed: [bold]{etf['weight'].sum():.0%}[/bold]\n")

    if stk is not None and not stk.empty:
        t2 = Table(box=box.SIMPLE_HEAVY, title="STOCK PICKS (multi-factor alpha)",
                   title_style="bold")
        for c in ("Ticker","Sector","Action","Size","Alpha","Mom","Qual","Val","Vol"):
            t2.add_column(c)
        shown = 0
        for tk,row in stk.iterrows():
            if shown >= cfg.stock_top and row["weight"] == 0: continue
            v = _verdict(row["weight"], row["alpha"], ejc["status"])
            t2.add_row(
                tk, str(row["sector"]), f"[{_VC[v]}]{v}[/{_VC[v]}]",
                f"{row['weight']:.1%}" if row["weight"]>0 else "—",
                f"{row['alpha']:+.2f}", f"{row['momentum_z']:+.1f}",
                f"{row['quality_z']:+.1f}", f"{row['value_z']:+.1f}",
                f"{row['ann_vol']:.0%}")
            shown += 1
        _con.print(t2)
        tot = stk["weight"].sum()
        _con.print(f"  Stock deployed: [bold]{tot:.0%}[/bold]   "
                   f"Cash on stock sleeve: [bold]{1-tot:.0%}[/bold]")
    _con.print("\n  [dim]Advisory only. Not order execution. Verify before acting.[/dim]\n")


def _render_plain(cfg, res, demo, etf_only):
    reg, corr, ejc = res["regime"], res["corr"], res["ejector"]
    etf, stk = res["etf"], res["stocks"]
    print("\n=== OMICRON QUANT v2.0" + (" (DEMO)" if demo else "") + " ===")
    print(f"Regime: {reg['label']} ({reg['confidence']:.0%}, {reg['streak']}d)"
          + ("  TRANSITIONING" if reg['transitioning'] else ""))
    print(f"Corr: {corr['avg_corr']:.2f} [{corr['level']}]   "
          f"VaR: {ejc['var']:.2%}   EJECTOR: {ejc['status']}")
    for t in ejc["triggers"]: print("  ! " + t)
    if ejc["status"] == "RED":
        print("\n>>> ALL POSITIONS -> CASH.\n"); return
    print("\n-- ETF --")
    print(f"{'ASSET':<9}{'ACTION':<12}{'SIZE':<7}{'TREND':<9}")
    for tk,row in etf.head(cfg.top+3).iterrows():
        v = _verdict(row["weight"], row["raw"], ejc["status"])
        print(f"{tk:<9}{v:<12}{(str(round(row['weight']*100,1))+'%') if row['weight']>0 else '-':<7}{row['trend']:+.2%}")
    if stk is not None and not stk.empty:
        print("\n-- STOCKS (multi-factor) --")
        print(f"{'TICKER':<8}{'SECTOR':<13}{'ACTION':<12}{'SIZE':<7}{'ALPHA':<8}{'VOL':<6}")
        shown = 0
        for tk,row in stk.iterrows():
            if shown >= cfg.stock_top and row["weight"] == 0: continue
            v = _verdict(row["weight"], row["alpha"], ejc["status"])
            print(f"{tk:<8}{str(row['sector']):<13}{v:<12}"
                  f"{(str(round(row['weight']*100,1))+'%') if row['weight']>0 else '-':<7}"
                  f"{row['alpha']:+.2f}   {row['ann_vol']:.0%}")
            shown += 1
        print(f"\nStock deployed: {stk['weight'].sum():.0%}")
    print()


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Omicron Quant v2 — cross-asset + stock factor engine")
    p.add_argument("--demo", action="store_true", help="synthetic data, no network")
    p.add_argument("--etf-only", action="store_true", help="skip individual stocks")
    p.add_argument("--no-fundamentals", action="store_true",
                   help="price-only factors (much faster, no .info calls)")
    p.add_argument("--lookback", type=int, default=504)
    p.add_argument("--top", type=int, default=6, help="ETF names to surface")
    p.add_argument("--stock-top", type=int, default=15, help="stock picks to surface")
    p.add_argument("--states", type=int, default=3)
    args = p.parse_args(argv)

    cfg = Config(lookback=args.lookback, top=args.top,
                 stock_top=args.stock_top, regime_states=args.states)
    use_fund = not args.no_fundamentals
    engine = OmicronQuant(cfg)
    try:
        res = engine.run(demo=args.demo, etf_only=args.etf_only, use_fund=use_fund)
    except RuntimeError as e:
        print(f"\n[error] {e}\n"); return 1
    render(cfg, res, args.demo, args.etf_only, use_fund)
    return 0


if __name__ == "__main__":
    sys.exit(main())