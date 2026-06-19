#!/usr/bin/env python3
"""
omicron_meta.py  —  Meta-Strategy Allocator with Citations
===========================================================
A meta-algorithm. It runs several independent trading strategies, scores
each one by how well it has ACTUALLY been performing lately (trailing
risk-adjusted return), and then blends their views — giving more weight to
the strategies that have earned it.

The headline feature: every recommendation is CITED. When the meta-algo
says "BUY GLD", it shows you exactly which strategies voted for it and how
strongly, so you can see the reasoning instead of trusting a black box.

    Child strategies (each votes on every asset):
        MOMENTUM     buy what's been trending up (12-1 month)
        MEAN-REV     buy what's stretched too far down (10-day z-score)
        TREND        buy what's above its 200-day moving average
        LOW-VOL      prefer calmer, lower-volatility names
        MACRO        rotate by market regime (risk-on vs risk-off)

    Meta layer:
        1. Replay each strategy over history -> daily P&L stream
        2. Score each strategy by trailing Sharpe (last ~3 months)
        3. Weight strategies by performance (softmax) -- winners get capital
        4. Blend current votes into one ranked recommendation
        5. CITE the top contributing strategies for each pick

    Survival layer (kept from omicron_quant):
        EjectorSeat scales total deployment down in systemic stress.

Data: 100% free (yfinance). No API key.

Usage:
    python omicron_meta.py            # live
    python omicron_meta.py --demo     # synthetic data, no network
    python omicron_meta.py --window 42 --top 10

NOTE: This is a learning / simulation tool for studying how strategies
combine. It is NOT financial advice and does not place trades. Paper-trade
ideas before ever risking real money.

Author: Xclaymation
"""

from __future__ import annotations

import argparse
import sys
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
    warnings.filterwarnings("ignore", message=".*not converging.*")
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
        # a few liquid single names for variety
        "AAPL": "Tech Stock", "MSFT": "Tech Stock", "NVDA": "Tech Stock",
        "JPM": "Financial", "XOM": "Energy", "JNJ": "Healthcare",
        "KO": "Staples", "CAT": "Industrial",
    })
    risk_assets: tuple = ("SPY", "QQQ", "IWM", "EEM", "HYG", "BTC-USD",
                          "ETH-USD", "NVDA", "AAPL")
    safe_assets: tuple = ("TLT", "IEF", "GLD", "UUP", "LQD", "JNJ", "KO")
    signal_only: list = field(default_factory=lambda: ["^VIX"])

    lookback: int = 504
    perf_window: int = 63      # trailing days used to score each strategy
    top: int = 10              # recommendations to surface
    softmax_temp: float = 1.2  # lower = winner-take-more, higher = more blended
    max_deploy: float = 0.85

    corr_panic: float = 0.80
    var_ceiling: float = 0.035
    vix_panic: float = 32.0


# ===========================================================================
# DATA
# ===========================================================================
class DataFeed:
    def __init__(self, cfg): self.cfg = cfg

    def fetch_live(self):
        if not HAVE_YF:
            raise RuntimeError("pip install yfinance — or use --demo")
        tk = list(self.cfg.universe) + self.cfg.signal_only
        raw = yf.download(tk, period=f"{self.cfg.lookback+50}d",
                          interval="1d", auto_adjust=True, progress=False)
        px = raw["Close"] if "Close" in raw else raw
        return px.dropna(how="all").ffill().tail(self.cfg.lookback)

    def fetch_demo(self, seed=7):
        rng = np.random.default_rng(seed)
        n = self.cfg.lookback
        tk = list(self.cfg.universe) + self.cfg.signal_only
        regime = np.zeros(n, dtype=int); state = 1
        for t in range(n):
            r = rng.random()
            if state == 1 and r < 0.012: state = 2
            elif state == 2 and r < 0.06: state = 1
            elif state == 1 and r < 0.03: state = 0
            elif state == 0 and r < 0.05: state = 1
            regime[t] = state
        regime[int(n*.55):int(n*.58)] = 2
        vol = {0:.006,1:.011,2:.038}; drift = {0:.0007,1:.0004,2:-.003}
        px = {}
        for t_ in tk:
            safe = t_ in self.cfg.safe_assets
            vix = t_ == "^VIX"
            beta = rng.uniform(0.7,1.5)
            s = [100*rng.uniform(.5,3)]
            for t in range(1,n):
                rg = regime[t]
                if vix:
                    s.append({0:13,1:17,2:36}[rg]+rng.normal(0,2)); continue
                v = vol[rg]*(0.6 if safe else 1.0)*beta
                mu = drift[rg]
                if safe and rg == 2: mu = 0.0015
                s.append(s[-1]*(1+rng.normal(mu,v)))
            px[t_] = s
        df = pd.DataFrame(px)
        df.index = pd.bdate_range(end=datetime.today(), periods=n)
        return df


# ===========================================================================
# REGIME (used by MACRO strategy + display)
# ===========================================================================
class Regime:
    LABELS = {0:"Calm",1:"Normal",2:"Stressed"}
    def __init__(self, cfg): self.cfg = cfg; self.sc = StandardScaler()
    def label_path(self, px):
        risk = [c for c in self.cfg.risk_assets if c in px]
        mkt = px[risk].pct_change().mean(axis=1)
        f = pd.DataFrame({"r": mkt, "v": mkt.rolling(20).std(),
                          "m": mkt.rolling(20).mean()})
        if "^VIX" in px: f["vix"] = px["^VIX"].pct_change()
        f = f.dropna()
        X = self.sc.fit_transform(f.values)
        if HAVE_HMM:
            m = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=200, random_state=42)
            m.fit(X); st = m.predict(X)
        else:
            m = GaussianMixture(n_components=3, covariance_type="full",
                                random_state=42, n_init=5)
            st = m.fit_predict(X)
        volby = {s: f["v"].values[st==s].mean() for s in np.unique(st)}
        order = sorted(volby, key=volby.get)
        remap = {raw:rank for rank,raw in enumerate(order)}
        return pd.Series([remap[s] for s in st], index=f.index)


# ===========================================================================
# CHILD STRATEGIES
# Each returns a score MATRIX (time x asset). Higher = more bullish.
# Scores are used two ways: (1) replayed historically to grade the strategy,
# (2) the latest row becomes the strategy's current "vote".
# ===========================================================================
class Strategy:
    name = "base"
    def score_matrix(self, px, ctx) -> pd.DataFrame:
        raise NotImplementedError

class Momentum(Strategy):
    name = "MOMENTUM"
    def score_matrix(self, px, ctx):
        # 12-1 month momentum: price[t-21] / price[t-252]
        return (px.shift(21) / px.shift(252) - 1)

class MeanReversion(Strategy):
    name = "MEAN-REV"
    def score_matrix(self, px, ctx):
        ma = px.rolling(10).mean(); sd = px.rolling(10).std()
        return -((px - ma) / (sd + 1e-9))   # buy what's stretched down

class Trend(Strategy):
    name = "TREND"
    def score_matrix(self, px, ctx):
        ma200 = px.rolling(200).mean()
        return (px / (ma200 + 1e-9) - 1)     # distance above 200dma

class LowVol(Strategy):
    name = "LOW-VOL"
    def score_matrix(self, px, ctx):
        vol = px.pct_change().rolling(20).std()
        return -vol                          # prefer low vol

class Macro(Strategy):
    name = "MACRO"
    def __init__(self, cfg): self.cfg = cfg
    def score_matrix(self, px, ctx):
        path = ctx["regime"].reindex(px.index).ffill().fillna(1)
        sc = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for a in px.columns:
            if a in self.cfg.risk_assets:
                # +1 in calm, 0 normal, -1 stressed
                sc[a] = path.map({0:1.0, 1:0.2, 2:-1.0})
            elif a in self.cfg.safe_assets:
                sc[a] = path.map({0:-0.5, 1:0.1, 2:1.0})
            else:
                sc[a] = path.map({0:0.5, 1:0.1, 2:-0.3})
        return sc


# ===========================================================================
# EJECTOR SEAT (survival overlay)
# ===========================================================================
class EjectorSeat:
    def __init__(self, cfg): self.cfg = cfg
    def assess(self, px):
        risk = [c for c in self.cfg.risk_assets if c in px]
        rets = px[risk].pct_change().dropna()
        cm = rets.tail(30).corr().values
        avgc = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
        eqw = rets.tail(252).mean(axis=1)
        var = float(-np.percentile(eqw, 5))
        vix = float(px["^VIX"].iloc[-1]) if "^VIX" in px else np.nan
        trig = []
        if avgc >= self.cfg.corr_panic: trig.append(f"corr {avgc:.2f}")
        if var >= self.cfg.var_ceiling: trig.append(f"VaR {var:.2%}")
        if not np.isnan(vix) and vix >= self.cfg.vix_panic: trig.append(f"VIX {vix:.0f}")
        status = "RED" if len(trig)>=2 else "YELLOW" if len(trig)==1 else "GREEN"
        return {"status":status,"triggers":trig,"avg_corr":avgc,"var":var,"vix":vix}


# ===========================================================================
# META ALLOCATOR  — the brain
# ===========================================================================
class MetaAllocator:
    def __init__(self, cfg, strategies):
        self.cfg = cfg
        self.strategies = strategies

    @staticmethod
    def _to_weights(score_row):
        """Cross-sectional: positive scores -> normalized long-only weights."""
        z = (score_row - score_row.mean()) / (score_row.std() + 1e-9)
        pos = z.clip(lower=0)
        return pos / pos.sum() if pos.sum() > 0 else pos

    def _strategy_pnl(self, SM, R):
        """Replay a strategy: each day hold weights from its scores, earn
        next-day return. Returns the strategy's daily P&L series."""
        # build weight matrix row by row (vectorized-ish)
        Z = SM.sub(SM.mean(axis=1), axis=0).div(SM.std(axis=1) + 1e-9, axis=0)
        W = Z.clip(lower=0)
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        pnl = (W.shift(1) * R).sum(axis=1)
        return pnl

    @staticmethod
    def _sharpe(pnl, window):
        x = pnl.tail(window)
        if x.std() == 0 or len(x) < 5:
            return 0.0
        return float(x.mean() / x.std() * np.sqrt(252))

    def run(self, px, ctx):
        tradables = [c for c in px.columns if c not in self.cfg.signal_only]
        P = px[tradables]
        R = P.pct_change()

        grades = {}        # strategy -> trailing sharpe
        cur_votes = {}     # strategy -> current z-scored vote per asset
        pnl_series = {}
        for strat in self.strategies:
            SM = strat.score_matrix(P, ctx).reindex(columns=tradables)
            pnl = self._strategy_pnl(SM, R)
            pnl_series[strat.name] = pnl
            grades[strat.name] = self._sharpe(pnl, self.cfg.perf_window)
            last = SM.iloc[-1]
            cur_votes[strat.name] = (last - last.mean()) / (last.std() + 1e-9)

        # meta weights via softmax over trailing Sharpe (winners get more)
        names = [s.name for s in self.strategies]
        sh = np.array([grades[n] for n in names])
        scaled = np.clip(sh, -2, 4) / self.cfg.softmax_temp
        ex = np.exp(scaled - scaled.max())
        meta_w = ex / ex.sum()
        meta_w = {n: float(w) for n, w in zip(names, meta_w)}

        # blended recommendation
        ensemble = pd.Series(0.0, index=tradables)
        contrib = {a: {} for a in tradables}      # asset -> {strat: contribution}
        for n in names:
            v = cur_votes[n].reindex(tradables).fillna(0.0)
            c = meta_w[n] * v
            ensemble += c
            for a in tradables:
                contrib[a][n] = float(c[a])

        return {"grades": grades, "meta_w": meta_w, "ensemble": ensemble,
                "contrib": contrib, "pnl": pnl_series}


# ===========================================================================
# ORCHESTRATION
# ===========================================================================
class OmicronMeta:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feed = DataFeed(cfg)
        self.regime = Regime(cfg)
        self.ejector = EjectorSeat(cfg)
        self.meta = MetaAllocator(cfg, [
            Momentum(), MeanReversion(), Trend(), LowVol(), Macro(cfg),
        ])

    def run(self, demo=False):
        px = self.feed.fetch_demo() if demo else self.feed.fetch_live()
        path = self.regime.label_path(px)
        reg_now = int(path.iloc[-1])
        ejc = self.ejector.assess(px)
        out = self.meta.run(px, {"regime": path})

        # convert ensemble -> deployable weights, scaled by ejector
        ens = out["ensemble"]
        pos = ens.clip(lower=0)
        w = pos / pos.sum() if pos.sum() > 0 else pos
        deploy = self.cfg.max_deploy
        if ejc["status"] == "YELLOW": deploy *= 0.5
        elif ejc["status"] == "RED": deploy = 0.0
        weights = (w * deploy)

        return {"px": px, "regime": reg_now, "regime_label": Regime.LABELS[reg_now],
                "ejector": ejc, "weights": weights, **out}


# ===========================================================================
# OUTPUT
# ===========================================================================
def _cite(contrib_row, meta_w, k=3):
    """Return top-k strategies that pushed an asset up, formatted."""
    items = [(n, c) for n, c in contrib_row.items() if c > 0.001]
    items.sort(key=lambda x: -x[1])
    return ", ".join(f"{n}({c:+.2f})" for n, c in items[:k]) or "—"


def render(cfg, res, demo):
    reg, ejc = res["regime_label"], res["ejector"]
    grades, meta_w, weights = res["grades"], res["meta_w"], res["weights"]
    contrib = res["contrib"]
    ejcol = {"GREEN":"green","YELLOW":"yellow","RED":"red"}[ejc["status"]]

    if not HAVE_RICH:
        return _plain(cfg, res, demo)

    _con.print()
    _con.print(Panel.fit(
        "[bold]OMICRON META[/bold]  ·  Multi-Strategy Allocator with Citations"
        + ("  [dim](DEMO)[/dim]" if demo else ""),
        box=box.DOUBLE, border_style="magenta"))
    eng = "HMM" if HAVE_HMM else "GaussianMixture"
    _con.print(f"  Regime: [bold]{reg}[/bold]   engine: {eng}")
    vixs = f"{ejc['vix']:.0f}" if not np.isnan(ejc['vix']) else "n/a"
    _con.print(f"  Stress overlay: corr {ejc['avg_corr']:.2f} · "
               f"VaR {ejc['var']:.2%} · VIX {vixs} "
               f"→ [{ejcol}]EJECTOR {ejc['status']}[/{ejcol}]")
    for t in ejc["triggers"]:
        _con.print(f"     ⚠  {t}", style=ejcol)
    _con.print()

    # Strategy scoreboard — who's winning and how much capital they get
    t0 = Table(box=box.SIMPLE_HEAVY, title="STRATEGY SCOREBOARD",
               title_style="bold")
    for c in ("Strategy","Trailing Sharpe","Meta Allocation"): t0.add_column(c)
    for n in sorted(grades, key=lambda k: -meta_w[k]):
        bar = "█" * int(round(meta_w[n]*30))
        col = "green" if grades[n] > 0.5 else "yellow" if grades[n] > 0 else "red"
        t0.add_row(n, f"[{col}]{grades[n]:+.2f}[/{col}]",
                   f"{meta_w[n]:5.1%}  {bar}")
    _con.print(t0)
    _con.print(f"  [dim]Higher trailing Sharpe → bigger vote. "
               f"temp={cfg.softmax_temp}[/dim]\n")

    if ejc["status"] == "RED":
        _con.print(Panel.fit(
            "[bold red]EJECTOR RED → recommend CASH[/bold red]\n"
            "Systemic stress. The allocator stands down regardless of votes.",
            border_style="red"))
        return

    # Recommendations WITH CITATIONS
    t1 = Table(box=box.SIMPLE_HEAVY, title="RECOMMENDATIONS (cited)",
               title_style="bold")
    for c in ("Asset","Class","Size","Conviction","Backed by (strategy · vote)"):
        t1.add_column(c)
    ranked = weights.sort_values(ascending=False)
    shown = 0
    for a, wv in ranked.items():
        if shown >= cfg.top: break
        if wv <= 0: continue
        ens = res["ensemble"][a]
        conv = "BUY" if wv >= 0.07 else "ACCUMULATE" if wv >= 0.03 else "WATCH"
        cv = {"BUY":"bold green","ACCUMULATE":"green","WATCH":"yellow"}[conv]
        t1.add_row(a, cfg.universe.get(a,""), f"{wv:.1%}",
                   f"[{cv}]{conv}[/{cv}]", _cite(contrib[a], meta_w))
        shown += 1
    _con.print(t1)
    _con.print(f"  Total deployed: [bold]{weights.sum():.0%}[/bold]   "
               f"Cash: [bold]{1-weights.sum():.0%}[/bold]")
    _con.print("\n  [dim]Each pick is backed by the strategies that voted for "
               "it — the ones winning lately count more.[/dim]")
    _con.print("  [dim]Learning/simulation tool. Not financial advice. "
               "Paper-trade first.[/dim]\n")


def _plain(cfg, res, demo):
    reg, ejc = res["regime_label"], res["ejector"]
    grades, meta_w, weights = res["grades"], res["meta_w"], res["weights"]
    contrib = res["contrib"]
    print("\n=== OMICRON META" + (" (DEMO)" if demo else "") + " ===")
    print(f"Regime: {reg}   EJECTOR: {ejc['status']}")
    for t in ejc["triggers"]: print("  ! " + t)
    print("\n-- STRATEGY SCOREBOARD --")
    print(f"{'STRATEGY':<12}{'SHARPE':<10}{'ALLOC':<8}")
    for n in sorted(grades, key=lambda k:-meta_w[k]):
        print(f"{n:<12}{grades[n]:+.2f}     {meta_w[n]:.1%}")
    if ejc["status"] == "RED":
        print("\n>>> EJECTOR RED -> CASH\n"); return
    print("\n-- RECOMMENDATIONS (cited) --")
    print(f"{'ASSET':<9}{'SIZE':<7}{'CONV':<12}{'BACKED BY'}")
    ranked = weights.sort_values(ascending=False); shown = 0
    for a, wv in ranked.items():
        if shown >= cfg.top or wv <= 0: break
        conv = "BUY" if wv>=0.07 else "ACCUMULATE" if wv>=0.03 else "WATCH"
        print(f"{a:<9}{wv*100:>4.1f}%  {conv:<12}{_cite(contrib[a], meta_w)}")
        shown += 1
    print(f"\nTotal deployed: {weights.sum():.0%}")
    print("Learning/simulation tool. Not financial advice.\n")


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Omicron Meta — multi-strategy allocator with citations")
    p.add_argument("--demo", action="store_true", help="synthetic data, no network")
    p.add_argument("--lookback", type=int, default=504)
    p.add_argument("--window", type=int, default=63, help="days to grade strategies")
    p.add_argument("--top", type=int, default=10, help="recommendations to show")
    p.add_argument("--temp", type=float, default=1.2, help="softmax temp (lower=winner-take-more)")
    args = p.parse_args(argv)

    cfg = Config(lookback=args.lookback, perf_window=args.window,
                 top=args.top, softmax_temp=args.temp)
    engine = OmicronMeta(cfg)
    try:
        res = engine.run(demo=args.demo)
    except RuntimeError as e:
        print(f"\n[error] {e}\n"); return 1
    render(cfg, res, args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())