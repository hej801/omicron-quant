#!/usr/bin/env python3
"""
omicron_quant.py  —  Cross-Asset Signal Engine
================================================
A survival-first quantitative ADVISORY engine. It does not place orders.
It tells you: what environment we're in, what has edge in that environment,
how much conviction to size with, and when to step aside to cash.

Five layers, stacked by priority (overrides flow downward):

    Layer 5  EjectorSeat        hard-coded cash triggers, bypasses everything
    Layer 4  CorrelationMonitor systemic stress / correlation-to-1 detection
    Layer 3  RegimeClassifier   what market state are we in? (HMM / GMM)
    Layer 2  SignalGenerator    regime-scoped momentum + mean-reversion
    Layer 1  SizingEngine       Kelly-capped conviction sizing

Data: 100% free. yfinance only (no API key, no signup). VIX and a credit
proxy (HYG/IEF ratio) are pulled as regime inputs, not traded.

Usage:
    python omicron_quant.py            # live run (needs internet)
    python omicron_quant.py --demo     # synthetic data, no network needed
    python omicron_quant.py --lookback 504 --top 6

Author: Xclaymation  |  v1.0
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

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully if missing
# ---------------------------------------------------------------------------
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

from sklearn.mixture import GaussianMixture  # always available, HMM fallback
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
    # Tradable cross-asset universe (what we may recommend BUYing)
    universe: dict = field(default_factory=lambda: {
        "SPY":  "US Large Cap",
        "QQQ":  "US Tech",
        "IWM":  "US Small Cap",
        "EFA":  "Intl Developed",
        "EEM":  "Emerging Mkts",
        "GLD":  "Gold",
        "SLV":  "Silver",
        "DBC":  "Commodities",
        "USO":  "Oil",
        "TLT":  "Long Treasuries",
        "IEF":  "7-10y Treasuries",
        "LQD":  "IG Credit",
        "HYG":  "High Yield",
        "UUP":  "US Dollar",
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    })
    # Regime / stress inputs — NOT traded, used as signal only
    signal_only: list = field(default_factory=lambda: ["^VIX"])

    lookback: int = 504          # trading days of history (~2y)
    regime_states: int = 3       # quiet / normal / stressed
    mom_fast: int = 20           # fast momentum window
    mom_slow: int = 100          # slow momentum / trend window
    mr_window: int = 10          # mean-reversion z-score window
    vol_window: int = 20         # realized vol window
    corr_window: int = 30        # rolling correlation window
    var_window: int = 252        # VaR estimation window
    var_conf: float = 0.95       # 95% VaR
    top: int = 6                 # max names to surface

    # --- Ejector seat hard limits (non-negotiable) ---
    corr_panic: float = 0.80     # avg cross-asset corr above this = systemic stress
    var_ceiling: float = 0.035   # 3.5% daily 95% VaR on equal-wt risk basket = red
    vix_panic: float = 32.0      # absolute VIX level for hard caution
    max_total_risk: float = 0.60 # never recommend >60% total deployed


# ===========================================================================
# DATA LAYER
# ===========================================================================
class DataFeed:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch_live(self) -> pd.DataFrame:
        if not HAVE_YF:
            raise RuntimeError(
                "yfinance not installed. Run: pip install yfinance\n"
                "Or use --demo for a no-network synthetic run."
            )
        tickers = list(self.cfg.universe.keys()) + self.cfg.signal_only
        raw = yf.download(
            tickers, period=f"{self.cfg.lookback + 50}d",
            interval="1d", auto_adjust=True, progress=False,
        )
        # handle single vs multi-index columns
        px = raw["Close"] if "Close" in raw else raw
        px = px.dropna(how="all").ffill().dropna()
        return px.tail(self.cfg.lookback)

    def fetch_demo(self, seed: int = 7) -> pd.DataFrame:
        """Synthetic but realistic: regime-switching GBM with a crash baked in.
        Lets the full pipeline run with zero network access."""
        rng = np.random.default_rng(seed)
        n = self.cfg.lookback
        tickers = list(self.cfg.universe.keys()) + self.cfg.signal_only

        # build a hidden regime path: 0 calm, 1 normal, 2 crash
        regime = np.zeros(n, dtype=int)
        state = 1
        for t in range(n):
            roll = rng.random()
            if state == 1 and roll < 0.01:
                state = 2
            elif state == 2 and roll < 0.06:
                state = 1
            elif state == 1 and roll < 0.03:
                state = 0
            elif state == 0 and roll < 0.05:
                state = 1
            regime[t] = state
        # force a sharp crash window so the ejector seat has something to catch
        regime[int(n * 0.70): int(n * 0.74)] = 2

        vol_map = {0: 0.006, 1: 0.011, 2: 0.040}
        drift_risk = {0: 0.0006, 1: 0.0004, 2: -0.0035}

        px = {}
        for tk in tickers:
            is_safe = tk in ("TLT", "IEF", "GLD", "UUP", "LQD")
            is_vix = tk == "^VIX"
            series = [100.0]
            for t in range(1, n):
                r = regime[t]
                if is_vix:
                    base = {0: 13, 1: 17, 2: 38}[r]
                    series.append(base + rng.normal(0, 2))
                    continue
                vol = vol_map[r] * (0.6 if is_safe else 1.0)
                mu = drift_risk[r]
                if is_safe and r == 2:        # flight to safety
                    mu = 0.0025
                shock = rng.normal(mu, vol)
                series.append(series[-1] * (1 + shock))
            px[tk] = series
        df = pd.DataFrame(px)
        df.index = pd.bdate_range(end=datetime.today(), periods=n)
        df.attrs["true_regime"] = regime
        return df


# ===========================================================================
# LAYER 3 — REGIME CLASSIFIER  (the Weather Radar)
# ===========================================================================
class RegimeClassifier:
    """HMM over market-wide features. Falls back to GaussianMixture if
    hmmlearn isn't installed. Output is a labeled regime + how confident
    and how *fresh* (just-shifted) the regime is."""

    LABELS = {0: "Calm / Low-Vol", 1: "Normal", 2: "Stressed / High-Vol"}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.scaler = StandardScaler()
        self._state_order = None

    def _features(self, px: pd.DataFrame) -> pd.DataFrame:
        # market-wide proxy = mean of the risk assets
        risk = [c for c in ("SPY", "QQQ", "IWM", "EEM", "HYG") if c in px]
        mkt = px[risk].pct_change().mean(axis=1)
        feats = pd.DataFrame(index=px.index)
        feats["ret"] = mkt
        feats["vol"] = mkt.rolling(self.cfg.vol_window).std()
        feats["mom"] = mkt.rolling(self.cfg.mom_fast).mean()
        if "^VIX" in px:
            feats["vix"] = px["^VIX"].pct_change()
        return feats.dropna()

    def fit_predict(self, px: pd.DataFrame) -> pd.Series:
        feats = self._features(px)
        X = self.scaler.fit_transform(feats.values)

        if HAVE_HMM:
            self.model = GaussianHMM(
                n_components=self.cfg.regime_states,
                covariance_type="full", n_iter=200, random_state=42,
            )
            self.model.fit(X)
            states = self.model.predict(X)
        else:
            self.model = GaussianMixture(
                n_components=self.cfg.regime_states,
                covariance_type="full", random_state=42, n_init=5,
            )
            states = self.model.fit_predict(X)

        # relabel raw states so 0=calm,1=normal,2=stressed by realized vol
        vol_by_state = {s: feats["vol"].values[states == s].mean()
                        for s in np.unique(states)}
        order = sorted(vol_by_state, key=vol_by_state.get)
        remap = {raw: rank for rank, raw in enumerate(order)}
        labeled = pd.Series([remap[s] for s in states], index=feats.index)
        self._last_feats = feats
        return labeled

    def summary(self, labeled: pd.Series) -> dict:
        current = int(labeled.iloc[-1])
        # how many consecutive days in this regime (freshness)
        streak = 1
        for v in labeled.iloc[::-1][1:]:
            if int(v) == current:
                streak += 1
            else:
                break
        # confidence = stability of last 5 days
        recent = labeled.tail(5)
        conf = (recent == current).mean()
        transitioning = streak <= 2
        return {
            "state": current,
            "label": self.LABELS.get(current, f"State {current}"),
            "confidence": float(conf),
            "streak": streak,
            "transitioning": transitioning,
        }


# ===========================================================================
# LAYER 4 — CORRELATION MONITOR  (the Early Warning System)
# ===========================================================================
class CorrelationMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def assess(self, px: pd.DataFrame) -> dict:
        risk = [c for c in ("SPY", "QQQ", "IWM", "EEM", "HYG", "DBC", "BTC-USD")
                if c in px]
        rets = px[risk].pct_change().dropna()
        recent = rets.tail(self.cfg.corr_window)
        cm = recent.corr().values
        iu = np.triu_indices_from(cm, k=1)
        avg_corr = float(np.nanmean(cm[iu]))

        # baseline = median pairwise corr over the longer window
        base = rets.tail(self.cfg.var_window).corr().values
        base_corr = float(np.nanmedian(base[np.triu_indices_from(base, k=1)]))
        spike = avg_corr - base_corr

        if avg_corr >= self.cfg.corr_panic:
            level = "RED"
        elif avg_corr >= self.cfg.corr_panic - 0.15 or spike > 0.20:
            level = "YELLOW"
        else:
            level = "GREEN"
        return {"avg_corr": avg_corr, "baseline": base_corr,
                "spike": spike, "level": level}


# ===========================================================================
# LAYER 2 — SIGNAL GENERATOR  (regime-scoped edge)
# ===========================================================================
class SignalGenerator:
    """Blends trend-following and mean-reversion, weighted by regime.
    Calm  -> lean momentum.  Stressed -> lean defense + mean-reversion."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _scores(self, px: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for tk in self.cfg.universe:
            if tk not in px:
                continue
            s = px[tk].dropna()
            if len(s) < self.cfg.mom_slow + 5:
                continue
            ret = s.pct_change()
            mom_f = s.iloc[-1] / s.iloc[-self.cfg.mom_fast] - 1
            mom_s = s.iloc[-1] / s.iloc[-self.cfg.mom_slow] - 1
            trend = float(np.sign(mom_s) * (0.6 * mom_f + 0.4 * mom_s))
            z = (s.iloc[-1] - s.tail(self.cfg.mr_window).mean()) / \
                (s.tail(self.cfg.mr_window).std() + 1e-9)
            meanrev = float(-z)  # buy what's stretched down
            vol = float(ret.tail(self.cfg.vol_window).std() * np.sqrt(252))
            out[tk] = {"trend": trend, "meanrev": meanrev, "ann_vol": vol}
        return pd.DataFrame(out).T

    def generate(self, px: pd.DataFrame, regime: dict) -> pd.DataFrame:
        sc = self._scores(px)
        if sc.empty:
            return sc
        state = regime["state"]
        # regime-dependent blend weights
        w_trend, w_mr = {0: (0.80, 0.20),
                         1: (0.55, 0.45),
                         2: (0.25, 0.75)}[state]

        # normalize each leg cross-sectionally
        def zc(col):
            v = sc[col]
            return (v - v.mean()) / (v.std() + 1e-9)

        sc["raw"] = w_trend * zc("trend") + w_mr * zc("meanrev")

        # in stress, tilt toward classic safe havens
        safe = {"TLT": 0.5, "IEF": 0.4, "GLD": 0.5, "UUP": 0.4, "LQD": 0.2}
        if state == 2:
            for tk, bump in safe.items():
                if tk in sc.index:
                    sc.loc[tk, "raw"] += bump
        sc = sc.sort_values("raw", ascending=False)
        return sc


# ===========================================================================
# LAYER 1 — SIZING ENGINE  (Kelly-capped conviction)
# ===========================================================================
class SizingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def size(self, scored: pd.DataFrame, regime: dict,
             ejector: dict) -> pd.DataFrame:
        if scored.empty:
            return scored
        df = scored.copy()
        # conviction proxy: positive raw score, scaled
        df["conv"] = df["raw"].clip(lower=0)
        if df["conv"].sum() == 0:
            df["weight"] = 0.0
            return df

        # inverse-vol weighting * conviction (risk-parity flavored Kelly cap)
        inv_vol = 1.0 / (df["ann_vol"] + 1e-6)
        score = df["conv"] * inv_vol
        weights = score / score.sum()

        # total deployment scaled by regime + ejector state
        deploy = {0: 0.60, 1: 0.45, 2: 0.25}[regime["state"]]
        if regime["transitioning"]:
            deploy *= 0.5                      # half size during transitions
        if ejector["status"] == "YELLOW":
            deploy *= 0.5
        elif ejector["status"] == "RED":
            deploy = 0.0                        # ejector wins, go to cash
        deploy = min(deploy, self.cfg.max_total_risk)

        df["weight"] = (weights * deploy).round(4)
        return df.sort_values("weight", ascending=False)


# ===========================================================================
# LAYER 5 — EJECTOR SEAT  (hard-coded, bypasses everything above)
# ===========================================================================
class EjectorSeat:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _portfolio_var(self, px: pd.DataFrame) -> float:
        risk = [c for c in ("SPY", "QQQ", "IWM", "EEM", "HYG") if c in px]
        rets = px[risk].pct_change().dropna().tail(self.cfg.var_window)
        eqw = rets.mean(axis=1)
        return float(-np.percentile(eqw, (1 - self.cfg.var_conf) * 100))

    def assess(self, px: pd.DataFrame, corr: dict) -> dict:
        triggers = []
        var = self._portfolio_var(px)
        vix = float(px["^VIX"].iloc[-1]) if "^VIX" in px else np.nan

        if corr["level"] == "RED":
            triggers.append(f"Cross-asset corr {corr['avg_corr']:.2f} "
                            f">= panic {self.cfg.corr_panic:.2f}")
        if var >= self.cfg.var_ceiling:
            triggers.append(f"Daily 95% VaR {var:.2%} "
                            f">= ceiling {self.cfg.var_ceiling:.2%}")
        if not np.isnan(vix) and vix >= self.cfg.vix_panic:
            triggers.append(f"VIX {vix:.1f} >= panic {self.cfg.vix_panic:.0f}")

        if len(triggers) >= 2:
            status = "RED"          # multiple confirmations -> go to cash
        elif len(triggers) == 1:
            status = "YELLOW"       # one trigger -> cut risk in half
        else:
            status = "GREEN"
        return {"status": status, "triggers": triggers,
                "var": var, "vix": vix}


# ===========================================================================
# ORCHESTRATION
# ===========================================================================
class OmicronQuant:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.feed = DataFeed(cfg)
        self.regime = RegimeClassifier(cfg)
        self.corr = CorrelationMonitor(cfg)
        self.signals = SignalGenerator(cfg)
        self.sizer = SizingEngine(cfg)
        self.ejector = EjectorSeat(cfg)

    def run(self, demo: bool = False) -> dict:
        px = self.feed.fetch_demo() if demo else self.feed.fetch_live()

        labeled = self.regime.fit_predict(px)
        reg = self.regime.summary(labeled)
        corr = self.corr.assess(px)
        ejc = self.ejector.assess(px, corr)
        scored = self.signals.generate(px, reg)
        sized = self.sizer.size(scored, reg, ejc)

        return {"px": px, "regime": reg, "corr": corr,
                "ejector": ejc, "sized": sized}


# ===========================================================================
# OUTPUT
# ===========================================================================
def _verdict(row, reg_state, ejector_status):
    w = row["weight"]
    if ejector_status == "RED":
        return "CASH"
    if w >= 0.08:
        return "BUY"
    if w >= 0.03:
        return "ACCUMULATE"
    if row["raw"] > 0:
        return "WATCH"
    return "AVOID"


def render(cfg: Config, res: dict, demo: bool):
    reg, corr, ejc, sized = res["regime"], res["corr"], res["ejector"], res["sized"]
    ej_color = {"GREEN": "green", "YELLOW": "yellow", "RED": "red"}[ejc["status"]]

    if HAVE_RICH:
        _con.print()
        _con.print(Panel.fit(
            "[bold]OMICRON QUANT[/bold]  ·  Cross-Asset Signal Engine v1.0"
            + ("  [dim](DEMO / synthetic data)[/dim]" if demo else ""),
            box=box.DOUBLE, border_style="cyan"))

        eng = "HMM" if HAVE_HMM else "GaussianMixture (install hmmlearn for HMM)"
        _con.print(f"  Regime engine : [cyan]{eng}[/cyan]")
        trans = " [yellow](TRANSITIONING — sizes halved)[/yellow]" if reg["transitioning"] else ""
        _con.print(f"  Current regime: [bold]{reg['label']}[/bold]  "
                   f"conf {reg['confidence']:.0%}  ({reg['streak']}d){trans}")
        _con.print(f"  Cross-asset ρ : {corr['avg_corr']:.2f} "
                   f"(base {corr['baseline']:.2f}, spike {corr['spike']:+.2f}) "
                   f"→ [{ {'GREEN':'green','YELLOW':'yellow','RED':'red'}[corr['level']] }]{corr['level']}[/]")
        vixs = f"{ejc['vix']:.1f}" if not np.isnan(ejc['vix']) else "n/a"
        _con.print(f"  Portfolio VaR : {ejc['var']:.2%} (95%)   VIX: {vixs}")
        _con.print(f"  [bold {ej_color}]EJECTOR SEAT: {ejc['status']}[/bold {ej_color}]")
        for t in ejc["triggers"]:
            _con.print(f"     ⚠  {t}", style=ej_color)
        _con.print()

        if ejc["status"] == "RED":
            _con.print(Panel.fit(
                "[bold red]ALL POSITIONS → CASH[/bold red]\n"
                "Multiple systemic stress triggers fired. The ejector seat "
                "overrides all signals. Park capital. Do not deploy.",
                border_style="red"))
            return

        tbl = Table(box=box.SIMPLE_HEAVY, title="SIGNALS", title_style="bold")
        for col in ("Asset", "Class", "Action", "Size", "Trend", "MeanRev", "Vol"):
            tbl.add_column(col)
        shown = 0
        for tk, row in sized.iterrows():
            if shown >= cfg.top and row["weight"] == 0:
                continue
            verdict = _verdict(row, reg["state"], ejc["status"])
            vc = {"BUY": "bold green", "ACCUMULATE": "green",
                  "WATCH": "yellow", "AVOID": "dim", "CASH": "red"}[verdict]
            tbl.add_row(
                tk, cfg.universe.get(tk, ""),
                f"[{vc}]{verdict}[/{vc}]",
                f"{row['weight']:.1%}" if row["weight"] > 0 else "—",
                f"{row['trend']:+.2%}", f"{row['meanrev']:+.2f}",
                f"{row['ann_vol']:.0%}",
            )
            shown += 1
        _con.print(tbl)
        total = sized["weight"].sum()
        _con.print(f"  Total deployed: [bold]{total:.0%}[/bold]   "
                   f"Cash: [bold]{1-total:.0%}[/bold]")
        _con.print("\n  [dim]Advisory only. Not order execution. "
                   "Verify before acting.[/dim]\n")
    else:
        # plain-text fallback
        print("\n=== OMICRON QUANT v1.0" + (" (DEMO)" if demo else "") + " ===")
        print(f"Regime: {reg['label']} ({reg['confidence']:.0%}, {reg['streak']}d)"
              + ("  TRANSITIONING" if reg["transitioning"] else ""))
        print(f"Cross-asset corr: {corr['avg_corr']:.2f}  [{corr['level']}]")
        print(f"VaR: {ejc['var']:.2%}   EJECTOR: {ejc['status']}")
        for t in ejc["triggers"]:
            print("  ! " + t)
        if ejc["status"] == "RED":
            print("\n>>> ALL POSITIONS -> CASH. Ejector seat override.\n")
            return
        print(f"\n{'ASSET':<10}{'ACTION':<12}{'SIZE':<8}{'TREND':<10}{'VOL':<8}")
        for tk, row in sized.iterrows():
            v = _verdict(row, reg["state"], ejc["status"])
            print(f"{tk:<10}{v:<12}"
                  f"{(str(round(row['weight']*100,1))+'%') if row['weight']>0 else '-':<8}"
                  f"{row['trend']:+.2%}   {row['ann_vol']:.0%}")
        print(f"\nTotal deployed: {sized['weight'].sum():.0%}\n")


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Omicron Quant — cross-asset signal engine")
    p.add_argument("--demo", action="store_true",
                   help="run on synthetic data, no network required")
    p.add_argument("--lookback", type=int, default=504, help="trading days of history")
    p.add_argument("--top", type=int, default=6, help="max names to surface")
    p.add_argument("--states", type=int, default=3, help="number of regimes")
    args = p.parse_args(argv)

    cfg = Config(lookback=args.lookback, top=args.top, regime_states=args.states)
    engine = OmicronQuant(cfg)
    try:
        res = engine.run(demo=args.demo)
    except RuntimeError as e:
        print(f"\n[error] {e}\n")
        return 1
    render(cfg, res, demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
