"""
Coachella & Glamour Portfolio Optimizer
OPIM 5641 — Production Optimizer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, Bounds
import yfinance as yf
from datetime import datetime, timedelta
import json, os, warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────

# 25 stocks across 5 sectors (whittled → 10 each run)
STOCKS = {
    # ── FASHION / LUXURY (5) ──
    "LVMH":        ("LVMUY", "Fashion"),
    "Ralph Lauren":("RL",    "Fashion"),
    "Tapestry":    ("TPR",   "Fashion"),
    "Nike":        ("NKE",   "Fashion"),
    "Lululemon":   ("LULU",  "Fashion"),
    # ── BEAUTY / MAKEUP (5) ──
    "Estee Lauder":("EL",    "Beauty"),
    "Ulta Beauty": ("ULTA",  "Beauty"),
    "e.l.f. Beauty":("ELF",  "Beauty"),
    "Coty":        ("COTY",  "Beauty"),
    "Inter Parfums":("IPAR", "Beauty"),
    # ── MUSIC / ENTERTAINMENT (5) ──
    "Live Nation": ("LYV",   "Music"),
    "Warner Music":("WMG",   "Music"),
    "Spotify":     ("SPOT",  "Music"),
    "iHeartMedia": ("IHRT",  "Music"),
    "SiriusXM":    ("SIRI",  "Music"),
    # ── HEALTHCARE (5) ──
    "UnitedHealth":("UNH",   "Healthcare"),
    "Johnson & J": ("JNJ",   "Healthcare"),
    "Eli Lilly":   ("LLY",   "Healthcare"),
    "Pfizer":      ("PFE",   "Healthcare"),
    "Abbvie":      ("ABBV",  "Healthcare"),
    # ── TECHNOLOGY (5) ──
    "Apple":       ("AAPL",  "Tech"),
    "Microsoft":   ("MSFT",  "Tech"),
    "Nvidia":      ("NVDA",  "Tech"),
    "Meta":        ("META",  "Tech"),
    "Alphabet":    ("GOOGL", "Tech"),
}

SECTORS       = ["Fashion", "Beauty", "Music", "Healthcare", "Tech"]
N_SELECT      = 10          # stocks to pick from 25
MIN_WEIGHT    = 0.05        # 5% minimum if selected
MAX_WEIGHT    = 0.40        # 40% maximum if selected
WINDOW_DAYS   = 60          # sliding lookback window (calendar days → ~42 trading days)
N_FRONTIER    = 60          # points on efficient frontier
RESULTS_DIR   = "results"
PAPER_VALUE   = 10_000.0    # starting paper-trade portfolio value
VIX_THRESHOLD = 28.0        # fear-index threshold → min-variance mode
BACKTEST_DAYS = 45

SECTOR_COLORS = {
    "Fashion":    "#C084FC",
    "Beauty":     "#F472B6",
    "Music":      "#FB923C",
    "Healthcare": "#34D399",
    "Tech":       "#60A5FA",
}

# ──────────────────────────────────────────────────────────────
#  1. FETCH DATA
# ──────────────────────────────────────────────────────────────

def fetch_all(window_days: int):
    end = datetime.today()
    start = end - timedelta(days=window_days + BACKTEST_DAYS + 30)

    tickers = [v[0] for v in STOCKS.values()] + ["SPY", "^VIX"]

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")

    return prices

# ──────────────────────────────────────────────────────────────
#  2. FEAR INDEX (VIX) CHECK
# ──────────────────────────────────────────────────────────────

def get_vix_level(prices: pd.DataFrame) -> float:
    if "^VIX" in prices.columns:
        v = prices["^VIX"].dropna()
        return float(v.iloc[-1]) if len(v) > 0 else 0.0
    return 0.0

# ──────────────────────────────────────────────────────────────
#  3. GOLDEN CROSS SIGNAL
# ──────────────────────────────────────────────────────────────

def golden_cross_signal(prices: pd.DataFrame) -> dict:
    signals = {}
    stock_tickers = [v[0] for v in STOCKS.values()]
    for tkr in stock_tickers:
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        if len(s) < 10:
            signals[tkr] = "neutral"
            continue
        ma_short = s.rolling(min(10, len(s))).mean().iloc[-1]
        ma_long  = s.rolling(min(len(s), 30)).mean().iloc[-1]
        if ma_short > ma_long:
            signals[tkr] = "bullish"   # golden cross
        elif ma_short < ma_long * 0.97:
            signals[tkr] = "bearish"   # death cross
        else:
            signals[tkr] = "neutral"
    return signals

# ──────────────────────────────────────────────────────────────
#  4. EFFICIENT FRONTIER + SHARPE-MAX (scipy — no integers yet)
# ──────────────────────────────────────────────────────────────

def portfolio_stats(w, mu, cov, ann=252):
    ret  = float(np.dot(w, mu)) * ann
    risk = float(np.sqrt(w @ cov @ w * ann))
    return ret, risk

def compute_frontier(mu, cov, n_points=N_FRONTIER):
    n = len(mu)
    results = []

    target_returns = np.linspace(float(mu.min()) * 252,
                                  float(mu.max()) * 252, n_points)

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_stats(w, mu, cov)[0] - t},
        ]
        x0  = np.ones(n) / n
        res = minimize(lambda w: portfolio_stats(w, mu, cov)[1],
                       x0, method="SLSQP",
                       bounds=Bounds(0, 1),
                       constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success and abs(np.sum(res.x) - 1) < 1e-4:
            ret, risk = portfolio_stats(res.x, mu, cov)
            results.append((ret, risk, res.x))

    return results

def sharpe_max_point(frontier: list, rf: float = 0.05):
    """Auto-select the point with highest Sharpe ratio."""
    best_sharpe = -np.inf
    best = None
    for ret, risk, w in frontier:
        if risk > 0:
            s = (ret - rf) / risk
            if s > best_sharpe:
                best_sharpe = s
                best = (ret, risk, w, s)
    return best   # (return, risk, weights, sharpe)

# ──────────────────────────────────────────────────────────────
#  5. MILP-STYLE SELECTION (binary Y via iterative rounding)
#     Scipy doesn't support true MILP, so we use a penalty approach:
#     run continuous opt on each possible combination of 10 sectors,
#     enforce per-stock bounds [5%, 40%], sum=1, pick 10 best by Sharpe.
# ──────────────────────────────────────────────────────────────

def milp_optimize(mu: pd.Series, cov: pd.DataFrame,
                  min_w: float, max_w: float, n_select: int,
                  fear_mode: bool = False) -> tuple:
    """
    Implements linking constraints:
        Y_i ∈ {0,1}   (binary: stock selected or not)
        X_i = 0               if Y_i = 0
        min_w ≤ X_i ≤ max_w   if Y_i = 1
        Σ Y_i = n_select
        Σ X_i = 1

    Strategy: enumerate feasible subsets from each sector (at least 1 per sector),
    run continuous QP for each, keep best Sharpe.
    For 25 stocks → 10 chosen with sector balance, we do a smart greedy + local search.
    """
    tickers = list(mu.index)
    n       = len(tickers)
    names   = list(STOCKS.keys())
    sectors = [STOCKS[nm][1] for nm in names if STOCKS[nm][0] in tickers]

    # Map ticker → sector
    tkr_to_sector = {STOCKS[nm][0]: STOCKS[nm][1] for nm in names}

    def qp_fixed_subset(subset_idx):
        """Continuous QP with linking constraints on a fixed subset."""
        k   = len(subset_idx)
        mu_s  = mu.iloc[subset_idx].values
        cov_s = cov.iloc[subset_idx, subset_idx].values

        if fear_mode:
            # Min-variance mode (VIX triggered)
            obj = lambda w: float(w @ cov_s @ w)
        else:
            # Max Sharpe (minimize negative Sharpe)
            def obj(w):
                ret  = float(np.dot(w, mu_s)) * 252
                risk = float(np.sqrt(w @ cov_s @ w * 252))
                return -(ret - 0.05) / (risk + 1e-8)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds      = Bounds(min_w, max_w)   # linking: 5% ≤ X_i ≤ 40% for selected
        x0          = np.ones(k) / k

        res = minimize(obj, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-10, "maxiter": 1000})
        return res

    # ── Build sector-balanced candidate subsets ──
    sector_pools = {s: [] for s in SECTORS}
    for i, tkr in enumerate(tickers):
        s = tkr_to_sector.get(tkr, "Unknown")
        if s in sector_pools:
            sector_pools[s].append(i)

    # Greedy: pick top-2 by mean return from each sector (5×2=10)
    chosen = []
    for s in SECTORS:
        pool = sector_pools[s]
        if not pool:
            continue
        pool_sorted = sorted(pool, key=lambda i: float(mu.iloc[i]), reverse=True)
        chosen.extend(pool_sorted[:2])   # 2 per sector → 10 total

    chosen = list(dict.fromkeys(chosen))[:n_select]   # deduplicate, cap at 10

    # ── Run QP on the greedy subset ──
    best_result   = None
    best_sharpe   = -np.inf
    best_subset   = chosen

    res = qp_fixed_subset(chosen)
    if res.success:
        w_full = np.zeros(n)
        for rank, idx in enumerate(chosen):
            w_full[idx] = res.x[rank]
        ret, risk = portfolio_stats(w_full, mu.values, cov.values)
        sharpe = (ret - 0.05) / (risk + 1e-8)
        if sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_result  = (ret, risk, w_full, sharpe)
            best_subset  = chosen

    # ── Local search: swap one stock per sector, keep if better ──
    for swap_sector in SECTORS:
        pool = sector_pools[swap_sector]
        sector_in_chosen = [i for i in best_subset if tkr_to_sector.get(tickers[i]) == swap_sector]
        sector_not_chosen = [i for i in pool if i not in best_subset]

        for drop in sector_in_chosen:
            for add in sector_not_chosen:
                candidate = [i if i != drop else add for i in best_subset]
                res = qp_fixed_subset(candidate)
                if not res.success:
                    continue
                w_full = np.zeros(n)
                for rank, idx in enumerate(candidate):
                    w_full[idx] = res.x[rank]
                ret, risk = portfolio_stats(w_full, mu.values, cov.values)
                sharpe = (ret - 0.05) / (risk + 1e-8)
                if sharpe > best_sharpe:
                    best_sharpe  = sharpe
                    best_result  = (ret, risk, w_full, sharpe)
                    best_subset  = candidate

    if best_result is None:
        # Fallback: equal weight the greedy set
        w_full = np.zeros(n)
        for idx in chosen:
            w_full[idx] = 1.0 / len(chosen)
        ret, risk = portfolio_stats(w_full, mu.values, cov.values)
        best_result = (ret, risk, w_full, (ret - 0.05) / (risk + 1e-8))

    return best_result   # (ann_return, ann_risk, weights_array_len_n, sharpe)

# ──────────────────────────────────────────────────────────────
#  6. S&P 500 BUY-AND-HOLD COMPARISON
# ──────────────────────────────────────────────────────────────

def compute_spy_comparison(prices: pd.DataFrame) -> dict:
    if "SPY" not in prices.columns:
        return {}
    spy = prices["SPY"].dropna()
    spy_ret  = float((spy.iloc[-1] / spy.iloc[0]) - 1)
    spy_ann  = float((1 + spy_ret) ** (252 / len(spy)) - 1)
    spy_dd   = float(((spy / spy.cummax()) - 1).min())
    spy_vol  = float(spy.pct_change().dropna().std() * np.sqrt(252))
    return {
        "total_return":   round(spy_ret,  4),
        "annualized_ret": round(spy_ann,  4),
        "max_drawdown":   round(spy_dd,   4),
        "annualized_vol": round(spy_vol,  4),
        "sharpe":         round((spy_ann - 0.05) / (spy_vol + 1e-8), 3),
    }

# ──────────────────────────────────────────────────────────────
#  7. PAPER TRADING TRACKER
# ──────────────────────────────────────────────────────────────

def update_paper_trading(history_path: str, run_date: str,
                         ann_return: float, weights: np.ndarray,
                         tickers: list, prices: pd.DataFrame) -> dict:
    """
    Simulate $10,000 portfolio. Each day we 'cash out and rebuy' at 4PM prices.
    Track cumulative value and compare to SPY.
    """
    # Load or initialise
    if os.path.exists(history_path):
        df = pd.read_csv(history_path, parse_dates=["date"])
        df = df[df["date"].dt.strftime("%Y-%m-%d") != run_date]
    else:
        df = pd.DataFrame(columns=["date", "portfolio_value", "spy_value", "daily_return"])

    # Last portfolio value (or starting capital)
    if len(df) == 0:
        port_val = PAPER_VALUE
        spy_val  = PAPER_VALUE
    else:
        port_val = float(df["portfolio_value"].iloc[-1])
        spy_val  = float(df["spy_value"].iloc[-1])

    # Today's portfolio return ≈ weighted sum of daily stock returns
    daily_rets = prices[[t for t in tickers if t in prices.columns]].pct_change().iloc[-1]
    w_map = {tkr: w for tkr, w in zip(tickers, weights)}
    port_daily = sum(w_map.get(t, 0) * daily_rets.get(t, 0) for t in tickers)
    port_val *= (1 + port_daily)

    # SPY daily
    if "SPY" in prices.columns:
        spy_daily = float(prices["SPY"].pct_change().iloc[-1])
        spy_val  *= (1 + spy_daily)

    new_row = pd.DataFrame([{
        "date":            run_date,
        "portfolio_value": round(port_val, 2),
        "spy_value":       round(spy_val, 2),
        "daily_return":    round(port_daily, 6),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(history_path, index=False)

    return {"portfolio_value": round(port_val, 2), "spy_value": round(spy_val, 2)}

# ──────────────────────────────────────────────────────────────
#  8. CHARTS
# ──────────────────────────────────────────────────────────────

def plot_efficient_frontier(frontier, opt_ret, opt_risk, run_date, path):
    rets  = [f[0] for f in frontier]
    risks = [f[1] for f in frontier]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Colour frontier by Sharpe
    sharpes = [(r - 0.05) / (s + 1e-8) for r, s in zip(rets, risks)]
    sc = ax.scatter(risks, rets, c=sharpes, cmap="plasma", s=30, zorder=3, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Sharpe ratio", color="#cccccc", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#888")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaaaaa", fontsize=8)

    # Optimal point
    ax.scatter([opt_risk], [opt_ret], color="#f9c74f", s=220, zorder=5,
               marker="*", label=f"Optimal  Sharpe={((opt_ret-0.05)/(opt_risk+1e-8)):.2f}")

    ax.set_xlabel("Annualised Risk (std dev)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Annualised Return",          color="#cccccc", fontsize=10)
    ax.set_title(f"Efficient Frontier — {run_date}", color="#ffffff", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="#ffffff", fontsize=9)
    ax.grid(True, alpha=0.15, color="#444466")

    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Frontier chart → {path}")


def plot_allocation_pie(weights_map: dict, run_date: str, path: str):
    labels  = list(weights_map.keys())
    sizes   = list(weights_map.values())
    sectors = [STOCKS[nm][1] for nm in labels if nm in STOCKS]
    colors  = [SECTOR_COLORS.get(s, "#888888") for s in sectors]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.82, wedgeprops=dict(linewidth=0.8, edgecolor="#0f0f1a")
    )
    for t in texts:
        t.set_color("#dddddd"); t.set_fontsize(8)
    for at in autotexts:
        at.set_color("#ffffff"); at.set_fontsize(8); at.set_fontweight("bold")

    # Sector legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SECTOR_COLORS[s], label=s) for s in SECTORS]
    ax.legend(handles=legend_elements, loc="lower left", facecolor="#1a1a2e",
              edgecolor="#333355", labelcolor="#cccccc", fontsize=8)

    ax.set_title(f"Coachella & Glamour Portfolio — {run_date}",
                 color="#ffffff", fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Allocation pie → {path}")


def plot_paper_trading(paper_path: str, out_path: str):
    if not os.path.exists(paper_path):
        return
    df = pd.read_csv(paper_path, parse_dates=["date"])
    if len(df) < 2:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.plot(df["date"], df["portfolio_value"], color="#f9c74f", lw=2,
            label="Our Portfolio", zorder=3)
    if "spy_value" in df.columns:
        ax.plot(df["date"], df["spy_value"], color="#90e0ef", lw=1.5,
                linestyle="--", label="S&P 500 (SPY)", zorder=2)

    ax.fill_between(df["date"], df["portfolio_value"], PAPER_VALUE,
                    where=df["portfolio_value"] >= PAPER_VALUE,
                    alpha=0.15, color="#f9c74f")
    ax.fill_between(df["date"], df["portfolio_value"], PAPER_VALUE,
                    where=df["portfolio_value"] < PAPER_VALUE,
                    alpha=0.15, color="#ef233c")

    ax.axhline(PAPER_VALUE, color="#555577", lw=0.8, linestyle=":")
    ax.set_title("Paper Trading — $10,000 Simulated Portfolio vs S&P 500",
                 color="#ffffff", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Portfolio Value ($)", color="#aaaaaa", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="#ffffff", fontsize=9)
    ax.grid(True, alpha=0.12, color="#444466")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Paper trading chart → {out_path}")


def plot_weight_history(history_path: str, out_path: str):
    if not os.path.exists(history_path):
        return
    df = pd.read_csv(history_path, parse_dates=["date"])
    if len(df) < 2:
        return

    weight_cols = [c for c in df.columns if c not in
                   ["date", "ann_return", "ann_risk", "sharpe"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    palette = list(SECTOR_COLORS.values()) + ["#a8dadc", "#e9c46a", "#264653",
                                               "#2a9d8f", "#e76f51", "#f4a261"]
    for i, col in enumerate(weight_cols):
        if col in df.columns and df[col].sum() > 0:
            sector = STOCKS.get(col, ("", "Unknown"))[1]
            color  = SECTOR_COLORS.get(sector, palette[i % len(palette)])
            ax.plot(df["date"], df[col], label=col, lw=1.5,
                    color=color, alpha=0.85)

    ax.set_title("Daily Allocation Shifts Over Time",
                 color="#ffffff", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Weight", color="#aaaaaa", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="#ffffff",
              fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.12, color="#444466")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Weight history chart → {out_path}")


# ──────────────────────────────────────────────────────────────
#  9. PERSIST WEIGHT HISTORY
# ──────────────────────────────────────────────────────────────

def append_weight_history(path: str, run_date: str,
                           weights_map: dict,
                           ann_return: float, ann_risk: float, sharpe: float):
    row = {"date": run_date}
    row.update({nm: round(w, 6) for nm, w in weights_map.items()})
    row["ann_return"] = round(ann_return, 6)
    row["ann_risk"]   = round(ann_risk,   6)
    row["sharpe"]     = round(sharpe,     4)

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["date"] != run_date]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
    print(f"  ✓ Weight history → {path}")


# ──────────────────────────────────────────────────────────────
#  10. MARKDOWN REPORT
# ──────────────────────────────────────────────────────────────

def save_report(path, run_date, weights_map, ann_return, ann_risk, sharpe,
                spy_stats, vix, fear_mode, gc_signals, paper):
    spy_ret = spy_stats.get("total_return", 0)
    spy_ann = spy_stats.get("annualized_ret", 0)
    spy_sh  = spy_stats.get("sharpe", 0)

    lines = [
        f"# 🎶 Coachella & Glamour Portfolio — {run_date}",
        "",
        "> **Strategy:** MPT with binary linking constraints, 60-day sliding window",
        f"> **Mode:** {'⚠️ MIN-VARIANCE (VIX=' + f'{vix:.1f}' + ' > ' + str(VIX_THRESHOLD) + ')' if fear_mode else '🚀 MAX SHARPE'}",
        f"> **Stocks selected:** {len(weights_map)} / 25 &nbsp;|&nbsp; **Sectors:** 5",
        "",
        "---",
        "",
        "## 📊 Optimal Allocation (Linking Constraints: 5% ≤ X ≤ 40%)",
        "",
        "| Stock | Ticker | Sector | Weight | Binary Y |",
        "|-------|--------|--------|--------|----------|",
    ]
    for nm, w in sorted(weights_map.items(), key=lambda x: -x[1]):
        tkr, sec = STOCKS[nm]
        lines.append(f"| {nm} | `{tkr}` | {sec} | **{w:.2%}** | ✅ 1 |")

    # Stocks NOT selected
    lines += ["", "**Stocks excluded (Y=0, X=0):**"]
    excluded = [f"`{STOCKS[nm][0]}`" for nm in STOCKS if nm not in weights_map]
    lines.append(", ".join(excluded))

    lines += [
        "",
        "---",
        "",
        "## 📈 Portfolio Performance",
        "",
        f"| Metric | Our Portfolio | S&P 500 (SPY) |",
        f"|--------|--------------|---------------|",
        f"| Annualised Return | **{ann_return:.2%}** | {spy_ann:.2%} |",
        f"| Annualised Risk   | {ann_risk:.2%} | {spy_stats.get('annualized_vol', 0):.2%} |",
        f"| Sharpe Ratio      | **{sharpe:.3f}** | {spy_sh:.3f} |",
        f"| Max Drawdown      | — | {spy_stats.get('max_drawdown', 0):.2%} |",
        "",
        "---",
        "",
        "## 💰 Paper Trading (Simulated $10,000)",
        "",
        f"| Portfolio Value | SPY Value |",
        f"|----------------|-----------|",
        f"| **${paper.get('portfolio_value', PAPER_VALUE):,.2f}** | ${paper.get('spy_value', PAPER_VALUE):,.2f} |",
        "",
        f"Return vs Buy-and-Hold SPY: **{((paper.get('portfolio_value', PAPER_VALUE) / PAPER_VALUE) - 1):.2%}** vs {((paper.get('spy_value', PAPER_VALUE) / PAPER_VALUE) - 1):.2%}",
        "",
        "---",
        "",
        "## 🌡️ Fear Index & Market Signals",
        "",
        f"**VIX Level:** {vix:.1f} {'⚠️ HIGH — min-variance mode active' if fear_mode else '✅ Normal — max-Sharpe mode active'}",
        "",
        "**Golden Cross Signals (10d MA vs 30d MA):**",
        "",
        "| Ticker | Signal |",
        "|--------|--------|",
    ]
    for tkr, sig in gc_signals.items():
        emoji = "🟢" if sig == "bullish" else ("🔴" if sig == "bearish" else "⚪")
        lines.append(f"| `{tkr}` | {emoji} {sig.capitalize()} |")

    lines += [
        "",
        "---",
        "",
        "## 📉 Charts",
        "",
        "![Efficient Frontier](efficient_frontier.png)",
        "",
        "![Optimal Allocation](allocation_pie.png)",
        "",
        "![Paper Trading vs SPY](paper_trading.png)",
        "",
        "![Weight History](weight_history.png)",
        "",
        "---",
        f"*Auto-generated by GitHub Actions · {run_date} · OPIM 5641*",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Report → {path}")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def run_historical_backtest(prices: pd.DataFrame):
    """
    Creates historical time-series files immediately:
    1. paper_trading.csv = portfolio value vs SPY over time
    2. weight_history.csv = daily stock allocation weights over time
    3. paper_trading.png = value growth/shrink chart
    4. weight_history.png = allocation changes over time
    """

    print("\n🔁 Running historical backtest to create time-series charts...")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    stock_tickers = [v[0] for v in STOCKS.values()]
    available = [t for t in stock_tickers if t in prices.columns]

    if len(available) < N_SELECT:
        print("❌ Not enough tickers for backtest.")
        return

    clean_prices = prices[available + ["SPY"]].dropna(how="all")

    if len(clean_prices) < WINDOW_DAYS + 5:
        print("❌ Not enough price history for backtest.")
        return

    port_val = PAPER_VALUE
    spy_val = PAPER_VALUE

    paper_rows = []
    weight_rows = []

    start_i = max(WINDOW_DAYS, len(clean_prices) - BACKTEST_DAYS)

    for i in range(start_i, len(clean_prices)):
        current_date = clean_prices.index[i].strftime("%Y-%m-%d")

        train_window = clean_prices.iloc[i - WINDOW_DAYS:i]
        today_prices = clean_prices.iloc[i]
        yesterday_prices = clean_prices.iloc[i - 1]

        stock_window = train_window[available].dropna(how="all")
        returns = stock_window.pct_change().dropna()

        if len(returns) < 5:
            continue

        mu = returns.mean()
        cov = returns.cov()

        vix_slice = prices.loc[:clean_prices.index[i]]
        vix_level = get_vix_level(vix_slice)
        fear_mode = vix_level > VIX_THRESHOLD

        ann_return, ann_risk, weights_arr, sharpe = milp_optimize(
            mu, cov, MIN_WEIGHT, MAX_WEIGHT, N_SELECT, fear_mode=fear_mode
        )

        daily_rets = (today_prices[available] / yesterday_prices[available]) - 1
        w_map = {tkr: w for tkr, w in zip(available, weights_arr)}

        port_daily = sum(w_map.get(t, 0) * daily_rets.get(t, 0) for t in available)
        port_val *= (1 + port_daily)

        if "SPY" in clean_prices.columns:
            spy_daily = (today_prices["SPY"] / yesterday_prices["SPY"]) - 1
            spy_val *= (1 + spy_daily)

        paper_rows.append({
            "date": current_date,
            "portfolio_value": round(port_val, 2),
            "spy_value": round(spy_val, 2),
            "daily_return": round(float(port_daily), 6)
        })

        weight_row = {
            "date": current_date,
            "ann_return": round(ann_return, 6),
            "ann_risk": round(ann_risk, 6),
            "sharpe": round(sharpe, 4)
        }

        name_list = [nm for nm in STOCKS if STOCKS[nm][0] in available]

        for nm, w in zip(name_list, weights_arr):
            weight_row[nm] = round(float(w), 6)

        weight_rows.append(weight_row)

    paper_df = pd.DataFrame(paper_rows)
    weight_df = pd.DataFrame(weight_rows)

    paper_path = f"{RESULTS_DIR}/paper_trading.csv"
    weight_path = f"{RESULTS_DIR}/weight_history.csv"

    paper_df.to_csv(paper_path, index=False)
    weight_df.to_csv(weight_path, index=False)

    print(f"  ✓ Backtest paper trading rows: {len(paper_df)}")
    print(f"  ✓ Backtest weight history rows: {len(weight_df)}")

    plot_paper_trading(paper_path, f"{RESULTS_DIR}/paper_trading.png")
    plot_weight_history(weight_path, f"{RESULTS_DIR}/weight_history.png")

    print("  ✓ Time-series charts created.")

def main():
    run_date = datetime.today().strftime("%Y-%m-%d")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  🎶 Coachella & Glamour Optimizer — {run_date}")
    print(f"  Sectors: Fashion · Beauty · Music · Healthcare · Tech")
    print(f"  Window: {WINDOW_DAYS} days  |  Selecting {N_SELECT} of {len(STOCKS)} stocks")
    print(f"{'='*58}\n")

    # ── Fetch ─────────────────────────────────────────────────
    print("📥 Fetching market data (25 stocks + SPY + VIX)...")
    prices = fetch_all(WINDOW_DAYS)
    paper = run_historical_backtest(prices)
    return {
    "portfolio_value": round(float(paper_df["portfolio_value"].iloc[-1]), 2),
    "spy_value": round(float(paper_df["spy_value"].iloc[-1]), 2)
}

    stock_tickers = [v[0] for v in STOCKS.values()]
    available     = [t for t in stock_tickers if t in prices.columns]
    print(f"  Available: {len(available)} / {len(stock_tickers)} stock tickers")

    if len(available) < N_SELECT:
        print("❌ Not enough data. Exiting.")
        return

    # ── Returns ───────────────────────────────────────────────
    stock_prices = prices[available].dropna(how="all")
    returns      = stock_prices.pct_change().dropna()

    if len(returns) < 5:
        print("❌ Insufficient return history. Exiting.")
        return

    mu  = returns.mean()          # daily mean returns
    cov = returns.cov()           # daily covariance matrix

    print(f"\n📊 Daily mean returns computed over {len(returns)} trading days")

    # ── Fear index ────────────────────────────────────────────
    vix       = get_vix_level(prices)
    fear_mode = vix > VIX_THRESHOLD
    print(f"\n🌡️  VIX: {vix:.1f}  →  {'⚠️  HIGH — switching to min-variance mode' if fear_mode else '✅ Normal — max-Sharpe mode'}")

    # ── Golden Cross ──────────────────────────────────────────
    gc_signals = golden_cross_signal(prices)
    bullish    = [t for t, s in gc_signals.items() if s == "bullish"]
    bearish    = [t for t, s in gc_signals.items() if s == "bearish"]
    print(f"\n📡 Golden Cross — Bullish: {len(bullish)}  Bearish: {len(bearish)}")

    # ── MILP-style optimisation ────────────────────────────────
    print(f"\n⚙️  Running optimisation (binary Y + linking constraints)...")
    ann_return, ann_risk, weights_arr, sharpe = milp_optimize(
        mu, cov, MIN_WEIGHT, MAX_WEIGHT, N_SELECT, fear_mode=fear_mode
    )

    # Build weights map (only selected stocks > threshold)
    weights_map = {}
    name_list   = [nm for nm in STOCKS if STOCKS[nm][0] in available]
    for nm, w in zip(name_list, weights_arr):
        if w >= MIN_WEIGHT - 1e-6:
            weights_map[nm] = round(float(w), 4)

    print(f"\n✅ Optimal Allocation ({len(weights_map)} stocks selected):")
    for nm, w in sorted(weights_map.items(), key=lambda x: -x[1]):
        tkr, sec = STOCKS[nm]
        print(f"  {nm:18s} ({tkr:5s}) [{sec:10s}]: {w:.2%}")
    print(f"\n  Annualised Return: {ann_return:.2%}")
    print(f"  Annualised Risk:   {ann_risk:.2%}")
    print(f"  Sharpe Ratio:      {sharpe:.3f}")

    # ── Efficient frontier (for chart) ────────────────────────
    print("\n📈 Computing efficient frontier for chart...")
    sel_tickers = [STOCKS[nm][0] for nm in weights_map if STOCKS[nm][0] in available]
    mu_sel  = mu[sel_tickers]
    cov_sel = cov.loc[sel_tickers, sel_tickers]
    frontier = compute_frontier(mu_sel, cov_sel, n_points=N_FRONTIER)
    print(f"  {len(frontier)} feasible frontier points")

    # ── SPY comparison ────────────────────────────────────────
    spy_stats = compute_spy_comparison(prices)
    if spy_stats:
        print(f"\n📊 S&P 500 comparison: {spy_stats['annualized_ret']:.2%} ann. return | Sharpe {spy_stats['sharpe']:.3f}")

    # ── Save JSON summary ─────────────────────────────────────
    summary = {
        "run_date":   run_date,
        "mode":       "min_variance" if fear_mode else "max_sharpe",
        "vix":        round(vix, 2),
        "window_days": WINDOW_DAYS,
        "n_stocks_selected": len(weights_map),
        "n_stocks_pool":     len(STOCKS),
        "min_weight":  MIN_WEIGHT,
        "max_weight":  MAX_WEIGHT,
        "allocation":  {nm: {"ticker": STOCKS[nm][0], "sector": STOCKS[nm][1], "weight": w}
                        for nm, w in weights_map.items()},
        "ann_return":  round(ann_return, 4),
        "ann_risk":    round(ann_risk,   4),
        "sharpe":      round(sharpe,     4),
        "spy":         spy_stats,
        "paper_trading": paper,
        "golden_cross":  gc_signals,
    }
    with open(f"{RESULTS_DIR}/latest.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ JSON → {RESULTS_DIR}/latest.json")

    # ── Save charts ───────────────────────────────────────────
    print("\n💾 Saving charts...")
    if frontier:
        opt_pt = sharpe_max_point(frontier)
        if opt_pt:
            plot_efficient_frontier(frontier, opt_pt[0], opt_pt[1],
                                    run_date, f"{RESULTS_DIR}/efficient_frontier.png")

    plot_allocation_pie(weights_map, run_date, f"{RESULTS_DIR}/allocation_pie.png")
    plot_paper_trading(paper_path, f"{RESULTS_DIR}/paper_trading.png")
    plot_weight_history(wh_path, f"{RESULTS_DIR}/weight_history.png")

    # ── Save report ───────────────────────────────────────────
    save_report(f"{RESULTS_DIR}/REPORT.md", run_date,
                weights_map, ann_return, ann_risk, sharpe,
                spy_stats, vix, fear_mode, gc_signals, paper)

    print(f"\n🎉 Done! All outputs in ./{RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
