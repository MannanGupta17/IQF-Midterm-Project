from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


@dataclass
class StrategyConfig:
    name: str
    rebalance_frequency: str  # monthly or quarterly
    strategy_type: str = "target_markowitz"  # target_markowitz or equal_weight
    covariance_estimator: str = "ledoit_wolf"  # ledoit_wolf or sample
    transaction_cost_rate: float = 0.0
    transaction_cost_penalty: float = 0.0
    turnover_limit: Optional[float] = None
    long_only: bool = True
    max_weight: Optional[float] = None
    target_return_mode: str = "equal_weight"
    target_return_scale: float = 1.0
    max_assets: Optional[int] = None
    shortlist_buffer: int = 2


# ------------------------------
# Data loading and preprocessing
# ------------------------------

def _parse_date_series(date_series: pd.Series) -> pd.Series:
    s = date_series.astype(str).str.strip()
    candidates = [
        pd.to_datetime(s, errors="coerce"),
        pd.to_datetime(s, errors="coerce", dayfirst=True),
        pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"),
        pd.to_datetime(s, format="%d-%m-%Y", errors="coerce"),
        pd.to_datetime(s, format="%m-%d-%Y", errors="coerce"),
        pd.to_datetime(s, format="%Y/%m/%d", errors="coerce"),
        pd.to_datetime(s, format="%d/%m/%Y", errors="coerce"),
        pd.to_datetime(s, format="%m/%d/%Y", errors="coerce"),
    ]
    best = max(candidates, key=lambda x: x.notna().sum())
    if best.notna().sum() == 0:
        raise ValueError("Could not parse the Date column.")
    bad_rows = s[best.isna()]
    if len(bad_rows) > 0:
        sample = ", ".join(bad_rows.head(5).tolist())
        raise ValueError(f"Some dates could not be parsed: {sample}")
    return best


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = _parse_date_series(df["Date"])
        df = df.set_index("Date")
    else:
        parsed_index = _parse_date_series(pd.Series(df.index.astype(str), index=df.index))
        df.index = parsed_index.values
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def load_price_data(csv_path: str) -> pd.DataFrame:
    prices = pd.read_csv(csv_path)
    prices = ensure_datetime_index(prices)
    prices = prices.dropna(how="all").ffill().dropna()
    return prices


def simulate_price_data(
    n_assets: int = 6,
    n_days: int = 2000,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    annual_means = np.array([0.10, 0.14, 0.08, 0.12, 0.07, 0.11])[:n_assets]
    annual_vols = np.array([0.16, 0.23, 0.12, 0.20, 0.10, 0.18])[:n_assets]
    market_loading = np.array([0.9, 1.1, 0.7, 1.0, 0.5, 0.8])[:n_assets]
    style_loading = np.array([0.4, -0.2, 0.5, -0.1, 0.2, 0.3])[:n_assets]
    mu_daily = annual_means / TRADING_DAYS
    vol_daily = annual_vols / np.sqrt(TRADING_DAYS)
    market_factor = rng.normal(0.00015, 0.008, size=n_days)
    style_factor = rng.normal(0.00005, 0.006, size=n_days)
    idio = rng.normal(0.0, 1.0, size=(n_days, n_assets))
    raw = (
        mu_daily
        + np.outer(market_factor, market_loading)
        + np.outer(style_factor, style_loading)
        + idio * (vol_daily * 0.55)
    )
    for t in range(1, n_days):
        raw[t] += 0.08 * raw[t - 1]
    raw = np.clip(raw, -0.18, 0.18)
    prices = start_price * np.cumprod(1.0 + raw, axis=0)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    cols = [f"Asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ------------------------------
# Backtest helpers
# ------------------------------

def get_rebalance_starts(index: pd.DatetimeIndex, lookback: int, frequency: str) -> List[int]:
    if frequency == "monthly":
        periods = index.to_period("M")
    elif frequency == "quarterly":
        periods = index.to_period("Q")
    else:
        raise ValueError("frequency must be 'monthly' or 'quarterly'")

    starts = [lookback]
    for i in range(lookback + 1, len(index)):
        if periods[i] != periods[i - 1]:
            starts.append(i)
    return sorted(set(starts))


def max_drawdown(wealth_index: pd.Series) -> float:
    running_max = wealth_index.cummax()
    drawdown = wealth_index / running_max - 1.0
    return float(drawdown.min())


def annualized_return(daily_returns: pd.Series) -> float:
    compounded = (1.0 + daily_returns).prod()
    n = len(daily_returns)
    if n == 0:
        return np.nan
    return float(compounded ** (TRADING_DAYS / n) - 1.0)


def annualized_vol(daily_returns: pd.Series) -> float:
    if len(daily_returns) == 0:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_vol(daily_returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_ret - rf) / ann_vol)


def estimate_mu_and_cov(
    estimation_window: pd.DataFrame,
    horizon_days: int,
    covariance_estimator: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mu = estimation_window.mean().values * horizon_days
    daily_values = estimation_window.values

    if covariance_estimator == "ledoit_wolf":
        lw = LedoitWolf().fit(daily_values)
        cov_daily = lw.covariance_
    elif covariance_estimator == "sample":
        cov_daily = estimation_window.cov().values
    else:
        raise ValueError("covariance_estimator must be 'ledoit_wolf' or 'sample'")

    cov = cov_daily * horizon_days
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(cov.shape[0]) * 1e-10
    return mu, cov


def compute_target_return(mu: np.ndarray, mode: str = "equal_weight", scale: float = 1.0) -> float:
    if mode == "equal_weight":
        base = float(np.mean(mu))
    elif mode == "median":
        base = float(np.median(mu))
    else:
        raise ValueError("Unsupported target_return_mode")
    return scale * base


def make_initial_guess(
    prev_weights: np.ndarray,
    active_idx: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    guess = prev_weights[active_idx].copy()
    if guess.sum() <= 0:
        guess = np.repeat(1.0 / len(active_idx), len(active_idx))
    else:
        guess = guess / guess.sum()
    guess = np.clip(guess, lower_bound, upper_bound)
    if guess.sum() <= 0:
        guess = np.repeat(1.0 / len(active_idx), len(active_idx))
    else:
        guess = guess / guess.sum()
    return guess


def solve_target_markowitz_on_subset(
    mu: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray,
    config: StrategyConfig,
    target_return: float,
    active_idx: np.ndarray,
) -> Tuple[Optional[np.ndarray], float, bool]:
    n = len(mu)
    active_idx = np.array(active_idx, dtype=int)
    mu_sub = mu[active_idx]
    cov_sub = cov[np.ix_(active_idx, active_idx)]

    lower_bound = 0.0 if config.long_only else -1.0
    upper_bound = 1.0 if config.max_weight is None else config.max_weight
    bounds = [(lower_bound, upper_bound) for _ in range(len(active_idx))]

    def assemble_full(x: np.ndarray) -> np.ndarray:
        w = np.zeros(n)
        w[active_idx] = x
        return w

    def objective(x: np.ndarray) -> float:
        w_full = assemble_full(x)
        variance_term = 0.5 * float(w_full @ cov @ w_full)
        turnover_penalty = config.transaction_cost_penalty * float(np.abs(w_full - prev_weights).sum())
        return variance_term + turnover_penalty

    constraints = [
        {"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
        {"type": "ineq", "fun": lambda x: float(x @ mu_sub - target_return)},
    ]

    if config.turnover_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: float(config.turnover_limit - np.abs(assemble_full(x) - prev_weights).sum()),
            }
        )

    x0 = make_initial_guess(prev_weights, active_idx, lower_bound, upper_bound)
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-9, "disp": False},
    )

    if not result.success:
        x0 = np.repeat(1.0 / len(active_idx), len(active_idx))
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-9, "disp": False},
        )

    if not result.success:
        return None, np.inf, False

    w = assemble_full(result.x)
    w = np.clip(w, 0.0 if config.long_only else -1.0, upper_bound)
    if w.sum() <= 0:
        return None, np.inf, False
    w = w / w.sum()

    if float(w @ mu) + 1e-8 < target_return:
        return None, np.inf, False
    if config.turnover_limit is not None and np.abs(w - prev_weights).sum() > config.turnover_limit + 1e-6:
        return None, np.inf, False

    return w, objective(result.x), True


def shortlist_assets(mu: np.ndarray, cov: np.ndarray, prev_weights: np.ndarray, k: int, buffer: int) -> np.ndarray:
    """
    Build a small ranked list of candidate assets for cardinality-constrained allocation.
    This is a fast heuristic shortlist rather than an exact mixed-integer search.
    """
    n = len(mu)
    shortlist_size = min(n, k + max(buffer, 0))
    vols = np.sqrt(np.diag(cov))
    risk_adj = mu / np.maximum(vols, 1e-8)

    mu_rank = pd.Series(mu).rank(method="first", ascending=False)
    prev_rank = pd.Series(prev_weights).rank(method="first", ascending=False)
    risk_rank = pd.Series(risk_adj).rank(method="first", ascending=False)
    combined = mu_rank + 0.5 * prev_rank + 0.5 * risk_rank
    idx = np.argsort(combined.values)[:shortlist_size]
    return np.array(sorted(idx), dtype=int)


def build_candidate_supports(mu: np.ndarray, cov: np.ndarray, prev_weights: np.ndarray, k: int) -> List[np.ndarray]:
    n = len(mu)
    vols = np.sqrt(np.diag(cov))
    risk_adj = mu / np.maximum(vols, 1e-8)

    supports: List[np.ndarray] = []
    candidate_sets = [
        np.argsort(-mu)[:k],
        np.argsort(-prev_weights)[:k],
        np.argsort(-risk_adj)[:k],
        shortlist_assets(mu, cov, prev_weights, k, buffer=0)[:k],
    ]

    # Mixed support: keep some existing positions and some high-alpha names.
    keep_prev = list(np.argsort(-prev_weights)[: max(1, k // 2)])
    add_mu = list(np.argsort(-mu)[: k])
    mixed = []
    for idx in keep_prev + add_mu:
        if idx not in mixed:
            mixed.append(int(idx))
        if len(mixed) == k:
            break
    candidate_sets.append(np.array(mixed, dtype=int))

    seen = set()
    for arr in candidate_sets:
        arr = np.array(sorted(set(int(x) for x in arr)), dtype=int)
        if len(arr) < k:
            fill = [int(x) for x in np.argsort(-mu) if int(x) not in set(arr)]
            arr = np.array(sorted(list(arr) + fill[: k - len(arr)]), dtype=int)
        key = tuple(arr.tolist())
        if len(arr) == k and key not in seen:
            seen.add(key)
            supports.append(arr)

    if not supports:
        supports.append(np.array(sorted(np.argsort(-mu)[:k]), dtype=int))
    return supports


def optimise_target_markowitz_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray,
    config: StrategyConfig,
    target_return: float,
) -> np.ndarray:
    n = len(mu)

    if config.max_assets is None or config.max_assets >= n:
        w, _, success = solve_target_markowitz_on_subset(
            mu, cov, prev_weights, config, target_return, np.arange(n)
        )
        if success and w is not None:
            return w
    else:
        k = config.max_assets
        best_w = None
        best_obj = np.inf
        for support in build_candidate_supports(mu, cov, prev_weights, k):
            w, obj, success = solve_target_markowitz_on_subset(
                mu, cov, prev_weights, config, target_return, support
            )
            if success and w is not None and obj < best_obj:
                best_w = w
                best_obj = obj
        if best_w is not None:
            return best_w

    # Robust fallback: keep current allocation if feasible, else equal weight.
    fallback = prev_weights.copy()
    if fallback.sum() <= 0:
        fallback = np.repeat(1.0 / n, n)
    fallback = np.clip(fallback, 0.0 if config.long_only else -1.0, 1.0)
    fallback /= fallback.sum()
    return fallback


def get_target_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray,
    config: StrategyConfig,
    n_assets: int,
) -> Tuple[np.ndarray, Optional[float]]:
    if config.strategy_type == "equal_weight":
        return np.repeat(1.0 / n_assets, n_assets), None

    target_return = compute_target_return(
        mu, mode=config.target_return_mode, scale=config.target_return_scale
    )
    weights = optimise_target_markowitz_weights(mu, cov, prev_weights, config, target_return)
    return weights, target_return




def suggest_initial_cardinality_weights(
    returns: pd.DataFrame,
    frequency: str,
    max_assets: int,
    lookback: int = 252,
) -> np.ndarray:
    n_assets = returns.shape[1]
    prev = np.repeat(1.0 / n_assets, n_assets)
    starts = get_rebalance_starts(returns.index, lookback, frequency)
    start_idx = starts[0]
    next_start = starts[1] if len(starts) > 1 else len(returns.index)
    horizon_days = next_start - start_idx
    window = returns.iloc[start_idx - lookback : start_idx]
    mu, cov = estimate_mu_and_cov(window, horizon_days, covariance_estimator="ledoit_wolf")
    support = build_candidate_supports(mu, cov, prev, max_assets)[0]
    w = np.zeros(n_assets)
    w[support] = 1.0 / len(support)
    return w


def backtest_strategy(
    returns: pd.DataFrame,
    config: StrategyConfig,
    lookback: int = 252,
    initial_weights: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame | pd.Series | float]:
    n_assets = returns.shape[1]
    asset_names = list(returns.columns)
    index = returns.index

    if initial_weights is None:
        current_weights = np.repeat(1.0 / n_assets, n_assets)
    else:
        current_weights = np.array(initial_weights, dtype=float)
        current_weights = current_weights / current_weights.sum()

    rebalance_starts = get_rebalance_starts(index, lookback, config.rebalance_frequency)
    segment_starts = rebalance_starts if rebalance_starts[-1] != len(index) else rebalance_starts[:-1]

    wealth = 1.0
    wealth_path = []
    net_daily_returns = []
    daily_weights = pd.DataFrame(index=index[lookback:], columns=asset_names, dtype=float)
    rebalance_rows = []
    turnover_rows = []
    target_return_rows = []

    for seg_i, start_idx in enumerate(segment_starts):
        next_start = segment_starts[seg_i + 1] if seg_i + 1 < len(segment_starts) else len(index)
        if next_start <= start_idx:
            continue

        horizon_days = next_start - start_idx
        estimation_window = returns.iloc[start_idx - lookback : start_idx]
        mu, cov = estimate_mu_and_cov(
            estimation_window=estimation_window,
            horizon_days=horizon_days,
            covariance_estimator=config.covariance_estimator,
        )

        target_weights, target_return = get_target_weights(mu, cov, current_weights, config, n_assets)
        turnover = float(np.abs(target_weights - current_weights).sum())
        trade_cost = config.transaction_cost_rate * turnover
        wealth *= (1.0 - trade_cost)

        rebalance_date = index[start_idx]
        rebalance_rows.append(pd.Series(target_weights, index=asset_names, name=rebalance_date))
        turnover_rows.append({
            "Date": rebalance_date,
            "Turnover": turnover,
            "TradingCostFraction": trade_cost,
            "ActivePositions": int(np.sum(np.abs(target_weights) > 1e-10)),
        })
        if target_return is not None:
            target_return_rows.append({"Date": rebalance_date, "TargetReturn": target_return})

        current_weights = target_weights.copy()

        for t in range(start_idx, next_start):
            r = returns.iloc[t].values
            portfolio_return = float(current_weights @ r)
            wealth *= (1.0 + portfolio_return)

            effective_return = portfolio_return
            if t == start_idx and trade_cost > 0:
                effective_return = (1.0 - trade_cost) * (1.0 + portfolio_return) - 1.0

            wealth_path.append((index[t], wealth))
            net_daily_returns.append((index[t], effective_return))
            daily_weights.loc[index[t]] = current_weights

            gross_asset_growth = current_weights * (1.0 + r)
            gross_portfolio_growth = 1.0 + portfolio_return
            if gross_portfolio_growth <= 0:
                current_weights = np.repeat(1.0 / n_assets, n_assets)
            else:
                current_weights = gross_asset_growth / gross_portfolio_growth

    wealth_series = pd.Series([x[1] for x in wealth_path], index=[x[0] for x in wealth_path], name=config.name)
    returns_series = pd.Series([x[1] for x in net_daily_returns], index=[x[0] for x in net_daily_returns], name=config.name)

    rebalance_weights = pd.DataFrame(rebalance_rows)
    if not rebalance_weights.empty:
        rebalance_weights.index.name = "Date"

    turnover_df = pd.DataFrame(turnover_rows)
    if not turnover_df.empty:
        turnover_df = turnover_df.set_index("Date")

    target_return_df = pd.DataFrame(target_return_rows)
    if not target_return_df.empty:
        target_return_df = target_return_df.set_index("Date")

    metrics = {
        "Strategy": config.name,
        "AnnualReturn": annualized_return(returns_series),
        "AnnualVolatility": annualized_vol(returns_series),
        "Sharpe": sharpe_ratio(returns_series),
        "MaxDrawdown": max_drawdown(wealth_series),
        "AverageTurnover": float(turnover_df["Turnover"].mean()) if not turnover_df.empty else np.nan,
        "TotalTurnover": float(turnover_df["Turnover"].sum()) if not turnover_df.empty else np.nan,
        "TotalTradingCostFraction": float(turnover_df["TradingCostFraction"].sum()) if not turnover_df.empty else 0.0,
        "AverageActivePositions": float(turnover_df["ActivePositions"].mean()) if not turnover_df.empty else np.nan,
        "NumRebalances": int(len(turnover_df)),
        "CovarianceEstimator": config.covariance_estimator,
        "StrategyType": config.strategy_type,
        "MaxAssets": config.max_assets if config.max_assets is not None else n_assets,
        "TurnoverCap": config.turnover_limit,
    }

    return {
        "wealth": wealth_series,
        "daily_returns": returns_series,
        "daily_weights": daily_weights,
        "rebalance_weights": rebalance_weights,
        "turnover": turnover_df,
        "target_return": target_return_df,
        "metrics": metrics,
    }


# ------------------------------
# Reporting and plotting
# ------------------------------

def compare_weight_instability(baseline_weights: pd.DataFrame, constrained_weights: pd.DataFrame) -> pd.DataFrame:
    common_index = baseline_weights.index.intersection(constrained_weights.index)
    if len(common_index) == 0:
        return pd.DataFrame()
    base = baseline_weights.loc[common_index]
    cons = constrained_weights.loc[common_index]
    diff = (cons - base).abs()
    return pd.DataFrame(
        {
            "MeanAbsWeightDifference": diff.mean(axis=1),
            "MaxAbsWeightDifference": diff.max(axis=1),
        },
        index=common_index,
    )


def plot_cumulative_wealth(wealth_dict: Dict[str, pd.Series], output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    for name, wealth in wealth_dict.items():
        plt.plot(wealth.index, wealth.values, label=name)
    plt.title("Cumulative Wealth")
    plt.xlabel("Date")
    plt.ylabel("Wealth Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_turnover(turnover_dict: Dict[str, pd.DataFrame], output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    for name, turnover_df in turnover_dict.items():
        if turnover_df is not None and not turnover_df.empty:
            plt.plot(turnover_df.index, turnover_df["Turnover"], marker="o", label=name)
    plt.title("Turnover at Rebalance Dates")
    plt.xlabel("Date")
    plt.ylabel("L1 Turnover")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_outputs(results: Dict[str, Dict[str, pd.DataFrame | pd.Series | float]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary = pd.DataFrame([results[name]["metrics"] for name in results]).sort_values("Sharpe", ascending=False)
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    wealth_df = pd.concat([results[name]["wealth"] for name in results], axis=1).dropna(how="all")
    wealth_df.to_csv(os.path.join(output_dir, "cumulative_wealth.csv"))

    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        res["daily_weights"].to_csv(os.path.join(output_dir, f"daily_weights_{safe_name}.csv"))
        if isinstance(res["rebalance_weights"], pd.DataFrame) and not res["rebalance_weights"].empty:
            res["rebalance_weights"].to_csv(os.path.join(output_dir, f"rebalance_weights_{safe_name}.csv"))
        if isinstance(res["turnover"], pd.DataFrame) and not res["turnover"].empty:
            res["turnover"].to_csv(os.path.join(output_dir, f"turnover_{safe_name}.csv"))
        if isinstance(res["target_return"], pd.DataFrame) and not res["target_return"].empty:
            res["target_return"].to_csv(os.path.join(output_dir, f"target_return_{safe_name}.csv"))

    plot_cumulative_wealth({name: results[name]["wealth"] for name in results}, os.path.join(output_dir, "cumulative_wealth.png"))
    plot_turnover({name: results[name]["turnover"] for name in results}, os.path.join(output_dir, "turnover_by_rebalance.png"))


# ------------------------------
# Project runner
# ------------------------------

def run_project(prices: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    returns = compute_returns(prices)
    n_assets = returns.shape[1]

    # With only 10 industry portfolios, a 15-20 stock cardinality rule is impossible.
    # We therefore impose a feasible cardinality cap for this dataset.
    feasible_cardinality_cap = min(5, n_assets)

    strategies = [
        StrategyConfig(
            name="Standard Markowitz (target-return, frictionless monthly)",
            rebalance_frequency="monthly",
            strategy_type="target_markowitz",
            covariance_estimator="ledoit_wolf",
            transaction_cost_rate=0.0,
            transaction_cost_penalty=0.0,
            turnover_limit=None,
            max_assets=None,
        ),
        StrategyConfig(
            name="Standard Markowitz + realised costs",
            rebalance_frequency="monthly",
            strategy_type="target_markowitz",
            covariance_estimator="ledoit_wolf",
            transaction_cost_rate=0.0010,
            transaction_cost_penalty=0.0,
            turnover_limit=None,
            max_assets=None,
        ),
        StrategyConfig(
            name=f"Friction-adjusted Markowitz (monthly, max {feasible_cardinality_cap} assets)",
            rebalance_frequency="monthly",
            strategy_type="target_markowitz",
            covariance_estimator="ledoit_wolf",
            transaction_cost_rate=0.0010,
            transaction_cost_penalty=0.0025,
            turnover_limit=0.15,
            max_assets=feasible_cardinality_cap,
        ),
        StrategyConfig(
            name=f"Friction-adjusted Markowitz (quarterly, max {feasible_cardinality_cap} assets)",
            rebalance_frequency="quarterly",
            strategy_type="target_markowitz",
            covariance_estimator="ledoit_wolf",
            transaction_cost_rate=0.0010,
            transaction_cost_penalty=0.0025,
            turnover_limit=0.15,
            max_assets=feasible_cardinality_cap,
        ),
        StrategyConfig(
            name="Equal Weight Monthly",
            rebalance_frequency="monthly",
            strategy_type="equal_weight",
            covariance_estimator="ledoit_wolf",
            transaction_cost_rate=0.0010,
            transaction_cost_penalty=0.0,
            turnover_limit=None,
            max_assets=n_assets,
        ),
        StrategyConfig(
            name="Equal Weight Quarterly",
            rebalance_frequency="quarterly",
            strategy_type="equal_weight",
            covariance_estimator="ledoit_wolf",
            transaction_cost_rate=0.0010,
            transaction_cost_penalty=0.0,
            turnover_limit=None,
            max_assets=n_assets,
        ),
    ]

    results: Dict[str, Dict] = {}
    for config in strategies:
        initial_weights = None
        if config.max_assets is not None and config.max_assets < n_assets and config.strategy_type == "target_markowitz":
            initial_weights = suggest_initial_cardinality_weights(
                returns=returns,
                frequency=config.rebalance_frequency,
                max_assets=config.max_assets,
                lookback=252,
            )
        results[config.name] = backtest_strategy(
            returns,
            config=config,
            lookback=252,
            initial_weights=initial_weights,
        )

    save_outputs(results, output_dir)

    summary = pd.DataFrame([results[name]["metrics"] for name in results]).sort_values("Sharpe", ascending=False)

    baseline = results["Standard Markowitz (target-return, frictionless monthly)"]["rebalance_weights"]
    constrained = results[f"Friction-adjusted Markowitz (monthly, max {feasible_cardinality_cap} assets)"]["rebalance_weights"]
    instability = compare_weight_instability(baseline, constrained)
    if not instability.empty:
        instability.to_csv(os.path.join(output_dir, "weight_difference_standard_vs_friction_adjusted.csv"))

    return summary, results


# ------------------------------
# CLI
# ------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mean-variance midterm project")
    parser.add_argument("--prices_csv", type=str, default=None, help="Path to CSV of prices. If omitted, synthetic data will be generated.")
    parser.add_argument("--output_dir", type=str, default="mv_midterm_outputs", help="Directory where outputs will be saved.")
    parser.add_argument("--save_synthetic_prices", action="store_true", help="Save the internally generated synthetic price data.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.prices_csv is None:
        prices = simulate_price_data()
        if args.save_synthetic_prices:
            os.makedirs(args.output_dir, exist_ok=True)
            prices.to_csv(os.path.join(args.output_dir, "synthetic_prices.csv"))
    else:
        prices = load_price_data(args.prices_csv)

    summary, _ = run_project(prices=prices, output_dir=args.output_dir)
    print("\nSummary metrics")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
