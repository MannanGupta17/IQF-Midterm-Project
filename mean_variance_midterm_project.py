
"""
Mean-Variance Midterm Project
-----------------------------

This script builds a complete project around a theoretical mean-variance
portfolio and then introduces practical implementation constraints:
1. transaction costs,
2. turnover limits,
3. lower-frequency (periodic) rebalancing.

It can work with:
- a user-provided CSV of prices, or
- internally generated synthetic price data for demonstration.

CSV format expected:
Date,Asset_A,Asset_B,Asset_C,...
2020-01-01,100,100,100,...
2020-01-02,101,99,100.4,...

Outputs:
- summary_metrics.csv
- cumulative_wealth.csv
- daily_weights_<strategy>.csv
- rebalance_weights_<strategy>.csv
- cumulative_wealth.png
- turnover_by_rebalance.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception as exc:
    raise ImportError(
        "This project requires scipy. Please install it with: pip install scipy"
    ) from exc


TRADING_DAYS = 252


@dataclass
class StrategyConfig:
    name: str
    rebalance_frequency: str  # "monthly" or "quarterly"
    risk_aversion: float = 4.0
    long_only: bool = True
    max_weight: Optional[float] = None
    transaction_cost_rate: float = 0.0     # realised cost deducted from wealth
    transaction_cost_penalty: float = 0.0  # optimisation penalty on turnover
    turnover_limit: Optional[float] = None
    allow_short: bool = False


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_price_data(csv_path: str) -> pd.DataFrame:
    prices = pd.read_csv(csv_path)
    prices = ensure_datetime_index(prices)
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()
    return prices


def simulate_price_data(
    n_assets: int = 6,
    n_days: int = 2000,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """
    Create synthetic daily prices with a factor structure.
    This makes the project fully runnable even without external data.
    """
    rng = np.random.default_rng(seed)

    annual_means = np.array([0.10, 0.14, 0.08, 0.12, 0.07, 0.11])[:n_assets]
    annual_vols = np.array([0.16, 0.23, 0.12, 0.20, 0.10, 0.18])[:n_assets]

    # One market factor + one style factor + idiosyncratic noise.
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

    # Add some mild serial variation to make estimation noisier and weights less stable.
    for t in range(1, n_days):
        raw[t] += 0.08 * raw[t - 1]

    raw = np.clip(raw, -0.18, 0.18)
    prices = start_price * np.cumprod(1.0 + raw, axis=0)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    cols = [f"Asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns


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
    starts = sorted(set(starts))
    return starts


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


def smooth_abs(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.sqrt(x * x + eps)


def optimise_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray,
    config: StrategyConfig,
) -> np.ndarray:
    n = len(mu)

    if config.long_only:
        lower_bound = 0.0
        upper_bound = 1.0 if config.max_weight is None else config.max_weight
    elif config.allow_short:
        lower_bound = -1.0 if config.max_weight is None else -config.max_weight
        upper_bound = 1.0 if config.max_weight is None else config.max_weight
    else:
        lower_bound = 0.0
        upper_bound = 1.0 if config.max_weight is None else config.max_weight

    bounds = [(lower_bound, upper_bound) for _ in range(n)]

    if config.long_only and config.max_weight is not None and config.max_weight * n < 1.0:
        raise ValueError("max_weight is too small to satisfy the budget constraint.")

    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(n)

    def objective(w: np.ndarray) -> float:
        variance_term = 0.5 * config.risk_aversion * float(w @ cov @ w)
        return_term = -float(mu @ w)
        cost_term = config.transaction_cost_penalty * smooth_abs(w - prev_weights).sum()
        return variance_term + return_term + cost_term

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if config.turnover_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: config.turnover_limit - smooth_abs(w - prev_weights).sum(),
            }
        )

    x0 = np.clip(prev_weights.copy(), lower_bound, upper_bound)

    # Ensure x0 sums to 1.
    if np.isclose(x0.sum(), 0.0):
        x0 = np.repeat(1.0 / n, n)
    else:
        x0 = np.maximum(x0, lower_bound)
        x0 = np.minimum(x0, upper_bound)
        total = x0.sum()
        if total <= 0:
            x0 = np.repeat(1.0 / n, n)
        else:
            x0 = x0 / total

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )

    if not result.success:
        # Conservative fallback: keep current allocation.
        fallback = prev_weights.copy()
        fallback = np.clip(fallback, lower_bound, upper_bound)
        if fallback.sum() <= 0:
            fallback = np.repeat(1.0 / n, n)
        fallback /= fallback.sum()
        return fallback

    w = result.x
    w = np.clip(w, lower_bound, upper_bound)
    w /= w.sum()
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
    if rebalance_starts[-1] != len(index):
        segment_starts = rebalance_starts
    else:
        segment_starts = rebalance_starts[:-1]

    wealth = 1.0
    wealth_path = []
    net_daily_returns = []

    daily_weights = pd.DataFrame(index=index[lookback:], columns=asset_names, dtype=float)
    rebalance_rows = []
    turnover_rows = []

    start_pointer = lookback

    for seg_i, start_idx in enumerate(segment_starts):
        if start_idx < lookback:
            continue

        next_start = (
            segment_starts[seg_i + 1] if seg_i + 1 < len(segment_starts) else len(index)
        )
        if next_start <= start_idx:
            continue

        horizon_days = next_start - start_idx
        estimation_window = returns.iloc[start_idx - lookback : start_idx]
        mu = estimation_window.mean().values * horizon_days
        cov = estimation_window.cov().values * horizon_days

        target_weights = optimise_weights(mu, cov, current_weights, config)

        turnover = float(np.abs(target_weights - current_weights).sum())
        trade_cost = config.transaction_cost_rate * turnover
        wealth *= (1.0 - trade_cost)

        rebalance_date = index[start_idx]
        rebalance_rows.append(
            pd.Series(target_weights, index=asset_names, name=rebalance_date)
        )
        turnover_rows.append(
            {
                "Date": rebalance_date,
                "Turnover": turnover,
                "TradingCostFraction": trade_cost,
            }
        )

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

    wealth_series = pd.Series(
        [x[1] for x in wealth_path],
        index=[x[0] for x in wealth_path],
        name=config.name,
    )
    returns_series = pd.Series(
        [x[1] for x in net_daily_returns],
        index=[x[0] for x in net_daily_returns],
        name=config.name,
    )

    rebalance_weights = pd.DataFrame(rebalance_rows)
    if not rebalance_weights.empty:
        rebalance_weights.index.name = "Date"

    turnover_df = pd.DataFrame(turnover_rows)
    if not turnover_df.empty:
        turnover_df = turnover_df.set_index("Date")

    metrics = {
        "Strategy": config.name,
        "AnnualReturn": annualized_return(returns_series),
        "AnnualVolatility": annualized_vol(returns_series),
        "Sharpe": sharpe_ratio(returns_series),
        "MaxDrawdown": max_drawdown(wealth_series),
        "AverageTurnover": float(turnover_df["Turnover"].mean()) if not turnover_df.empty else np.nan,
        "TotalTurnover": float(turnover_df["Turnover"].sum()) if not turnover_df.empty else np.nan,
        "TotalTradingCostFraction": float(turnover_df["TradingCostFraction"].sum())
        if not turnover_df.empty
        else 0.0,
        "NumRebalances": int(len(turnover_df)),
    }

    return {
        "wealth": wealth_series,
        "daily_returns": returns_series,
        "daily_weights": daily_weights,
        "rebalance_weights": rebalance_weights,
        "turnover": turnover_df,
        "metrics": metrics,
    }


def compare_weight_instability(
    baseline_weights: pd.DataFrame,
    constrained_weights: pd.DataFrame,
) -> pd.DataFrame:
    common_index = baseline_weights.index.intersection(constrained_weights.index)
    if len(common_index) == 0:
        return pd.DataFrame()

    base = baseline_weights.loc[common_index]
    cons = constrained_weights.loc[common_index]

    diff = (cons - base).abs()
    out = pd.DataFrame(
        {
            "MeanAbsWeightDifference": diff.mean(axis=1),
            "MaxAbsWeightDifference": diff.max(axis=1),
        },
        index=common_index,
    )
    return out


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


def save_outputs(
    results: Dict[str, Dict[str, pd.DataFrame | pd.Series | float]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary = pd.DataFrame([results[name]["metrics"] for name in results])
    summary = summary.sort_values("Sharpe", ascending=False)
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    wealth_df = pd.concat(
        [results[name]["wealth"] for name in results], axis=1
    ).dropna(how="all")
    wealth_df.to_csv(os.path.join(output_dir, "cumulative_wealth.csv"))

    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        res["daily_weights"].to_csv(os.path.join(output_dir, f"daily_weights_{safe_name}.csv"))
        if isinstance(res["rebalance_weights"], pd.DataFrame) and not res["rebalance_weights"].empty:
            res["rebalance_weights"].to_csv(
                os.path.join(output_dir, f"rebalance_weights_{safe_name}.csv")
            )
        if isinstance(res["turnover"], pd.DataFrame) and not res["turnover"].empty:
            res["turnover"].to_csv(os.path.join(output_dir, f"turnover_{safe_name}.csv"))

    plot_cumulative_wealth(
        {name: results[name]["wealth"] for name in results},
        os.path.join(output_dir, "cumulative_wealth.png"),
    )
    plot_turnover(
        {name: results[name]["turnover"] for name in results},
        os.path.join(output_dir, "turnover_by_rebalance.png"),
    )


def run_project(prices: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    returns = compute_returns(prices)

    strategies = [
        StrategyConfig(
            name="Theoretical MV (frictionless monthly)",
            rebalance_frequency="monthly",
            risk_aversion=4.0,
            long_only=True,
            transaction_cost_rate=0.0,
            transaction_cost_penalty=0.0,
            turnover_limit=None,
        ),
        StrategyConfig(
            name="Theoretical MV + realised costs",
            rebalance_frequency="monthly",
            risk_aversion=4.0,
            long_only=True,
            transaction_cost_rate=0.0020,     # 20 bps per dollar traded
            transaction_cost_penalty=0.0,     # optimiser ignores costs
            turnover_limit=None,
        ),
        StrategyConfig(
            name="Implementation-aware MV",
            rebalance_frequency="monthly",
            risk_aversion=4.0,
            long_only=True,
            transaction_cost_rate=0.0020,
            transaction_cost_penalty=0.0100,
            turnover_limit=0.35,
        ),
        StrategyConfig(
            name="Implementation-aware MV (quarterly)",
            rebalance_frequency="quarterly",
            risk_aversion=4.0,
            long_only=True,
            transaction_cost_rate=0.0020,
            transaction_cost_penalty=0.0100,
            turnover_limit=0.35,
        ),
    ]

    results: Dict[str, Dict] = {}
    for config in strategies:
        results[config.name] = backtest_strategy(returns, config=config, lookback=252)

    save_outputs(results, output_dir)

    summary = pd.DataFrame([results[name]["metrics"] for name in results]).sort_values(
        "Sharpe", ascending=False
    )

    # Save an extra diagnostic comparison: monthly theoretical vs monthly constrained weights.
    monthly_base = results["Theoretical MV (frictionless monthly)"]["rebalance_weights"]
    monthly_impl = results["Implementation-aware MV"]["rebalance_weights"]
    instability = compare_weight_instability(monthly_base, monthly_impl)
    if not instability.empty:
        instability.to_csv(os.path.join(output_dir, "weight_difference_monthly_base_vs_impl.csv"))

    return summary, results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mean-variance midterm project")
    parser.add_argument(
        "--prices_csv",
        type=str,
        default=None,
        help="Path to CSV of prices. If omitted, synthetic data will be generated.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mv_midterm_outputs",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--save_synthetic_prices",
        action="store_true",
        help="Save the internally generated synthetic price data.",
    )
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
