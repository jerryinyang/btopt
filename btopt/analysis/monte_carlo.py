from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..reporter import Reporter
from .stress_test import StressTest


class MonteCarloTest(StressTest):
    def __init__(
        self,
        name: str,
        reporter: Reporter,
        num_simulations: int = 1000,
        confidence_levels: List[float] = [0.95, 0.99],
        simulation_method: str = "resample",
    ):
        super().__init__(name)
        self.reporter = reporter
        self.num_simulations = num_simulations
        self.confidence_levels = confidence_levels
        self.simulation_method = simulation_method
        self.original_returns = None
        self.simulated_returns = None
        self.simulated_equity_curves = None

    def run(self) -> None:
        self._log_start()
        try:
            self.original_returns = self._preprocess_returns()
            self.simulated_returns = self._generate_simulations()
            self.simulated_equity_curves = self._calculate_equity_curves()
            self.results = self._calculate_metrics()
            self._log_end()
        except Exception as e:
            self._log_error(e)
            raise

    def _preprocess_returns(self) -> pd.Series:
        """Preprocess the equity curve to get daily returns."""
        equity_curve = self.reporter.get_portfolio_returns()

        # Forward fill to handle days without trades
        equity_curve = equity_curve.resample("D").last().ffill()

        # Calculate daily returns
        returns = equity_curve.pct_change().dropna()

        return returns

    def _generate_simulations(self) -> np.ndarray:
        """Generate Monte Carlo simulations of returns."""
        if self.simulation_method == "reshuffle":
            return self._monte_carlo_reshuffle()
        elif self.simulation_method == "resample":
            return self._monte_carlo_resample()
        else:
            raise ValueError(
                "Invalid simulation method. Choose 'reshuffle' or 'resample'."
            )

    def _monte_carlo_reshuffle(self) -> np.ndarray:
        """Perform Monte Carlo Reshuffle simulation."""
        return np.array(
            [
                np.random.permutation(self.original_returns)
                for _ in range(self.num_simulations)
            ]
        ).T

    def _monte_carlo_resample(self) -> np.ndarray:
        """Perform Monte Carlo Resample simulation."""
        return np.random.choice(
            self.original_returns,
            size=(len(self.original_returns), self.num_simulations),
            replace=True,
        )

    def _calculate_equity_curves(self) -> np.ndarray:
        """Calculate equity curves from simulated returns."""
        return (1 + self.simulated_returns).cumprod(axis=0)

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for simulated returns."""
        total_returns = self.simulated_equity_curves[-1, :] - 1
        max_drawdowns = np.array(
            [
                self._calculate_max_drawdown(curve)
                for curve in self.simulated_equity_curves.T
            ]
        )

        metrics = {
            "total_return": self._calculate_confidence_levels(total_returns),
            "cagr": self._calculate_confidence_levels(
                self._calculate_cagr(self.simulated_equity_curves)
            ),
            "max_drawdown": self._calculate_confidence_levels(max_drawdowns),
            "sharpe_ratio": self._calculate_confidence_levels(
                self._calculate_sharpe_ratio(self.simulated_returns)
            ),
            "sortino_ratio": self._calculate_confidence_levels(
                self._calculate_sortino_ratio(self.simulated_returns)
            ),
        }

        return metrics

    def _calculate_confidence_levels(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate confidence levels for a given metric."""
        percentiles = [100 * (1 - level) for level in self.confidence_levels] + [50]
        results = np.percentile(data, percentiles)
        return {
            f"{level:.0%}": result
            for level, result in zip(self.confidence_levels + [0.5], results)
        }

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown for a single equity curve."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    def _calculate_cagr(self, equity_curves: np.ndarray) -> np.ndarray:
        """Calculate Compound Annual Growth Rate for all simulations."""
        total_return = equity_curves[-1, :] / equity_curves[0, :]
        years = len(self.original_returns) / 252  # Assuming 252 trading days per year
        return (total_return ** (1 / years)) - 1

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> np.ndarray:
        """Calculate Sharpe Ratio for all simulations."""
        excess_returns = returns.mean(axis=0) - risk_free_rate / 252
        return (excess_returns / returns.std(axis=0)) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        target_return: float = 0,
    ) -> np.ndarray:
        """Calculate Sortino Ratio for all simulations."""
        excess_returns = returns.mean(axis=0) - risk_free_rate / 252
        downside_returns = np.minimum(returns - target_return, 0)
        downside_deviation = np.sqrt(np.mean(downside_returns**2, axis=0)) * np.sqrt(
            252
        )
        return excess_returns / downside_deviation

    def visualize(self) -> None:
        self._plot_equity_curves()
        self._plot_return_distribution()
        self._plot_drawdown_distribution()
        self._plot_metric_distributions()

    def _plot_equity_curves(self) -> None:
        """Plot equity curves for all simulations."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.simulated_equity_curves, alpha=0.1, color="blue")
        sns.lineplot(
            data=(1 + self.original_returns).cumprod(), color="red", linewidth=2
        )
        plt.title(
            f"{self.name}: Monte Carlo Simulated Equity Curves ({self.simulation_method})"
        )
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value")
        plt.legend(["Simulations", "Original"])
        plt.savefig(f"{self.name}_equity_curves_{self.simulation_method}.png")
        plt.close()

    def _plot_return_distribution(self) -> None:
        """Plot the distribution of total returns."""
        total_returns = self.simulated_equity_curves[-1, :] - 1
        plt.figure(figsize=(10, 6))
        sns.histplot(total_returns, kde=True)
        plt.axvline(np.median(total_returns), color="r", linestyle="--")
        plt.title(
            f"{self.name}: Distribution of Total Returns ({self.simulation_method})"
        )
        plt.xlabel("Total Return")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.name}_return_distribution_{self.simulation_method}.png")
        plt.close()

    def _plot_drawdown_distribution(self) -> None:
        """Plot the distribution of maximum drawdowns."""
        max_drawdowns = np.array(
            [
                self._calculate_max_drawdown(curve)
                for curve in self.simulated_equity_curves.T
            ]
        )
        plt.figure(figsize=(10, 6))
        sns.histplot(max_drawdowns, kde=True)
        plt.axvline(np.median(max_drawdowns), color="r", linestyle="--")
        plt.title(
            f"{self.name}: Distribution of Maximum Drawdowns ({self.simulation_method})"
        )
        plt.xlabel("Maximum Drawdown")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.name}_drawdown_distribution_{self.simulation_method}.png")
        plt.close()

    def _plot_metric_distributions(self) -> None:
        """Plot distributions of various metrics."""
        metrics = [
            "total_return",
            "cagr",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
        ]
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(10, 6 * len(metrics)), sharex=False
        )
        fig.suptitle(
            f"{self.name}: Distributions of Metrics ({self.simulation_method})"
        )

        for ax, metric in zip(axes, metrics):
            data = [sim[metric] for sim in self.results]
            sns.boxplot(data=data, ax=ax)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Confidence Level")
            ax.set_ylabel("Value")

        plt.tight_layout()
        plt.savefig(f"{self.name}_metric_distributions_{self.simulation_method}.png")
        plt.close()


# Example usage
if __name__ == "__main__":
    # This is just to demonstrate how this class might be used
    class DummyReporter:
        def get_portfolio_returns(self):
            # Generate some dummy equity curve data
            dates = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
            equity = 100000 * (1 + np.random.normal(0.0002, 0.01, len(dates))).cumprod()
            return pd.Series(equity, index=dates)

    # Create a dummy reporter
    dummy_reporter = DummyReporter()

    # Create and run Monte Carlo tests with both methods
    for method in ["reshuffle", "resample"]:
        mc_test = MonteCarloTest(
            f"Example Monte Carlo ({method})",
            dummy_reporter,
            num_simulations=500,
            simulation_method=method,
        )
        mc_test.run()
        mc_test.visualize()

        # Print results
        print(f"\nResults for {method} method:")
        print(mc_test.get_results())  # Print results
        print(f"\nResults for {method} method:")
        print(mc_test.get_results())
