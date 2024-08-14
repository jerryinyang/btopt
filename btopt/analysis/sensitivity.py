import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..engine import Engine
from ..strategy import Strategy
from .stress_test import StressTest


class SensitivityTest(StressTest):
    def __init__(
        self,
        name: str,
        engine: Engine,
        strategy: Strategy,
        parameter_ranges: Dict[str, List[Any]],
        metrics: List[str],
    ):
        super().__init__(name)
        self.engine = engine
        self.strategy = strategy
        self.parameter_ranges = parameter_ranges
        self.metrics = metrics
        self.results = pd.DataFrame()

    def run(self) -> None:
        self._log_start()
        try:
            parameter_combinations = self._generate_parameter_combinations()
            self.results = self._run_backtests(parameter_combinations)
            self._log_end()
        except Exception as e:
            self._log_error(e)
            raise

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        keys, values = zip(*self.parameter_ranges.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _run_backtests(
        self, parameter_combinations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Run backtests for all parameter combinations."""
        results = []
        total_combinations = len(parameter_combinations)

        for i, params in enumerate(parameter_combinations, 1):
            self._log_progress(i, total_combinations)

            # Update strategy parameters
            self.strategy.set_parameters(params)

            # Run backtest
            reporter = self.engine.run()

            # Get performance metrics
            metrics = reporter.calculate_performance_metrics()

            # Combine parameters and metrics
            result = {**params, **{metric: metrics[metric] for metric in self.metrics}}
            results.append(result)

        return pd.DataFrame(results)

    def _log_progress(self, current: int, total: int) -> None:
        """Log the progress of backtests."""
        progress = (current / total) * 100
        self._log_info(f"Progress: {progress:.2f}% ({current}/{total})")

    def visualize(self) -> None:
        """Visualize the results of the sensitivity analysis."""
        self._plot_heatmaps()
        self._plot_3d_surfaces()

    def _plot_heatmaps(self) -> None:
        """Plot heatmaps for each metric against pairs of parameters."""
        param_pairs = list(itertools.combinations(self.parameter_ranges.keys(), 2))
        metrics = self.results.columns.drop(list(self.parameter_ranges.keys()))

        for metric in metrics:
            fig = make_subplots(
                rows=len(param_pairs),
                cols=1,
                subplot_titles=[f"{p1} vs {p2}" for p1, p2 in param_pairs],
            )

            for i, (param1, param2) in enumerate(param_pairs, 1):
                pivot = self.results.pivot(index=param1, columns=param2, values=metric)

                heatmap = go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorbar=dict(title=metric, y=0.5, len=0.75 / len(param_pairs)),
                )

                fig.add_trace(heatmap, row=i, col=1)

            fig.update_layout(
                height=400 * len(param_pairs),
                width=800,
                title_text=f"Sensitivity Analysis: {metric}",
            )
            fig.write_html(f"{self.name}_heatmap_{metric}.html")

    def _plot_3d_surfaces(self) -> None:
        """Plot 3D surfaces for each metric against pairs of parameters."""
        param_pairs = list(itertools.combinations(self.parameter_ranges.keys(), 2))
        metrics = self.results.columns.drop(list(self.parameter_ranges.keys()))

        for metric in metrics:
            fig = make_subplots(
                rows=1,
                cols=len(param_pairs),
                specs=[[{"type": "surface"}] * len(param_pairs)],
                subplot_titles=[f"{p1} vs {p2}" for p1, p2 in param_pairs],
            )

            for i, (param1, param2) in enumerate(param_pairs, 1):
                pivot = self.results.pivot(index=param1, columns=param2, values=metric)

                surface = go.Surface(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorbar=dict(title=metric, x=0.95, len=0.75),
                )

                fig.add_trace(surface, row=1, col=i)

            fig.update_layout(
                height=600,
                width=800 * len(param_pairs),
                title_text=f"Sensitivity Analysis: {metric}",
            )
            fig.write_html(f"{self.name}_3d_surface_{metric}.html")

    def get_results(self) -> Dict[str, Any]:
        """Return the results of the sensitivity analysis."""
        return {
            "parameter_ranges": self.parameter_ranges,
            "metrics": self.metrics,
            "results": self.results.to_dict(orient="records"),
        }


# Example usage
if __name__ == "__main__":
    # This is just to demonstrate how this class might be used
    class DummyEngine:
        def run(self):
            return DummyReporter()

    class DummyStrategy:
        def set_parameters(self, params):
            pass

    class DummyReporter:
        def calculate_performance_metrics(self):
            return {
                "total_return": np.random.uniform(0.1, 0.5),
                "sharpe_ratio": np.random.uniform(0.5, 2.0),
                "max_drawdown": np.random.uniform(0.1, 0.3),
            }

    # Create dummy objects
    dummy_engine = DummyEngine()
    dummy_strategy = DummyStrategy()

    # Define parameter ranges and metrics
    parameter_ranges = {
        "lookback_period": [10, 20, 30],
        "volatility_window": [5, 10, 15],
        "risk_factor": [0.1, 0.2, 0.3],
    }
    metrics = ["total_return", "sharpe_ratio", "max_drawdown"]

    # Create and run Sensitivity Test
    sensitivity_test = SensitivityTest(
        "Example Sensitivity", dummy_engine, dummy_strategy, parameter_ranges, metrics
    )
    sensitivity_test.run()
    sensitivity_test.visualize()

    # Print results
    print(sensitivity_test.get_results())
    # Print results
    print(sensitivity_test.get_results())
