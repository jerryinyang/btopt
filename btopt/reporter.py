import asyncio
from functools import lru_cache
from typing import Any, Dict, Optional

import pandas as pd
import quantstats_lumi as qs

from .log_config import logger_main
from .types import EngineType, PortfolioType
from .util.decimal import ExtendedDecimal


class Reporter:
    """
    A comprehensive reporting and analysis class for financial trading systems.

    This class leverages the Quantstats library for most metric calculations, reporting,
    and visualizations. It provides an interface to analyze and visualize the performance
    of trading strategies.

    Attributes:
        portfolio (PortfolioType): An instance of the Portfolio class containing trading data.
        engine (EngineType): An instance of the Engine class managing the trading system.
    """

    def __init__(self, portfolio: PortfolioType, engine: EngineType):
        """
        Initialize the Reporter instance.

        Args:
            portfolio (PortfolioType): The Portfolio instance to report on.
            engine (EngineType): The Engine instance managing the trading system.
        """
        self.portfolio = portfolio
        self.engine = engine

    # region Data Retrieval
    def _convert_decimal_to_float(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all ExtendedDecimal columns in a DataFrame to float.

        Args:
            data (pd.DataFrame): The DataFrame to convert.

        Returns:
            pd.DataFrame: The DataFrame with ExtendedDecimal columns converted to float.
        """
        for column in data.select_dtypes(include=[ExtendedDecimal]).columns:
            data[column] = data[column].astype(float)
        return data

    def get_metrics_data(self) -> pd.DataFrame:
        """
        Retrieve the metrics data, converting ExtendedDecimal values to float.

        Returns:
            pd.DataFrame: The metrics DataFrame with ExtendedDecimal values converted to float.
        """
        metrics = self.portfolio.get_metrics_data()
        return self._convert_decimal_to_float(metrics)

    def get_portfolio_returns(self) -> pd.Series:
        """
        Retrieve the portfolio returns series, converting ExtendedDecimal to float.

        Returns:
            pd.Series: A series of portfolio returns indexed by timestamp.
        """
        metrics = self.portfolio.get_metrics_data()
        # Convert ExtendedDecimal to float
        return pd.Series(
            metrics["portfolio_return"].astype(float).values, index=metrics["timestamp"]
        )

    # endregion

    # region Performance Metrics

    @lru_cache(maxsize=None)
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return various performance metrics using Quantstats.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics.
        """
        returns = self.get_portfolio_returns()
        metrics = {
            "total_return": qs.stats.total_return(returns),
            "cagr": qs.stats.cagr(returns),
            "sharpe_ratio": qs.stats.sharpe(returns),
            "sortino_ratio": qs.stats.sortino(returns),
            "max_drawdown": qs.stats.max_drawdown(returns),
            "win_rate": qs.stats.win_rate(returns),
            "profit_factor": qs.stats.profit_factor(returns),
            "volatility": qs.stats.volatility(returns),
            "calmar_ratio": qs.stats.calmar(returns),
            "omega_ratio": qs.stats.omega(returns),
        }

        return metrics

    def get_drawdown_details(self) -> pd.DataFrame:
        """
        Get detailed information about drawdowns.

        Returns:
            pd.DataFrame: A DataFrame containing drawdown details.
        """
        returns = self.get_portfolio_returns()
        return qs.stats.drawdown_details(returns)

    # endregion

    # region Report Generation

    def generate_performance_report(
        self, output: str = "html", filename: Optional[str] = None
    ) -> None:
        """
        Generate a comprehensive performance report using Quantstats.

        Args:
            output (str): The output format ('html' or 'pdf').
            filename (Optional[str]): The filename to save the report. If None, a default name will be used.
        """
        returns = self.get_portfolio_returns()

        if output == "html":
            qs.reports.html(
                returns,
                output=filename or "performance_report.html",
            )
        elif output == "pdf":
            qs.reports.pdf(
                returns,
                output=filename or "performance_report.pdf",
            )
        else:
            logger_main.warning(
                f"Unsupported output format: {output}. Using HTML instead."
            )
            qs.reports.html(
                returns,
                output=filename or "performance_report.html",
            )

    # endregion

    # region Visualization

    def plot_equity_curve(self) -> None:
        """
        Plot the equity curve of the portfolio.
        """
        returns = self.get_portfolio_returns()
        qs.plots.earnings(returns, show=True)

    def plot_drawdown_curve(self) -> None:
        """
        Plot the drawdown curve of the portfolio.
        """
        returns = self.get_portfolio_returns()
        qs.plots.drawdown(returns, show=True)

    def plot_monthly_returns_heatmap(self) -> None:
        """
        Plot a heatmap of monthly returns.
        """
        returns = self.get_portfolio_returns()
        qs.plots.monthly_heatmap(returns, show=True)

    def plot_return_distribution(self) -> None:
        """
        Plot the distribution of returns.
        """
        returns = self.get_portfolio_returns()
        qs.plots.distribution(returns, show=True)

    def plot_rolling_metrics(self, window: int = 60) -> None:
        """
        Plot rolling Sharpe ratio and Sortino ratio.

        Args:
            window (int): The rolling window size in days.
        """
        returns = self.get_portfolio_returns()
        qs.plots.rolling_sharpe(returns, window=window, show=True)
        qs.plots.rolling_sortino(returns, window=window, show=True)

    # endregion

    # region Additional Metrics

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.

        Args:
            confidence_level (float): The confidence level for VaR calculation.

        Returns:
            float: The calculated Value at Risk.
        """
        returns = self.get_portfolio_returns()
        return qs.stats.value_at_risk(returns, confidence=confidence_level)

    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) for the portfolio.

        Args:
            confidence_level (float): The confidence level for CVaR calculation.

        Returns:
            float: The calculated Conditional Value at Risk.
        """
        returns = self.get_portfolio_returns()
        return qs.stats.conditional_value_at_risk(returns, confidence=confidence_level)

    # endregion

    # region Utility Methods

    async def generate_report_async(
        self, output: str = "html", filename: Optional[str] = None
    ) -> None:
        """
        Asynchronously generate a performance report.

        Args:
            output (str): The output format ('html' or 'pdf').
            filename (Optional[str]): The filename to save the report. If None, a default name will be used.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.generate_performance_report, output, filename
        )

    def clear_cache(self) -> None:
        """
        Clear the LRU cache for performance metrics calculation.
        """
        self.calculate_performance_metrics.cache_clear()
        logger_main.info("Performance metrics cache cleared")

    # endregion
