import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .log_config import logger_main
from .types import EngineType, PortfolioType


class Reporter:
    """
    A comprehensive reporting and analysis class for financial trading systems.

    This class serves as a central hub for all performance evaluation, reporting, and visualization
    tasks related to trading strategies. It is designed to work in conjunction with a Portfolio
    instance, which contains the trading data, and an Engine instance, which manages the overall
    trading system.

    The Reporter class offers a wide range of functionalities, including:

    1. Performance Metric Calculation:
       - Computes essential trading metrics such as total return, Sharpe ratio, Sortino ratio,
         maximum drawdown, win rate, profit factor, and more.
       - Implements advanced risk metrics like Value at Risk (VaR) and Expected Shortfall (ES).
       - Calculates time-weighted and money-weighted returns for accurate performance assessment.

    2. Report Generation:
       - Creates comprehensive performance summaries that encapsulate the strategy's effectiveness.
       - Generates detailed trade history reports for granular analysis of individual trades.
       - Produces risk reports to assess the strategy's risk profile.
       - Compiles position reports to provide snapshots of current holdings.

    3. Data Visualization:
       - Plots equity curves to visualize the growth of the portfolio over time.
       - Creates drawdown curves to illustrate periods and magnitudes of losses.
       - Generates return distribution charts for statistical analysis of returns.
       - Visualizes trade history to identify patterns in winning and losing trades.
       - Plots position sizes over time for individual securities.

    4. Advanced Analysis:
       - Performs Monte Carlo simulations to project potential future outcomes.
       - Conducts sensitivity analysis to understand the impact of various parameters on performance.
       - Implements performance attribution to break down returns by different factors.
       - Compares performance across different time periods or against other strategies.

    5. Data Export:
       - Provides functionality to export reports and data to various formats including CSV, Excel, and PDF.

    6. Real-time Reporting:
       - Offers methods for updating and reporting metrics in real-time for live trading scenarios.

    The class utilizes a caching mechanism to store results of computationally expensive calculations,
    improving performance for repeated queries. This cache can be cleared as needed to ensure
    fresh calculations.

    Attributes:
        portfolio (Portfolio): An instance of the Portfolio class containing all trading data,
                               including trades, positions, and equity history. This attribute
                               provides the raw data necessary for all calculations and analyses.

        engine (Engine): An instance of the Engine class that manages the overall trading system.
                         This attribute allows the Reporter to access additional system-level
                         information and functionality when needed.

        cache (Dict[str, Any]): A dictionary serving as a cache to store the results of
                                computationally intensive calculations. This improves performance
                                by avoiding unnecessary recalculations of unchanged data.

    The Reporter class is designed to be flexible and extensible, allowing for easy addition of
    new metrics, reports, or visualization types as needed. It aims to provide a comprehensive
    toolkit for strategy evaluation, risk management, and performance reporting in both backtesting
    and live trading environments.
    """

    def __init__(self, portfolio: PortfolioType, engine: EngineType):
        """
        Initialize the Reporter instance.

        Args:
            portfolio (Portfolio): The Portfolio instance to report on.
            engine (Engine): The Engine instance managing the trading system.
        """
        self.portfolio = portfolio
        self.engine = engine
        self.cache: Dict[str, Any] = {}

    # region Performance Metrics

    def calculate_total_return(self) -> Decimal:
        """
        Calculate the total return of the portfolio.

        Returns:
            Decimal: The total return as a percentage.
        """
        return self._cached_calculation(
            "total_return", self._calculate_total_return_impl
        )

    def _calculate_total_return_impl(self) -> Decimal:
        """
        Implementation of total return calculation.

        Returns:
            Decimal: The total return as a percentage.
        """
        initial_equity = self.portfolio.initial_capital
        final_equity = self.portfolio.calculate_equity()
        return (final_equity - initial_equity) / initial_equity * Decimal("100")

    def calculate_sharpe_ratio(
        self, risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal:
        """
        Calculate the Sharpe ratio of the portfolio.

        Args:
            risk_free_rate (Decimal): The risk-free rate, default is 2%.

        Returns:
            Decimal: The calculated Sharpe ratio.
        """
        return self._cached_calculation(
            "sharpe_ratio", self._calculate_sharpe_ratio_impl, risk_free_rate
        )

    def _calculate_sharpe_ratio_impl(self, risk_free_rate: Decimal) -> Decimal:
        """
        Implementation of Sharpe ratio calculation.

        Args:
            risk_free_rate (Decimal): The risk-free rate.

        Returns:
            Decimal: The calculated Sharpe ratio.
        """
        returns = self.portfolio.metrics["equity"].pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return Decimal(str(sharpe_ratio))

    def calculate_sortino_ratio(
        self,
        risk_free_rate: Decimal = Decimal("0.02"),
        target_return: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate the Sortino ratio of the portfolio.

        Args:
            risk_free_rate (Decimal): The risk-free rate, default is 2%.
            target_return (Decimal): The target return, default is 0%.

        Returns:
            Decimal: The calculated Sortino ratio.
        """
        return self._cached_calculation(
            "sortino_ratio",
            self._calculate_sortino_ratio_impl,
            risk_free_rate,
            target_return,
        )

    def _calculate_sortino_ratio_impl(
        self, risk_free_rate: Decimal, target_return: Decimal
    ) -> Decimal:
        """
        Implementation of Sortino ratio calculation.

        Args:
            risk_free_rate (Decimal): The risk-free rate.
            target_return (Decimal): The target return.

        Returns:
            Decimal: The calculated Sortino ratio.
        """
        returns = self.portfolio.metrics["equity"].pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < target_return]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return Decimal(str(sortino_ratio))

    def calculate_max_drawdown(self) -> Decimal:
        """
        Calculate the maximum drawdown of the portfolio.

        Returns:
            Decimal: The maximum drawdown as a percentage.
        """
        return self._cached_calculation(
            "max_drawdown", self._calculate_max_drawdown_impl
        )

    def _calculate_max_drawdown_impl(self) -> Decimal:
        """
        Implementation of maximum drawdown calculation.

        Returns:
            Decimal: The maximum drawdown as a percentage.
        """
        equity_curve = self.portfolio.metrics["equity"]
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        return Decimal(str(abs(max_drawdown) * 100))

    def calculate_win_rate(self) -> Decimal:
        """
        Calculate the win rate of closed trades.

        Returns:
            Decimal: The win rate as a percentage.
        """
        return self._cached_calculation("win_rate", self._calculate_win_rate_impl)

    def _calculate_win_rate_impl(self) -> Decimal:
        """
        Implementation of win rate calculation.

        Returns:
            Decimal: The win rate as a percentage.
        """
        closed_trades = self.portfolio.closed_trades
        if not closed_trades:
            return Decimal("0")

        winning_trades = sum(1 for trade in closed_trades if trade.metrics.pnl > 0)
        return Decimal(winning_trades) / Decimal(len(closed_trades)) * 100

    def calculate_profit_factor(self) -> Decimal:
        """
        Calculate the profit factor of the portfolio.

        Returns:
            Decimal: The profit factor.
        """
        return self._cached_calculation(
            "profit_factor", self._calculate_profit_factor_impl
        )

    def _calculate_profit_factor_impl(self) -> Decimal:
        """
        Implementation of profit factor calculation.

        Returns:
            Decimal: The profit factor.
        """
        closed_trades = self.portfolio.closed_trades
        total_profit = sum(
            trade.metrics.pnl for trade in closed_trades if trade.metrics.pnl > 0
        )
        total_loss = abs(
            sum(trade.metrics.pnl for trade in closed_trades if trade.metrics.pnl < 0)
        )

        if total_loss == 0:
            return Decimal("inf") if total_profit > 0 else Decimal("0")

        return total_profit / total_loss

    def calculate_calmar_ratio(self, years: int = 3) -> Decimal:
        """
        Calculate the Calmar ratio of the portfolio.

        Args:
            years (int): The number of years to consider for the calculation.

        Returns:
            Decimal: The calculated Calmar ratio.
        """
        return self._cached_calculation(
            "calmar_ratio", self._calculate_calmar_ratio_impl, years
        )

    def _calculate_calmar_ratio_impl(self, years: int) -> Decimal:
        """
        Implementation of Calmar ratio calculation.

        Args:
            years (int): The number of years to consider for the calculation.

        Returns:
            Decimal: The calculated Calmar ratio.
        """
        total_return = self.calculate_total_return()
        max_drawdown = self.calculate_max_drawdown()

        if max_drawdown == Decimal("0"):
            return Decimal("inf") if total_return > 0 else Decimal("0")

        annual_return = (1 + total_return / 100) ** (1 / years) - 1
        return annual_return / (max_drawdown / 100)

    def calculate_value_at_risk(self, confidence_level: float = 0.95) -> Decimal:
        """
        Calculate the Value at Risk (VaR) of the portfolio.

        Args:
            confidence_level (float): The confidence level for VaR calculation.

        Returns:
            Decimal: The calculated Value at Risk.
        """
        return self._cached_calculation(
            "value_at_risk", self._calculate_value_at_risk_impl, confidence_level
        )

    def _calculate_value_at_risk_impl(self, confidence_level: float) -> Decimal:
        """
        Implementation of Value at Risk calculation.

        Args:
            confidence_level (float): The confidence level for VaR calculation.

        Returns:
            Decimal: The calculated Value at Risk.
        """
        returns = self.portfolio.metrics["equity"].pct_change().dropna()
        var = returns.quantile(1 - confidence_level)
        return Decimal(str(abs(var)))

    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> Decimal:
        """
        Calculate the Expected Shortfall (ES) of the portfolio.

        Args:
            confidence_level (float): The confidence level for ES calculation.

        Returns:
            Decimal: The calculated Expected Shortfall.
        """
        return self._cached_calculation(
            "expected_shortfall",
            self._calculate_expected_shortfall_impl,
            confidence_level,
        )

    def _calculate_expected_shortfall_impl(self, confidence_level: float) -> Decimal:
        """
        Implementation of Expected Shortfall calculation.

        Args:
            confidence_level (float): The confidence level for ES calculation.

        Returns:
            Decimal: The calculated Expected Shortfall.
        """
        returns = self.portfolio.metrics["equity"].pct_change().dropna()
        var = returns.quantile(1 - confidence_level)
        es = returns[returns <= var].mean()
        return Decimal(str(abs(es)))

    # endregion

    # region Report Generation

    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary of the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing various performance metrics.
        """
        return {
            "total_return": self.calculate_total_return(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "max_drawdown": self.calculate_max_drawdown(),
            "win_rate": self.calculate_win_rate(),
            "profit_factor": self.calculate_profit_factor(),
            "calmar_ratio": self.calculate_calmar_ratio(),
            "value_at_risk": self.calculate_value_at_risk(),
            "expected_shortfall": self.calculate_expected_shortfall(),
        }

    def generate_trade_history_report(self) -> pd.DataFrame:
        """
        Generate a detailed trade history report.

        Returns:
            pd.DataFrame: A DataFrame containing the trade history.
        """
        trades = self.portfolio.closed_trades + self.portfolio.get_open_trades()
        trade_data = []
        for trade in trades:
            trade_data.append(
                {
                    "id": trade.id,
                    "symbol": trade.ticker,
                    "entry_date": trade.entry_timestamp,
                    "exit_date": trade.exit_timestamp,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "size": trade.initial_size,
                    "pnl": trade.metrics.pnl,
                    "return": trade.metrics.pnl_percent,
                    "status": trade.status.name,
                }
            )
        return pd.DataFrame(trade_data)

    def generate_position_report(self) -> pd.DataFrame:
        """
        Generate a report of current positions.

        Returns:
            pd.DataFrame: A DataFrame containing current position information.
        """
        positions = self.portfolio.positions
        position_data = []
        for symbol, size in positions.items():
            current_price = self.engine.get_current_price(symbol)
            avg_entry_price = self.portfolio.avg_entry_prices.get(symbol, Decimal("0"))
            unrealized_pnl = (current_price - avg_entry_price) * size
            position_data.append(
                {
                    "symbol": symbol,
                    "size": size,
                    "avg_entry_price": avg_entry_price,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                }
            )
        return pd.DataFrame(position_data)

    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate a risk report for the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing various risk metrics.
        """
        return {
            "value_at_risk": self.calculate_value_at_risk(),
            "expected_shortfall": self.calculate_expected_shortfall(),
            "max_drawdown": self.calculate_max_drawdown(),
            "beta": self.calculate_beta(),
            "correlation_with_market": self.analyze_correlation_with_market(),
        }

    # endregion

    # region Visualization

    def plot_equity_curve(self) -> None:
        """
        Plot the equity curve of the portfolio.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="timestamp", y="equity", data=self.portfolio.metrics)
        plt.title("Portfolio Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_drawdown_curve(self) -> None:
        """
        Plot the drawdown curve of the portfolio.
        """
        equity_curve = self.portfolio.metrics["equity"]
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.portfolio.metrics["timestamp"], y=drawdown)
        plt.title("Portfolio Drawdown Curve")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_return_distribution(self) -> None:
        """
        Plot the distribution of returns.
        """
        returns = self.portfolio.metrics["equity"].pct_change().dropna()

        plt.figure(figsize=(12, 6))
        sns.histplot(returns, kde=True)
        plt.title("Distribution of Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_trade_history(self) -> None:
        """
        Plot the trade history showing profits and losses.
        """
        trade_history = self.generate_trade_history_report()

        plt.figure(figsize=(12, 6))
        sns.scatterplot(x="exit_date", y="pnl", hue="symbol", data=trade_history)
        plt.title("Trade History")
        plt.xlabel("Exit Date")
        plt.ylabel("Profit/Loss")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_position_over_time(self, symbol: str) -> None:
        """
        Plot the position size over time for a specific symbol.

        Args:
            symbol (str): The symbol to plot the position for.
        """
        position_history = self.portfolio.get_position_history(symbol)

        plt.figure(figsize=(12, 6))
        sns.lineplot(x="timestamp", y="size", data=position_history)
        plt.title(f"Position Size Over Time for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Position Size")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # endregion

    # region Export Functions

    def export_to_csv(self, data: pd.DataFrame, filename: str) -> None:
        """
        Export data to a CSV file.

        Args:
            data (pd.DataFrame): The data to export.
            filename (str): The name of the file to save the data to.
        """
        data.to_csv(filename, index=False)
        logger_main.info(f"Data exported to {filename}")

    def export_to_excel(self, data: Dict[str, pd.DataFrame], filename: str) -> None:
        """
        Export data to an Excel file with multiple sheets.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary of DataFrames to export, where keys are sheet names.
            filename (str): The name of the file to save the data to.
        """
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger_main.info(f"Data exported to {filename}")

    def export_to_pdf(self, filename: str) -> None:
        """
        Export a comprehensive report to a PDF file.

        Args:
            filename (str): The name of the file to save the report to.
        """
        # Note: This is a placeholder. Actual PDF generation would require additional libraries.
        logger_main.info(
            "PDF export not implemented. Consider using a library like ReportLab for PDF generation."
        )

    # endregion

    # region Utility Methods

    def _cached_calculation(
        self, metric_name: str, calculation_func: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Perform a cached calculation, storing the result for future use.

        Args:
            metric_name (str): The name of the metric to calculate.
            calculation_func (Callable): The function to perform the calculation.
            *args: Positional arguments to pass to the calculation function.
            **kwargs: Keyword arguments to pass to the calculation function.

        Returns:
            Any: The result of the calculation.
        """
        if metric_name not in self.cache:
            self.cache[metric_name] = calculation_func(*args, **kwargs)
        return self.cache[metric_name]

    def clear_cache(self) -> None:
        """
        Clear the calculation cache.
        """
        self.cache.clear()
        logger_main.info("Calculation cache cleared")

    # endregion

    # region Advanced Analysis

    def generate_comprehensive_report(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report combining multiple metrics and visualizations.

        Args:
            start_date (Optional[datetime]): The start date for the report period.
            end_date (Optional[datetime]): The end date for the report period.

        Returns:
            Dict[str, Any]: A dictionary containing the comprehensive report data.
        """
        report = {
            "performance_summary": self.generate_performance_summary(),
            "trade_history": self.generate_trade_history_report(),
            "position_report": self.generate_position_report(),
            "risk_report": self.generate_risk_report(),
        }

        # Generate plots
        self.plot_equity_curve()
        self.plot_drawdown_curve()
        self.plot_return_distribution()
        self.plot_trade_history()

        # Add any period-specific analysis here if start_date and end_date are provided

        return report

    async def generate_report_async(self) -> Dict[str, Any]:
        """
        Asynchronously generate a comprehensive report.

        Returns:
            Dict[str, Any]: A dictionary containing the comprehensive report data.
        """
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.generate_comprehensive_report)
        return report

    def compare_performance_periods(
        self, period1: Tuple[datetime, datetime], period2: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Compare performance between two time periods.

        Args:
            period1 (Tuple[datetime, datetime]): The start and end dates of the first period.
            period2 (Tuple[datetime, datetime]): The start and end dates of the second period.

        Returns:
            Dict[str, Any]: A dictionary containing the comparison results.
        """
        # Implement period comparison logic here
        # This is a placeholder implementation
        return {
            "period1": self.generate_performance_summary(),
            "period2": self.generate_performance_summary(),
        }

    def compare_strategies(self, other_strategy: "Reporter") -> Dict[str, Any]:
        """
        Compare performance with another strategy.

        Args:
            other_strategy (Reporter): The Reporter instance of another strategy to compare with.

        Returns:
            Dict[str, Any]: A dictionary containing the comparison results.
        """
        # Implement strategy comparison logic here
        # This is a placeholder implementation
        return {
            "this_strategy": self.generate_performance_summary(),
            "other_strategy": other_strategy.generate_performance_summary(),
        }

    def update_real_time_metrics(self) -> Dict[str, Any]:
        """
        Update and return real-time performance metrics.

        Returns:
            Dict[str, Any]: A dictionary containing updated real-time metrics.
        """
        # Implement real-time metric updates here
        # This is a placeholder implementation
        self.clear_cache()  # Clear cache to ensure fresh calculations
        return self.generate_performance_summary()

    def analyze_by_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """
        Analyze performance grouped by a specific timeframe.

        Args:
            timeframe (str): The timeframe to group by (e.g., 'daily', 'weekly', 'monthly').

        Returns:
            Dict[str, Any]: A dictionary containing the timeframe-based analysis.
        """
        # Implement timeframe-based analysis here
        # This is a placeholder implementation
        return {"timeframe_analysis": "Not implemented"}

    def analyze_by_day_of_week(self) -> Dict[str, Any]:
        """
        Analyze performance grouped by day of the week.

        Returns:
            Dict[str, Any]: A dictionary containing the day-of-week analysis.
        """
        # Implement day-of-week analysis here
        # This is a placeholder implementation
        return {"day_of_week_analysis": "Not implemented"}

    def analyze_by_month(self) -> Dict[str, Any]:
        """
        Analyze performance grouped by month.

        Returns:
            Dict[str, Any]: A dictionary containing the month-based analysis.
        """
        # Implement month-based analysis here
        # This is a placeholder implementation
        return {"month_analysis": "Not implemented"}

    def calculate_risk_reward_ratio(self) -> Decimal:
        """
        Calculate the overall risk-reward ratio of the strategy.

        Returns:
            Decimal: The calculated risk-reward ratio.
        """
        avg_win = self._calculate_average_win()
        avg_loss = self._calculate_average_loss()
        return avg_win / abs(avg_loss) if avg_loss != 0 else Decimal("inf")

    def _calculate_average_win(self) -> Decimal:
        """
        Calculate the average winning trade amount.

        Returns:
            Decimal: The average winning trade amount.
        """
        winning_trades = [
            trade for trade in self.portfolio.closed_trades if trade.metrics.pnl > 0
        ]
        if not winning_trades:
            return Decimal("0")
        return sum(trade.metrics.pnl for trade in winning_trades) / len(winning_trades)

    def _calculate_average_loss(self) -> Decimal:
        """
        Calculate the average losing trade amount.

        Returns:
            Decimal: The average losing trade amount.
        """
        losing_trades = [
            trade for trade in self.portfolio.closed_trades if trade.metrics.pnl < 0
        ]
        if not losing_trades:
            return Decimal("0")
        return sum(trade.metrics.pnl for trade in losing_trades) / len(losing_trades)

    def analyze_correlation_with_market(self, market_returns: pd.Series) -> float:
        """
        Analyze the correlation of the strategy's returns with market returns.

        Args:
            market_returns (pd.Series): A series of market returns to compare against.

        Returns:
            float: The correlation coefficient between strategy returns and market returns.
        """
        strategy_returns = self.portfolio.metrics["equity"].pct_change().dropna()
        return strategy_returns.corr(market_returns)

    def analyze_parameter_sensitivity(
        self, parameter: str, values: List[Any]
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Analyze the sensitivity of the strategy to a specific parameter.

        Args:
            parameter (str): The name of the parameter to analyze.
            values (List[Any]): A list of values to test for the parameter.

        Returns:
            Dict[Any, Dict[str, Any]]: A dictionary mapping parameter values to performance metrics.
        """
        # This is a placeholder implementation. In a real scenario, you would need to re-run
        # the strategy with different parameter values, which is beyond the scope of this class.
        return {value: self.generate_performance_summary() for value in values}

    def generate_monte_carlo_simulations(
        self, num_simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Generate Monte Carlo simulations based on the strategy's historical returns.

        Args:
            num_simulations (int): The number of simulations to run.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the Monte Carlo simulations.
        """
        returns = self.portfolio.metrics["equity"].pct_change().dropna()
        simulations = []
        for _ in range(num_simulations):
            sim_returns = np.random.choice(returns, size=len(returns), replace=True)
            sim_equity = (1 + sim_returns).cumprod()
            simulations.append(sim_equity)
        return pd.DataFrame(simulations).T

    def add_custom_metric(self, name: str, calculation_func: Callable[[], Any]) -> None:
        """
        Add a custom metric to the reporter.

        Args:
            name (str): The name of the custom metric.
            calculation_func (Callable[[], Any]): A function that calculates the metric.
        """
        setattr(self, f"calculate_{name}", calculation_func)
        logger_main.info(f"Custom metric '{name}' added to the reporter.")

    def perform_attribution_analysis(self) -> Dict[str, Any]:
        """
        Perform a basic performance attribution analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the attribution analysis results.
        """
        # This is a simplified implementation. A full attribution analysis would be more complex.
        total_return = self.calculate_total_return()
        symbol_returns = {}
        for symbol in self.portfolio.positions:
            symbol_trades = [
                trade
                for trade in self.portfolio.closed_trades
                if trade.ticker == symbol
            ]
            symbol_return = sum(trade.metrics.pnl for trade in symbol_trades)
            symbol_returns[symbol] = symbol_return

        return {
            "total_return": total_return,
            "symbol_attribution": symbol_returns,
        }

    # endregion
