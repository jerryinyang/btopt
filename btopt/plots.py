from typing import Dict, List, Literal, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CandlestickChart:
    """
    A class for creating interactive candlestick charts with indicators using Plotly.

    This class allows for dynamic addition of OHLCV (Open, High, Low, Close, Volume) data
    and various indicators. It provides methods to add data, indicators, and plot the chart.

    Attributes:
        ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data.
        indicators (List[Dict]): List of dictionaries containing indicator data and properties.
        datetime_range (Tuple[pd.Timestamp, pd.Timestamp]): The date range of the OHLCV data.
    """

    def __init__(self):
        """Initialize the CandlestickChart with empty data structures."""
        self.ohlcv_data: Union[pd.DataFrame, None] = None
        self.indicators: List[Dict] = []
        self.datetime_range: Union[Tuple[pd.Timestamp, pd.Timestamp], None] = None

    def add_ohlcv_data(self, data: pd.DataFrame) -> None:
        """
        Add OHLCV data to the chart.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data. Must have 'open', 'high',
                                 'low', 'close', and 'volume' columns. The index should be
                                 a DatetimeIndex, or it should contain a 'time' column.

        Raises:
            ValueError: If the required columns are missing or if the data doesn't have
                        a proper datetime index.
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(
                "Data must contain 'open', 'high', 'low', 'close', and 'volume' columns"
            )

        if isinstance(data.index, pd.DatetimeIndex):
            self.ohlcv_data = data
        elif "time" in data.columns:
            self.ohlcv_data = data.set_index("time")
        else:
            raise ValueError("Data must have a datetime index or a 'time' column")

        self.datetime_range = (self.ohlcv_data.index.min(), self.ohlcv_data.index.max())

    def add_indicator(
        self,
        name: str,
        data: pd.Series,
        plot_type: Literal["line", "shape"],
        chart_index: int = 0,
        color: str = "blue",
        style: Literal[
            "solid",
            "dotted",
            "dashed",
            "triangle-down",
            "triangle-up",
            "star",
            "circle",
            "cross",
        ] = "solid",
        width: Literal[1, 2, 3, 4, 5] = 1,
    ) -> None:
        """
        Add an indicator to the chart.

        Args:
            name (str): Name of the indicator.
            data (pd.Series): Series containing indicator data with a DatetimeIndex.
            plot_type (Literal["line", "shape"]): Type of plot for the indicator.
            chart_index (int, optional): Index of the subchart. Defaults to 0 (main chart).
            color (str, optional): Color of the indicator. Defaults to "blue".
            style (Literal["solid", "dotted", "dashed", "triangle-down", "triangle-up", "star", "circle", "cross"], optional):
                Style of the line or shape. Defaults to "solid".
            width (Literal[1, 2, 3, 4, 5], optional): Width or size of the plot. Defaults to 1.

        Raises:
            ValueError: If the indicator data is invalid or doesn't match the OHLCV data range.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Indicator data must have a datetime index")

        if self.datetime_range is None:
            raise ValueError("OHLCV data must be added before adding indicators")

        if not (
            self.datetime_range[0] <= data.index.min() <= self.datetime_range[1]
            and self.datetime_range[0] <= data.index.max() <= self.datetime_range[1]
        ):
            raise ValueError(
                "Indicator data must be within the datetime range of the OHLCV data"
            )

        if plot_type not in ["line", "shape"]:
            raise ValueError("Plot type must be 'line' or 'shape'")

        if plot_type == "line" and style not in ["solid", "dotted", "dashed"]:
            raise ValueError("Line style must be 'solid', 'dotted', or 'dashed'")

        if plot_type == "shape" and style not in [
            "triangle-down",
            "triangle-up",
            "star",
            "circle",
            "cross",
        ]:
            raise ValueError(
                "Shape style must be 'triangle-down', 'triangle-up', 'star', 'circle', or 'cross'"
            )

        self.indicators.append(
            {
                "name": name,
                "data": data,
                "plot_type": plot_type,
                "chart_index": chart_index,
                "color": color,
                "style": style,
                "width": width,
            }
        )

    def plot(self, show_volume: bool = False) -> go.Figure:
        """
        Create and return a Plotly figure with the candlestick chart and indicators.

        Args:
            show_volume (bool, optional): Whether to display the volume bar chart. Defaults to False.

        Returns:
            go.Figure: A Plotly figure object containing the candlestick chart and indicators.

        Raises:
            ValueError: If OHLCV data hasn't been added before plotting.
        """
        if self.ohlcv_data is None:
            raise ValueError("OHLCV data must be added before plotting")

        max_chart_index = max([0] + [ind["chart_index"] for ind in self.indicators])
        fig = make_subplots(
            rows=max_chart_index + 1, cols=1, shared_xaxes=True, vertical_spacing=0.05
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.ohlcv_data.index,
                open=self.ohlcv_data["open"],
                high=self.ohlcv_data["high"],
                low=self.ohlcv_data["low"],
                close=self.ohlcv_data["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Add volume bar chart if show_volume is True
        if show_volume:
            fig.add_trace(
                go.Bar(
                    x=self.ohlcv_data.index,
                    y=self.ohlcv_data["volume"],
                    name="Volume",
                    marker_color="rgba(0, 0, 255, 0.3)",
                ),
                row=1,
                col=1,
            )

        # Add indicators
        for indicator in self.indicators:
            chart_index = min(indicator["chart_index"], max_chart_index)
            if indicator["plot_type"] == "line":
                fig.add_trace(
                    go.Scatter(
                        x=indicator["data"].index,
                        y=indicator["data"],
                        mode="lines",
                        name=indicator["name"],
                        line=dict(
                            color=indicator["color"],
                            width=indicator["width"],
                            dash=indicator["style"],
                        ),
                    ),
                    row=chart_index + 1,
                    col=1,
                )
            else:  # shape plot
                fig.add_trace(
                    go.Scatter(
                        x=indicator["data"].index,
                        y=indicator["data"],
                        mode="markers",
                        name=indicator["name"],
                        marker=dict(
                            color=indicator["color"],
                            size=5 * indicator["width"],
                            symbol=indicator["style"],
                        ),
                    ),
                    row=chart_index + 1,
                    col=1,
                )

        fig.update_layout(
            title="Candlestick Chart with Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            height=800,
            xaxis_rangeslider_visible=False,
        )

        return fig
        return fig
