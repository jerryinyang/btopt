from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .data.manager import DataManager
    from .engine import Engine
    from .portfolio import Portfolio
    from .portfolio_managers import OrderManager
    from .reporter import Reporter
    from .strategy import Strategy

EngineType = Union["Engine", None]
PortfolioType = Union["Portfolio", None]
ReporterType = Union["Reporter", None]
StrategyType = Union["Strategy", None]
DataManagerType = Union["DataManager", None]
OrderManagerType = Union["OrderManager", None]
