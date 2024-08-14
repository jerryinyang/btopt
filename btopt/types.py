from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .data.manager import DataManager
    from .engine import Engine
    from .portfolio import Portfolio
    from .reporter import Reporter
    from .strategy.strategy import Strategy

EngineType = Union["Engine", None]
PortfolioType = Union["Portfolio", None]
ReporterType = Union["Reporter", None]
StrategyType = Union["Strategy", None]
DataManagerType = Union["DataManager", None]
