from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .engine import Engine
    from .portfolio import Portfolio
    from .reporter import Reporter

EngineType = Union["Engine", None]
PortfolioType = Union["Portfolio", None]
ReporterType = Union["Reporter", None]
