from dataclasses import dataclass
from typing import Any


@dataclass
class DATA:
    data: Any
    name: str
    timeframe: int
    compression: int

    @property
    def resolution(self):
        return self.timeframe * self.compression

    def __lt__(self, other):
        return self.resolution < other.resolution

    def __eq__(self, value: object) -> bool:
        return self.resolution == value.resolution
