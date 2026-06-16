from dataclasses import dataclass
from typing import Optional
from src.core.module import Context

@dataclass
class CounterContext(Context):
    number: Optional[int] = None
    doubled: Optional[int] = None
