from dataclasses import dataclass
from src.core.module import Context

@dataclass
class NullContext(Context):
    """Empty context for modules that don't need to pass any state."""
    pass