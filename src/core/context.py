"""Shared local state between pipeline modules."""

from abc import ABC
from dataclasses import dataclass


@dataclass
class Context(ABC):
    """Shared state between pipeline modules. Gets passed down the pipeline."""
