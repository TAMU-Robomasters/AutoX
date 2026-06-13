"""Tiny key-value JSON store with atomic writes.

Replaces the old ``ColdStorage`` autosaving-dict subclass with an explicit,
boring API:

- ``set()`` persists immediately and atomically (write to a temp file in the
  same directory, then ``os.replace``), so a crash can never leave a
  half-written file behind.
- Values must be JSON-serializable. The file is pretty-printed with sorted
  keys so diffs stay readable.
- Single-writer: exactly one process should own a store instance for a given
  path. Readers in other processes should open their own instance (a snapshot
  of the file at open time).

Usage:
    store = JsonStore(absolute_path_to.robot_constants)
    store.set("standard", {"r_low": 20.0})
    if "standard" in store:
        constants = store.get("standard")
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from src.toolbox.logger import get_logger

log = get_logger("json_store")


class JsonStore:
    """Dict-of-JSON-values persisted to one file, saved atomically on every write."""

    def __init__(self, path: Union[str, Path]):
        """Load the store from ``path``, starting empty if it doesn't exist.

        A corrupt (unparseable) file is renamed to ``<path>.corrupt`` and the
        store starts empty rather than crashing.
        """
        self._path = Path(path)
        self._data: Dict[str, Any] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                corrupt_path = self._path.with_suffix(self._path.suffix + ".corrupt")
                os.replace(self._path, corrupt_path)
                log.warning(
                    "corrupt store at %s; moved to %s and starting empty",
                    self._path,
                    corrupt_path,
                )
            if not isinstance(self._data, dict):
                log.warning("store at %s is not a JSON object; starting empty", self._path)
                self._data = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Return the value for ``key``, or ``default`` if missing."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set ``key`` to ``value`` and atomically persist the whole store."""
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> None:
        """Remove ``key`` (no-op if missing) and persist."""
        if key in self._data:
            del self._data[key]
            self._save()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Iterable[str]:
        """Return the stored keys."""
        return self._data.keys()

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the full store contents."""
        return dict(self._data)

    @property
    def path(self) -> Path:
        """The file this store persists to."""
        return self._path

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, self._path)  # atomic on POSIX: never a torn file
