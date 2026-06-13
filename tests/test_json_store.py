"""Unit tests for src.toolbox.storage.JsonStore."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.toolbox.storage import JsonStore  # noqa: E402


def test_round_trip_and_contains(tmp_path):
    """set/get/contains round-trips, including through a reopened instance."""
    path = tmp_path / "store.json"
    store = JsonStore(path)
    assert "standard" not in store
    store.set("standard", {"r_low": 20.0, "r_high": 27.5})

    assert "standard" in store
    assert store.get("standard") == {"r_low": 20.0, "r_high": 27.5}
    assert store.get("missing", "fallback") == "fallback"

    # A fresh instance reads back the persisted data.
    reopened = JsonStore(path)
    assert reopened.get("standard") == {"r_low": 20.0, "r_high": 27.5}
    assert list(reopened.keys()) == ["standard"]
    assert reopened.as_dict() == {"standard": {"r_low": 20.0, "r_high": 27.5}}


def test_set_persists_valid_json_immediately(tmp_path):
    """Every set() leaves a complete, valid JSON file (and no temp file)."""
    path = tmp_path / "store.json"
    store = JsonStore(path)
    store.set("a", 1)
    store.set("b", [1, 2, 3])

    on_disk = json.loads(path.read_text())
    assert on_disk == {"a": 1, "b": [1, 2, 3]}
    assert not path.with_suffix(".json.tmp").exists()  # temp file cleaned up


def test_delete(tmp_path):
    """delete() removes a key and persists; deleting a missing key is a no-op."""
    path = tmp_path / "store.json"
    store = JsonStore(path)
    store.set("a", 1)
    store.delete("a")
    store.delete("never-existed")  # no-op, no crash

    assert "a" not in store
    assert json.loads(path.read_text()) == {}


def test_corrupt_file_recovery(tmp_path):
    """A corrupt store file is preserved as .corrupt and the store starts empty."""
    path = tmp_path / "store.json"
    path.write_text("{not valid json")

    store = JsonStore(path)
    assert store.as_dict() == {}
    assert path.with_suffix(".json.corrupt").exists()  # original preserved for debugging

    store.set("a", 1)
    assert JsonStore(path).get("a") == 1


def test_non_dict_file_starts_empty(tmp_path):
    """A JSON file that isn't an object is ignored (store starts empty)."""
    path = tmp_path / "store.json"
    path.write_text("[1, 2, 3]")

    store = JsonStore(path)
    assert store.as_dict() == {}
