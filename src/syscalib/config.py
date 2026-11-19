"""Configuration loading. All experiment parameters live in a JSON file."""
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.json"


def load_config(path=None):
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
