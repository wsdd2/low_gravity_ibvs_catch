from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _deep_update(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal indented YAML subset for when PyYAML is unavailable."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _coerce_scalar(value.strip())
    return root


def _coerce_scalar(token: str) -> Any:
    if token in ("true", "True"):
        return True
    if token in ("false", "False"):
        return False
    if token in ("null", "None"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token.strip("\"'")


def load_experiment_config(config_dir: Path | None = None) -> SimpleNamespace:
    directory = Path(config_dir or Path(__file__).resolve().parent.parent / "configs")
    merged: dict[str, Any] = {}
    for name in ("cameras.yaml", "estimator.yaml", "controller.yaml", "evaluation.yaml"):
        path = directory / name
        if path.is_file():
            merged = _deep_update(merged, _read_yaml(path))
    return _to_namespace(merged)
