"""
Config loading with dotted command-line overrides, e.g.
    --set training.action_masking=false --set ippo.total_timesteps=200000
Values are parsed as YAML, so numbers, booleans, null and lists work.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional

import yaml

# Layout names with this prefix are loaded from <environment.layouts_dir>/<name>.json (data/mapf_layouts.py)
FILE_LAYOUT_PREFIX = "mapf:"


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config section {p!r} in override {item!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key {key!r} (typo? add it to the YAML first)")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the Phase 3 preprocessing scripts (see data/README.md)")
    with open(path, "r") as f:
        return json.load(f)


def order_config_from_cfg(env_cfg: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
    ocfg = env_cfg.get("orders", {})
    mode = mode or ocfg.get("mode", "synthetic")
    if mode == "synthetic":
        return {"mode": "synthetic"}
    stats = _load_json(ocfg["empirical_stats_file"])
    return {"mode": "empirical", "empirical": {
        "sku_probs": stats["sku_probs"], "lines_pmf": stats["lines_pmf"],
        "hourly_profile": stats["hourly_profile"], "steps_per_hour": ocfg["steps_per_hour"],
        "start_hour": ocfg["start_hour"], "shelf_assignment_seed": ocfg["shelf_assignment_seed"],
    }}


def env_kwargs_from_cfg(env_cfg: Dict[str, Any], **overrides) -> Dict[str, Any]:
    """
    Builds WarehouseEnv kwargs. Extra keyword overrides may include the pseudo-key `order_mode`
    ("synthetic" / "empirical"); a layout_type of "mapf:<name>" loads a held-out layout file.
    """
    order_mode = overrides.pop("order_mode", None)
    kwargs = dict(
        num_robots=env_cfg["num_robots"],
        grid_width=env_cfg["grid_width"],
        grid_height=env_cfg["grid_height"],
        layout_type=env_cfg["layout_type"],
        order_lambda=env_cfg["order_lambda"],
        max_steps=env_cfg["max_steps"],
        rewards=env_cfg.get("rewards"),
        mask_useless_interact=env_cfg.get("mask_useless_interact", True),
        state_max_robots=env_cfg.get("state_max_robots"),
    )
    kwargs.update(overrides)
    if str(kwargs["layout_type"]).startswith(FILE_LAYOUT_PREFIX):
        name = kwargs.pop("layout_type")[len(FILE_LAYOUT_PREFIX):]
        kwargs["layout_spec"] = _load_json(os.path.join(env_cfg["layouts_dir"], f"{name}.json"))
    if order_mode is not None or env_cfg.get("orders", {}).get("mode", "synthetic") != "synthetic":
        kwargs["order_config"] = order_config_from_cfg(env_cfg, order_mode)
    return kwargs
