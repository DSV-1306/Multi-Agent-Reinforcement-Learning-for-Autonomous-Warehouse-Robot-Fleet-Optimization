"""
Phase 0 audit tests.

The first two tests pinned bugs found in the audit as strict xfails; both were fixed in
Phase 1 and are now regular regression tests.
"""

import pytest
import yaml

from evaluation.phase0_audit import (
    check_chain_collision_overlap,
    check_gae_episode_boundary,
    static_findings,
    run_diagnostics,
    RandomPolicy,
)
from environment.warehouse_env import WarehouseEnv

CONFIG_PATH = "configs/default_config.yaml"


def _cfg():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Fixed in Phase 1 (iterative conflict resolution in WarehouseEnv._resolve_moves).
def test_no_two_robots_share_a_cell_after_chain_conflict():
    assert not check_chain_collision_overlap()["overlap"]


# Fixed in Phase 1 (post-step done convention in MultiAgentRolloutBuffer).
def test_gae_does_not_bootstrap_across_episode_boundary():
    hp = _cfg()["hyperparameters"]
    assert not check_gae_episode_boundary(hp["gamma"], hp["gae_lambda"])["leaks_across_episodes"]


def test_static_findings_budget_arithmetic():
    cfg = _cfg()
    s = static_findings(cfg)
    env_cfg = cfg["environment"]
    for kind in ("ippo", "mappo"):
        assert s[kind]["episodes_at_config_budget"] == cfg[kind]["total_timesteps"] / env_cfg["max_steps"]
        assert s[kind]["samples_per_update"] == cfg[kind]["buffer_size"] * env_cfg["num_robots"]


def test_diagnostics_smoke():
    env = WarehouseEnv(num_robots=3, grid_width=10, grid_height=10, max_steps=20)
    d = run_diagnostics(env, RandomPolicy(env, seed=0), seeds=[0], gamma=0.99)
    assert abs(sum(d["action_freq"].values()) - 1.0) < 1e-9
    assert 0.0 <= d["wall_bump_share_of_collisions"] <= 1.0
    assert d["episodes"] == 1
