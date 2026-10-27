"""
Phase 4 tests: hierarchical allocation (RL allocator + Space-Time A*).
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from agents.hier_agent import (AllocatorNet, GreedyNearestDispatcher, RLDispatcher, candidates_and_features,
                               _DistanceCache, make_hier_env, run_episode, load_allocator, CAND_FEATURES,
                               CTX_FEATURES)
from environment.warehouse_env import WarehouseEnv
from evaluation.evaluate import run_baseline_episodes
from training.config_utils import load_config
from training.train_hier import episode_transitions, train_hier

HCFG = {"candidates_nearest": 2, "candidates_oldest": 2, "allow_defer": True, "queue_norm": 20.0, "hidden_dim": 16}


def test_greedy_hook_reproduces_builtin_dispatch_exactly():
    kw = dict(num_robots=4, order_lambda=0.25, max_steps=120)
    for seed in (5001, 5002):
        base = run_baseline_episodes(kw, [seed])[0]
        env, ctrl = make_hier_env(kw, GreedyNearestDispatcher())
        hier = run_episode(env, ctrl, seed)
        for key in ("throughput", "collisions", "travel_distance", "orders_spawned"):
            assert base[key] == hier[key], key


def test_external_dispatch_without_dispatcher_assigns_nothing():
    env = WarehouseEnv(num_robots=3, order_lambda=0.5, dispatch_mode="external")
    env.reset(seed=0)
    for _ in range(10):
        env.step({a: 4 for a in env.agents})
    assert all(r.is_idle for r in env.robots) and env.order_generator.pending_orders
    env.assign_order(0, 0)
    assert not env.robots[0].is_idle
    with pytest.raises(ValueError):
        env.assign_order(0, 0)
    with pytest.raises(ValueError):
        WarehouseEnv(dispatch_mode="nope")


def test_candidate_features_and_mask():
    env = WarehouseEnv(num_robots=3, order_lambda=1.0, dispatch_mode="external")
    env.reset(seed=1)
    for _ in range(5):
        env.step({a: 4 for a in env.agents})
    chosen, cand, ctx, mask = candidates_and_features(env, env.robots[0], HCFG, _DistanceCache())
    K = HCFG["candidates_nearest"] + HCFG["candidates_oldest"]
    assert cand.shape == (K, CAND_FEATURES) and ctx.shape == (CTX_FEATURES,) and mask.shape == (K + 1,)
    assert len(chosen) == len(set(chosen)) <= K and mask[:len(chosen)].all() and mask[K]
    assert cand[:len(chosen), 5].sum() == 1  # exactly one "nearest" flag
    assert np.all(cand >= 0) and np.all(cand[:, :5] <= 1.0 + 1e-6)


def test_allocator_net_respects_mask():
    torch.manual_seed(0)
    net = AllocatorNet(16)
    cand, ctx = torch.rand(64, 4, CAND_FEATURES), torch.rand(64, CTX_FEATURES)
    mask = torch.zeros(64, 5, dtype=torch.bool)
    mask[:, 1] = mask[:, 3] = True
    dist, value = net(cand, ctx, mask)
    a = dist.sample()
    assert mask[torch.arange(64), a].all() and value.shape == (64,)


def test_semi_mdp_transitions():
    gamma = 0.5
    decisions = [{"t": 0, "value": 1.0}, {"t": 0, "value": 2.0}, {"t": 2, "value": 3.0}]
    rewards = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}
    tr = episode_transitions(decisions, rewards, last_step=3, gamma=gamma)
    assert tr[0]["R"] == 0.0 and tr[0]["disc"] == 1.0 and tr[0]["next_value"] == 2.0   # same-time decision
    assert tr[1]["R"] == pytest.approx(1.0 + 0.5 * 2.0) and tr[1]["disc"] == pytest.approx(0.25)
    assert tr[2]["R"] == pytest.approx(4.0 + 0.5 * 8.0) and tr[2]["done"] and tr[2]["next_value"] == 0.0


def test_hier_training_smoke_and_reload(tmp_path):
    cfg = load_config("configs/default_config.yaml", [
        f"training.runs_dir={tmp_path.as_posix()}", "seeds.dev_eval=[5001]", "training.eval_interval=400",
        "environment.max_steps=100", "hier.rollout_decisions=20", "hier.batch_size=16"])
    s = train_hier(cfg, seed=1, run_name="h", total_timesteps=300)
    curve = pd.read_csv(os.path.join(s["run_dir"], "learning_curve.csv"))
    assert "dev_throughput_mean" in curve and s["decisions"] > 0
    disp = load_allocator(os.path.join(s["run_dir"], "final.pt"))
    assert isinstance(disp, RLDispatcher)


# ----------------------------- Phase 5: statistics / queue -----------------

def test_mean_ci_and_paired_difference():
    from evaluation.stats import mean_ci, summarize
    r = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert r["mean"] == 2.0 and r["ci"] == pytest.approx(4.302653 * 1.0 / np.sqrt(3), rel=1e-5)
    rows = []
    for seed in (1, 2):
        rows.append({"variant": "baseline", "train_seed": -1, "scenario": "a", "seed": seed, "throughput": 10.0 * seed})
        for ts in (1, 2, 3):
            rows.append({"variant": "rl", "train_seed": ts, "scenario": "a", "seed": seed,
                         "throughput": 10.0 * seed + ts})
    s = summarize(pd.DataFrame(rows), ["throughput"])
    rl = s[(s.variant == "rl") & (s.scenario == "a")].iloc[0]
    assert rl["diff_vs_baseline"] == pytest.approx(2.0) and rl["n"] == 3 and rl["unit"] == "training seeds"
    assert set(s["scenario"]) == {"a", "ALL"}


def test_run_queue_builds_one_job_per_variant_and_seed():
    from training.run_queue import build_jobs
    cfg = load_config("configs/default_config.yaml")
    jobs = build_jobs(cfg)
    ex = cfg["experiments"]
    assert len(jobs) == len(ex["variants"]) * len(ex["seeds"])
    nomask = [j for j in jobs if j["name"].startswith("ippo_no_mask_")][0]
    assert "training.action_masking=false" in nomask["cmd"] and "--variant" in nomask["cmd"]
    assert len({j["name"] for j in jobs}) == len(jobs)


def test_plot_results_smoke(tmp_path, monkeypatch):
    import sys
    from evaluation import plot_results
    rows = [{"scenario": sc, "variant": v, "metric": "throughput", "mean": 1.0, "ci95": 0.1}
            for sc in ("a", "ALL") for v in ("baseline", "ippo")]
    p = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    out = tmp_path / "f.png"
    monkeypatch.setattr(sys, "argv", ["x", "--summary", str(p), "--variants", "baseline", "ippo",
                                      "--metrics", "throughput", "--output", str(out)])
    plot_results.main()
    assert out.exists()
