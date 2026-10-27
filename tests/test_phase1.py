"""
Phase 1 tests: action masking, collision resolution, reward config, GAE, normalization,
vectorised env, MAPPO critic input, checkpoints, config overrides, seeded eval, trainer.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from agents.models import Actor
from environment.robot import RobotStatus
from environment.warehouse_env import WarehouseEnv, ACTION_INTERACT, ACTION_WAIT
from evaluation.evaluate import PolicyRunner, load_policy, run_policy_episodes, run_baseline_episodes
from training.buffer import MultiAgentRolloutBuffer
from training.config_utils import apply_overrides, load_config, env_kwargs_from_cfg
from training.normalization import RunningMeanStd, ObsNormalizer, RewardScaler
from training.trainer import train
from training.vec_env import SyncVectorWarehouseEnv

CONFIG_PATH = "configs/default_config.yaml"


def _place(env, positions):
    for r, p in zip(env.robots, positions):
        r.reset(start_pos=p)


# ----------------------------- environment ---------------------------------

def test_action_mask_blocks_static_obstacles_only():
    env = WarehouseEnv(num_robots=2, seed=0)
    env.reset(seed=0)
    _place(env, [(0, 0), (1, 0)])  # corner; robot_1 is east of robot_0
    m = env.action_mask("robot_0")
    assert not m[0] and not m[3]          # North and West leave the grid
    assert m[1] and m[2]                  # South free; East is a robot -> NOT masked
    assert m[ACTION_WAIT]
    _place(env, [(1, 3), (8, 8)])         # (2, 3) is a shelf in kiva_standard
    assert env.grid.is_shelf(2, 3)
    assert not env.action_mask("robot_0")[2]


def test_interact_mask_only_when_adjacent_pickup_possible():
    env = WarehouseEnv(num_robots=1, seed=0)
    env.reset(seed=0)
    r = env.robots[0]
    r.reset(start_pos=(0, 0))
    assert not env.action_mask("robot_0")[ACTION_INTERACT]      # idle
    r.assign_order(order_id=99, shelf_access_pos=(1, 0))
    assert env.action_mask("robot_0")[ACTION_INTERACT]          # adjacent to target
    r.assign_order(order_id=99, shelf_access_pos=(5, 0))
    assert not env.action_mask("robot_0")[ACTION_INTERACT]      # too far
    legacy = WarehouseEnv(num_robots=1, seed=0, mask_useless_interact=False)
    legacy.reset(seed=0)
    assert legacy.action_mask("robot_0")[ACTION_INTERACT]


def test_masks_in_infos():
    env = WarehouseEnv(num_robots=3, max_steps=5)
    _, infos = env.reset(seed=1)
    assert infos["robot_0"]["action_mask"].shape == (6,)
    _, _, _, _, infos = env.step({a: ACTION_WAIT for a in env.agents})
    assert infos["robot_2"]["action_mask"].dtype == bool
    assert env.action_masks().shape == (3, 6)


def test_state_from_cached_observations_is_identical():
    env = WarehouseEnv(num_robots=5, max_steps=10)
    obs, _ = env.reset(seed=3)
    assert np.array_equal(env.state(obs), env.state())
    obs, _, _, _, _ = env.step({a: 1 for a in env.agents})
    assert np.array_equal(env.state(obs), env.state())


def test_random_policy_never_puts_two_robots_in_one_cell():
    env = WarehouseEnv(num_robots=5, max_steps=300)
    rng = np.random.default_rng(0)
    for seed in range(3):
        env.reset(seed=seed)
        while env.agents:
            env.step({a: int(rng.integers(6)) for a in env.agents})
            assert len({r.pos for r in env.robots}) == env.num_robots


def test_follow_into_vacated_cell_is_allowed():
    env = WarehouseEnv(num_robots=2, seed=0)
    env.reset(seed=0)
    _place(env, [(0, 0), (1, 0)])
    env.step({"robot_0": 2, "robot_1": 2})  # both move East in a line
    assert [r.pos for r in env.robots] == [(1, 0), (2, 0)]
    assert env.total_fleet_collisions == 0


def test_collision_counters_split_by_kind():
    env = WarehouseEnv(num_robots=2, seed=0)
    env.reset(seed=0)
    _place(env, [(0, 0), (2, 0)])
    env.step({"robot_0": 0, "robot_1": 3})  # robot_0 bumps the boundary, robot_1 moves West freely
    assert env.total_wall_bumps == 1 and env.total_robot_collisions == 0
    _place(env, [(0, 0), (1, 0)])
    env.step({"robot_0": 2, "robot_1": 3})  # swap conflict
    assert env.total_robot_collisions == 2
    assert env.total_fleet_collisions == env.total_wall_bumps + env.total_robot_collisions


def _single_robot_with_far_target(rewards=None):
    env = WarehouseEnv(num_robots=1, seed=0, rewards=rewards, order_lambda=0.0)
    env.reset(seed=0)
    r = env.robots[0]
    r.reset(start_pos=(0, 0))
    r.assign_order(order_id=99, shelf_access_pos=(5, 0))
    return env


def test_noop_interact_gets_idle_penalty_by_default():
    env = _single_robot_with_far_target()
    _, rew, _, _, _ = env.step({"robot_0": ACTION_INTERACT})
    rc = env.reward_cfg
    assert rew["robot_0"] == pytest.approx(rc["time"] + rc["idle"])


def test_legacy_interact_has_no_idle_penalty():
    env = _single_robot_with_far_target(rewards={"penalize_noop_interact": False})
    _, rew, _, _, _ = env.step({"robot_0": ACTION_INTERACT})
    assert rew["robot_0"] == pytest.approx(env.reward_cfg["time"])


def test_reward_overrides_and_unknown_key():
    env = _single_robot_with_far_target(rewards={"time": -1.0})
    _, rew, _, _, _ = env.step({"robot_0": ACTION_WAIT})
    assert rew["robot_0"] == pytest.approx(-1.0 + env.reward_cfg["idle"])
    with pytest.raises(ValueError):
        WarehouseEnv(rewards={"tme": -1.0})


def test_adjacent_interact_picks_up():
    from environment.order import Order
    env = WarehouseEnv(num_robots=1, seed=0, order_lambda=0.0)
    env.reset(seed=0)
    env.robots[0].reset(start_pos=(1, 1))
    env.order_generator.active_orders[7] = Order(
        order_id=7, shelf_pos=(2, 2), shelf_access_pos=(1, 2), station_pos=env.grid.stations[0], spawn_step=0)
    env.robots[0].assign_order(7, (1, 2))
    _, rew, _, _, _ = env.step({"robot_0": ACTION_INTERACT})
    assert env.robots[0].status == RobotStatus.CARRYING_TO_STATION
    assert rew["robot_0"] > env.reward_cfg["pickup"] - 1.0


# ----------------------------- policy / buffer -----------------------------

def test_masked_actor_never_samples_masked_actions():
    torch.manual_seed(0)
    actor = Actor()
    obs = torch.randn(500, 28)
    mask = torch.zeros(500, 6, dtype=torch.bool)
    mask[:, 1] = True
    mask[::2, 4] = True
    actions, logp, ent = actor.get_action_and_logprob(obs, action_mask=mask)
    assert mask[torch.arange(500), actions].all()
    _, lp_masked, _ = actor.get_action_and_logprob(obs, torch.zeros(500, dtype=torch.int64), action_mask=mask)
    assert (lp_masked < -1e6).all()
    assert torch.isfinite(ent).all()


def test_gae_matches_manual_computation_two_envs():
    gamma, lam = 0.9, 0.8
    buf = MultiAgentRolloutBuffer(2, num_agents=1, obs_dim=1, state_dim=1, gamma=gamma, gae_lambda=lam, num_envs=2)
    # env 0: no episode end; env 1: transition 0 ends the episode
    rewards = [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]
    dones = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])]
    values = [torch.tensor([0.5, 0.5]), torch.tensor([0.25, 0.25])]
    for t in range(2):
        buf.insert(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, dtype=torch.int64), torch.zeros(2),
                   rewards[t], dones[t], values[t])
    last_v = torch.tensor([1.0, 1.0])
    buf.compute_returns_and_advantages(last_v)
    d1 = 2.0 + gamma * 1.0 - 0.25
    d0_env0 = 1.0 + gamma * 0.25 - 0.5
    d0_env1 = 1.0 - 0.5  # no bootstrap across the boundary
    assert buf.advantages[1, 0, 0] == pytest.approx(d1)
    assert buf.advantages[0, 0, 0] == pytest.approx(d0_env0 + gamma * lam * d1)
    assert buf.advantages[0, 1, 0] == pytest.approx(d0_env1)


def test_mappo_critic_inputs():
    obs = torch.randn(3, 5, 28)
    state = torch.randn(3, 150)
    glob = MAPPOAgent(critic_input="global")
    _, _, v = glob.get_actions_and_values(obs, state)
    assert v.shape == (3, 5) and torch.allclose(v, v[:, :1].expand(3, 5))
    spec = MAPPOAgent(critic_input="agent_specific")
    _, _, v = spec.get_actions_and_values(obs, state)
    assert v.shape == (3, 5) and not torch.allclose(v, v[:, :1].expand(3, 5))
    # single-env shapes used by the original code still work
    _, _, v1 = spec.get_actions_and_values(obs[0], state[0])
    assert v1.shape == (5,)
    with pytest.raises(ValueError):
        MAPPOAgent(critic_input="nope")


# ----------------------------- normalization -------------------------------

def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 4))
    rms = RunningMeanStd(shape=(4,), epsilon=0.0 + 1e-12)
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(axis=0), atol=1e-6)
    assert np.allclose(rms.var, data.var(axis=0), atol=1e-6)


def test_obs_normalizer_roundtrip_and_disabled():
    norm = ObsNormalizer(3, enabled=True, clip=5.0, epsilon=1e-8)
    x = np.random.default_rng(1).normal(size=(50, 3))
    z = norm(x, update=True)
    assert abs(z.mean()) < 0.5
    restored = ObsNormalizer.from_state_dict(norm.state_dict())
    assert np.allclose(restored(x, update=False), norm(x, update=False))
    off = ObsNormalizer(3, enabled=False, clip=5.0, epsilon=1e-8)
    assert np.allclose(off(x, update=True), x.astype(np.float32))


def test_reward_scaler_keeps_sign_and_resets():
    sc = RewardScaler((2, 1), gamma=0.99, enabled=True, clip=10.0, epsilon=1e-8)
    for _ in range(20):
        out = sc(np.array([[5.0], [-5.0]]), np.array([False, False]))
    assert out[0, 0] > 0 > out[1, 0]
    sc(np.array([[1.0], [1.0]]), np.array([True, False]))
    assert sc.returns[0, 0] == 0.0 and sc.returns[1, 0] != 0.0


# ----------------------------- vec env / eval / config ---------------------

def test_vec_env_shapes_and_autoreset():
    venv = SyncVectorWarehouseEnv(dict(num_robots=3, max_steps=4), num_envs=2)
    obs, masks, states = venv.reset(seeds=[1, 2])
    assert obs.shape == (2, 3, 28) and masks.shape == (2, 3, 6) and states.shape == (2, venv.state_dim)
    for t in range(4):
        obs, rew, dones, masks, states, infos = venv.step(np.full((2, 3), ACTION_WAIT))
    assert dones.all()
    assert infos[0]["truncated"] and infos[0]["final_obs"].shape == (3, 28)
    assert infos[1]["episode"]["steps"] == 4
    assert venv.envs[0].current_step == 0  # already reset


def test_config_overrides():
    cfg = {"a": {"b": 1, "c": True}}
    out = apply_overrides(cfg, ["a.b=2.5", "a.c=false"])
    assert out["a"]["b"] == 2.5 and out["a"]["c"] is False and cfg["a"]["b"] == 1
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["a.typo=1"])
    full = load_config(CONFIG_PATH, ["hyperparameters.value_clip=0.2"])
    assert full["hyperparameters"]["value_clip"] == 0.2
    assert env_kwargs_from_cfg(full["environment"])["rewards"]["delivery"] == 10.0


def test_seed_sets_are_disjoint():
    cfg = load_config(CONFIG_PATH)
    dev, test = set(cfg["seeds"]["dev_eval"]), set(cfg["evaluation"]["seeds"])
    stride = cfg["training"]["env_seed_stride"]
    train_env_seeds = {s * stride + k for s in cfg["seeds"]["train"] for k in range(cfg["training"]["num_envs"])}
    assert not (dev & test) and not (dev & train_env_seeds) and not (test & train_env_seeds)


def test_seeded_policy_eval_is_reproducible():
    kw = dict(num_robots=3, max_steps=30)
    runner = PolicyRunner(IPPOAgent(), "ippo", use_mask=True)
    a = run_policy_episodes(runner, kw, [11, 12], torch_seed=3)
    b = run_policy_episodes(runner, kw, [11, 12], torch_seed=3)
    assert pd.DataFrame(a).equals(pd.DataFrame(b))  # .equals treats NaN == NaN
    base = run_baseline_episodes(kw, [11])
    assert base[0]["steps"] == 30 and base[0]["collisions"] == base[0]["wall_bumps"] + base[0]["robot_collisions"]


def test_summarize_episodes_skips_text_and_nan():
    from evaluation.metrics import summarize_episodes
    s = summarize_episodes([{"method": "x", "seed": 1, "thru": 2.0, "lat": float("nan")},
                            {"method": "x", "seed": 2, "thru": 4.0, "lat": 3.0}])
    assert s["n"] == 2 and s["thru_mean"] == 3.0 and s["lat_mean"] == 3.0
    assert "method_mean" not in s and "seed_mean" not in s


def test_legacy_checkpoints_still_load():
    for method, path in (("ippo", "results/models/ippo_best.pt"), ("mappo", "results/models/mappo_best.pt")):
        if not os.path.exists(path):
            pytest.skip("committed checkpoints not present")
        runner = load_policy(method, path, obs_dim=28, state_dim=150)
        assert runner.obs_norm is None and runner.use_mask is False


# ----------------------------- trainer -------------------------------------

def _tiny_cfg(tmp_path, method_overrides=()):
    return load_config(CONFIG_PATH, [
        "environment.max_steps=20", "environment.num_robots=3",
        "training.num_envs=2", "training.eval_interval=32", f"training.runs_dir={tmp_path.as_posix()}",
        "seeds.dev_eval=[5001]",
        "ippo.buffer_size=16", "ippo.batch_size=32", "mappo.buffer_size=16", "mappo.batch_size=32",
        *method_overrides,
    ])


@pytest.mark.parametrize("method", ["ippo", "mappo"])
def test_trainer_end_to_end_and_reproducible(tmp_path, method):
    cfg = _tiny_cfg(tmp_path)
    s1 = train(method, cfg, seed=1, run_name="a", total_timesteps=64)
    s2 = train(method, cfg, seed=1, run_name="b", total_timesteps=64)
    assert s1["env_steps"] == 64
    c1 = pd.read_csv(os.path.join(s1["run_dir"], "learning_curve.csv"))
    c2 = pd.read_csv(os.path.join(s2["run_dir"], "learning_curve.csv"))
    cols = ["env_steps", "dev_throughput_mean", "dev_collisions_mean", "policy_loss"]
    pd.testing.assert_frame_equal(c1[cols], c2[cols])
    runner = load_policy(method, os.path.join(s1["run_dir"], "best_dev.pt"), 28, 10 * 28 + 10)
    assert runner.use_mask and runner.obs_norm is not None and runner.obs_norm.enabled


def test_trainer_respects_ablation_switches(tmp_path):
    cfg = _tiny_cfg(tmp_path, ["training.action_masking=false", "training.normalize_obs=false",
                               "training.normalize_reward=false"])
    s = train("ippo", cfg, seed=2, run_name="abl", total_timesteps=32)
    runner = load_policy("ippo", os.path.join(s["run_dir"], "final.pt"), 28, 3 * 28 + 10)
    assert not runner.use_mask and not runner.obs_norm.enabled


def test_plot_learning_curves_smoke(tmp_path, monkeypatch):
    import sys
    from evaluation import plot_learning_curves as plc
    run = tmp_path / "ippo_seed1"
    run.mkdir()
    pd.DataFrame({"env_steps": [0, 100], "dev_throughput_mean": [1.0, 2.0], "dev_throughput_std": [0.1, 0.2],
                  "dev_robot_collisions_mean": [5.0, 3.0], "entropy": [None, 1.2],
                  "explained_variance": [None, 0.3]}).to_csv(run / "learning_curve.csv", index=False)
    base = tmp_path / "base.csv"
    pd.DataFrame([{"method": "baseline", "throughput_mean": 3.0, "robot_collisions_mean": 1.0}]).to_csv(base, index=False)
    out = tmp_path / "c.png"
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(run), "--baseline-csv", str(base), "--output", str(out)])
    plc.main()
    assert out.exists() and out.stat().st_size > 0



# ----------------------------- Phase 2: curriculum --------------------------

def test_curriculum_runs_all_stages_and_ablation_uses_last(tmp_path):
    from training.trainer import curriculum_stages
    cfg = _tiny_cfg(tmp_path, ["training.eval_interval=8", "ippo.buffer_size=4", "ippo.batch_size=16",
                               "curriculum.enabled=true"])
    s = train("ippo", cfg, seed=3, run_name="cur", total_timesteps=80)  # 10 updates -> 1 / 1 / 8
    curve = pd.read_csv(os.path.join(s["run_dir"], "learning_curve.csv"))
    assert set(curve["stage"]) == {0, 1, 2}
    off = apply_overrides(cfg, ["curriculum.enabled=false"])
    stages = curriculum_stages(off)
    assert len(stages) == 1 and stages[0]["num_robots"] == cfg["curriculum"]["stages"][-1]["num_robots"]
    bad = apply_overrides(cfg, ["curriculum.stages=[{fraction: 0.5, num_robots: 2, order_lambda_range: [0.1, 0.1]}]"])
    with pytest.raises(ValueError):
        curriculum_stages(bad)


def test_vec_env_mixed_layouts_and_lambda_range():
    kws = [dict(num_robots=2, max_steps=3, layout_type=l) for l in ("kiva_standard", "narrow_corridor")]
    venv = SyncVectorWarehouseEnv(kws, 2, lambda_range=(0.05, 0.3), seed=0)
    venv.reset(seeds=[1, 2])
    assert venv.envs[1].layout_type == "narrow_corridor"
    lams = {e.order_lambda for e in venv.envs}
    assert all(0.05 <= l <= 0.3 for l in lams) and len(lams) == 2
    with pytest.raises(ValueError):
        SyncVectorWarehouseEnv([dict(num_robots=2), dict(num_robots=3)], 2)


def test_state_padding_keeps_size_fixed():
    small = WarehouseEnv(num_robots=2, state_max_robots=10)
    obs, _ = small.reset(seed=0)
    st = small.state(obs)
    assert st.shape == (10 * 28 + 10,)
    assert np.all(st[2 * 28:10 * 28] == 0)
    with pytest.raises(ValueError):
        WarehouseEnv(num_robots=5, state_max_robots=3)
