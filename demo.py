"""
Interactive Live Simulation Demo for Warehouse Multi-Agent System.
Visualizes the warehouse fleet executing either Baseline, IPPO, or MAPPO in real-time.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.warehouse_env import WarehouseEnv
from visualization.renderer import WarehouseVisualizer
from baselines.heuristic_agent import HeuristicFleetController
from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from evaluation.evaluate import load_policy


def run_demo(
    method: str = "baseline",
    fps: int = 15,
    max_steps: int = 400,
    order_lambda: float = 0.25,
    seed: int = 42,
    checkpoint: str = None,
):
    print(f"\n=======================================================")
    print(f"   WAREHOUSE MULTI-AGENT SIMULATION: {method.upper()}")
    print(f"=======================================================")

    state_max_robots = None
    target_ckpt = checkpoint or ("results/models/mappo_best.pt" if method == "mappo" else None)
    if target_ckpt and os.path.exists(target_ckpt):
        try:
            ckpt_meta = torch.load(target_ckpt, map_location="cpu", weights_only=False)
            extra0 = ckpt_meta.get("extra", {})
            if "state_max_robots" in extra0:
                state_max_robots = extra0["state_max_robots"]
            elif "critic" in ckpt_meta:
                in_f = ckpt_meta["critic"]["network.0.weight"].shape[1]
                c_in = ckpt_meta.get("critic_input", "global")
                s_dim = in_f - (28 if c_in == "agent_specific" else 0)
                state_max_robots = (s_dim - 10) // 28
        except Exception:
            pass

    env = WarehouseEnv(
        num_robots=5,
        grid_width=10,
        grid_height=10,
        layout_type="kiva_standard",
        order_lambda=order_lambda,
        max_steps=max_steps,
        state_max_robots=state_max_robots,
        render_mode="human",
        seed=seed,
    )

    visualizer = WarehouseVisualizer(env, cell_size=55, fps=fps)

    # Controller initialization
    controller = None
    ippo_agent = None
    mappo_agent = None

    if method == "baseline":
        controller = HeuristicFleetController(env)
    elif checkpoint:
        pass  # Checkpoint runner is loaded below
    elif method == "ippo":
        ippo_agent = IPPOAgent(obs_dim=env.obs_dim, action_dim=6)
        path = "results/models/ippo_best.pt"
        if os.path.exists(path):
            ippo_agent.load(path)
            print(f"Loaded IPPO model: {path}")
        else:
            print(f"Warning: {path} not found! Using initialized agent.")
    elif method == "mappo":
        mappo_agent = MAPPOAgent(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=6)
        path = "results/models/mappo_best.pt"
        if os.path.exists(path):
            mappo_agent.load(path)
            print(f"Loaded MAPPO model: {path}")
        else:
            print(f"Warning: {path} not found! Using initialized agent.")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Phase 1 checkpoints carry normalizer stats and the masking flag; load_policy applies them.
    runner = load_policy(method, checkpoint, env.obs_dim, env.state_dim) if checkpoint and method != "baseline" else None
    if runner:
        print(f"Loaded {method.upper()} checkpoint via load_policy: {checkpoint}")

    obs_dict, _ = env.reset(seed=seed)
    if controller:
        controller.reset()

    print("Running simulation... Close the Pygame window to stop.")
    running = True

    for step in range(max_steps):
        # Determine actions
        if runner:
            obs_arr = np.stack([obs_dict[a] for a in env.possible_agents])
            act = runner.act(obs_arr[None], env.state(obs_dict)[None], env.action_masks()[None])[0]
            actions = {f"robot_{i}": int(act[i]) for i in range(env.num_robots)}
        elif method == "baseline":
            actions = controller.get_actions()
        elif method == "ippo":
            obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
            with torch.no_grad():
                act_tensor, _, _ = ippo_agent.get_actions_and_values(obs_tensor)
            actions = {f"robot_{i}": int(act_tensor[i]) for i in range(env.num_robots)}
        elif method == "mappo":
            obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
            state_tensor = torch.tensor(env.state(), dtype=torch.float32)
            with torch.no_grad():
                act_tensor, _, _ = mappo_agent.get_actions_and_values(obs_tensor, state_tensor)
            actions = {f"robot_{i}": int(act_tensor[i]) for i in range(env.num_robots)}

        # Environment step
        obs_dict, rewards, term, trunc, infos = env.step(actions)

        # Render frame
        rendered = visualizer.render()
        if rendered is None and not visualizer._is_initialized:
            print("Visualizer window closed.")
            break

        if all(trunc.values()):
            print("Episode reached max steps.")
            break

    visualizer.close()
    print("\n---------------- Simulation Summary ----------------")
    print(f"Method:             {method.upper()}")
    print(f"Steps Completed:    {env.current_step}")
    print(f"Orders Fulfilled:   {env.total_orders_completed}")
    print(f"Fleet Collisions:   {env.total_fleet_collisions}")
    print("----------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "ippo", "mappo"])
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--order_lambda", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None,
                        help="Phase 1 checkpoint (e.g. results/runs/ippo_seed1/best_dev.pt); default: legacy results/models/*_best.pt")
    args = parser.parse_args()

    run_demo(
        method=args.method,
        fps=args.fps,
        max_steps=args.steps,
        order_lambda=args.order_lambda,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
