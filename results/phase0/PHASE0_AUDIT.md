# Phase 0 – Audit & Reproduction

All numbers below come from `python evaluation/phase0_audit.py` (output: `results/phase0/audit_report.json`,
`results/phase0/repro_eval.csv`) run on: Intel i3-1005G1 (2 cores / 4 threads, 1.2 GHz), 7.8 GB RAM,
Windows 11, **Python 3.12.2** (3.13 is not installed on this machine), torch 2.8.0+cpu,
gymnasium 1.4.0, pettingzoo 1.27.0. Timings on this laptop vary roughly ±40 % between runs (thermal / power throttling).

## 1. Architecture (as implemented)

| Part | File | What it does |
|---|---|---|
| Grid | `environment/grid.py` | 10×10. `kiva_standard`: shelf pods at x∈{2,3,6,7}, y∈2..6; 2 stations on bottom row. `narrow_corridor`: wall across the middle with one gap, 1 station. Shelves are obstacles; robots pick from an adjacent "access cell". |
| Orders | `environment/order.py` | Poisson(λ) arrivals per step; 1 item per order; first 20 % of the **sorted** shelf list gets 80 % of the probability; random station. |
| Robot | `environment/robot.py` | Status IDLE → EN_ROUTE_TO_SHELF → CARRYING_TO_STATION. |
| Env | `environment/warehouse_env.py` | PettingZoo ParallelEnv. **Task allocation is not learned**: `_auto_dispatch_orders` gives each idle robot its nearest pending order. RL controls movement only. Episode = `max_steps`, truncation only. |
| Obs (28) | `_get_obs` | own pos, status, target pos, distance; 4 wall flags; 3 nearest robots (dx, dy, carrying, near); unit vector to target; nearest-station offset; neighbour count; wait counter. |
| State (150) | `state()` | 5×28 obs concatenated + 10 global slots (only 8 are filled). |
| Baseline | `baselines/` | Same nearest-robot dispatch (from the env) + Space-Time A* with a reservation table. |
| IPPO | `agents/ippo_agent.py` | Shared actor + shared local critic, one Adam optimiser. |
| MAPPO | `agents/mappo_agent.py` | Shared actor + centralized critic V(s) → **one** number, copied to all 5 robots. |

### Reward per robot per step (`warehouse_env.py:238-273`)

| Term | Value | When |
|---|---|---|
| time | −0.02 | every step |
| collision | −5.0 | robot was blocked (wall/shelf/out-of-bounds bump, vertex or swap conflict). Only the **mover** is penalised. |
| progress shaping | +0.2 × (d_prev − d_now) | Manhattan distance to target (ignores obstacles; no γ factor) |
| idle | −0.05 | has a target **and chose action 4 (Wait)** |
| pickup | +3.0 | standing on access cell, or action 5 when adjacent |
| delivery | +10.0 | standing on station |

## 2. Reproduction of the committed eval (`evaluate.py` protocol: seeds 100–109, λ = 0.25, 300 steps)

| Method | Committed CSV thru / coll | Re-run A | Re-run B | Seeded re-run (torch seed 0) |
|---|---|---|---|---|
| Baseline | 46.2 / 10.3 | 46.2 / 10.3 | 46.2 / 10.3 | 46.2 / 10.3 |
| IPPO | 4.4 / 37.5 | 3.9 / 49.5 | 5.2 / 47.5 | 5.6 / 33.5 |
| MAPPO | 3.1 / 34.8 | 3.5 / 25.1 | 4.0 / 24.8 | 3.6 / 36.7 |

The baseline reproduces exactly. The RL rows change on every run because policy sampling during evaluation is not seeded.
**The conclusion holds: RL delivers about 10× fewer orders than the baseline.**

## 3. Speed on this CPU (env steps / second, 1 env step = 5 robot transitions)

| | kiva_standard | narrow_corridor |
|---|---|---|
| env only, random actions | 557–873 | 824 |
| env + IPPO actor inference | 122–127 | 113 |
| env + MAPPO actor + state() | 90 | 92 |
| env + A* baseline | 153 | 91 |

PPO update (256 steps × 5 robots, 4 epochs, batch 64): 3.5 s at the default torch threads, **2.4 s with
`torch.set_num_threads(1)`**, 8.9 s with 4 threads (more threads means slower on these tiny networks).
Effective training speed ≈ 41–47 env steps/s, so **1 M env steps ≈ 6–7 h per run** with the current code.
Profile: ~3.6 ms per step in the env (mostly `_get_obs`), ~6.6 ms in torch overhead (Categorical sample/log_prob
~3.6 ms, tiny `Linear` calls ~2 ms). Batched inference over several envs should recover most of that.

## 4. Findings (ranked by likely impact) – evidence from 10 diagnostic episodes, seeds 9000–9009

| # | Finding | Evidence | Hypothesis status |
|---|---|---|---|
| F1 | **Reward loophole: action 5 (Interact) is a free "wait".** The −0.05 idle penalty applies only to action 4. Action 5 almost never does anything (pickup is automatic on arrival). | Trained IPPO picks Interact **77 %** of the time and MAPPO **72 %**. Only 34/11 478 and 29/10 732 Interact actions changed the outcome. Movement actions are 2–4 % each. | **New finding**. The policies learned "stand still". |
| F2 | **Early training punishes all movement.** | Random policy: 207 wall bumps + 46 robot bumps per episode over 1 500 robot-steps, so about 17 % of steps carry −5. That is an expected −0.85 per step vs at most +0.2 shaping, so standing still is the rational first thing to learn. 82 % of random-policy collisions are wall bumps. | **Masking hypothesis: confirmed for exploration.** For trained policies, wall bumps are only 5–7 % of collisions (they have already learned to stop moving), so masking alone will not fix the final collision count. |
| F3 | **Physics bug: two robots can occupy one cell.** Vertex conflicts are grouped once before any revert. If B is blocked, a follower A that targeted B's cell still moves in. | Unit check: A(0,0)→E, B(1,0)→E, C(3,0)→W gives A and B both at (1,0). Episodes with ≥1 overlap step: IPPO 163 steps, MAPPO 11, baseline 4, random 3. | **New finding, confirmed** (`tests/test_phase0_audit.py`, strict xfail). |
| F4 | **GAE episode-boundary off-by-one.** Training loops store the post-step `done` at index t; `buffer.py` reads `dones[t+1]`. Advantages leak across episode resets, and the step before each terminal one gets cut. | Synthetic buffer: advantage at the terminal step = **4.95** (should be 0). | **New finding, confirmed** (strict xfail test). |
| F5 | **Training budget is tiny, and the CLI ignores the config.** `--timesteps` defaults to 15 000 and always overrides YAML's 50 000. | 15 000 / 400 = **37.5 episodes**; 50 000 → 125 episodes, 195 updates. README tells you to use 20 000 (= 50 episodes). The committed checkpoints do not record their budget. | **Confirmed.** "Timesteps" = env steps (consistent between scripts), = 5× robot transitions. |
| F6 | **Value targets unnormalised + value clip 0.2.** | Per-robot discounted return std = 16–27 (range −174…+79). The clip of 0.2 (shared with the policy clip) is about 1 % of the return scale, so the critic can barely move per update. | Plausible cause, **to test** in a Phase 1 ablation. |
| F7 | **MAPPO critic/reward mismatch.** One V(s) for all robots; rewards are per robot. | Between-robot share of return variance at the same time step: **87–88 %** (RL policies), 41 % (baseline). A single fleet value cannot fit most of the target variance. | **Confirmed** as a mismatch; its effect on learning is still to test. |
| F8 | **Evaluation protocol issues.** | Unseeded sampling (§2). `evaluate.py` hardcodes 300 steps (training uses 400), λ 0.25 (training uses 0.2), seeds 100.. (config `evaluation.seeds` unused). "Best" checkpoint = best mean of the last 5 *training* episodes (noisy, and selected on training data). One training seed per method. | **Confirmed.** |
| F9 | **Default eval load is above capacity.** | λ 0.25 × 300 = 76 orders spawned; baseline delivers 45.5 and leaves 25.8 pending; robots idle only 5–6 % of steps. | Fact. Throughput is capacity-limited, so the scenario matrix should include a load where it is not. |
| F10 | **Collision metric conflates kinds.** Wall bumps, robot–robot conflicts and "walked into a standing robot" all count the same; only the mover is penalised. | See F2. | Fact. Report separate counters. |
| F11 | Shaping uses Manhattan distance (ignores shelves/walls) and is not γ-consistent potential shaping. | Code read. | Minor; to test. |
| F12 | Obs/state details: obs already in [−1, 1] with no constant dims (no gross scaling bug), but there is no queue/time info. Global-state slots 8 and 9 are always 0. Idle robots get target = own position. | Diagnostics. | Obs **normalisation is a weaker suspect than expected**. Reward scale (F6) matters more. |
| F13 | Minor: popular shelves = first 20 % of sorted coordinates = left shelf columns (fixed spatial bias); initial placement in `__init__` may use station cells but `reset` excludes them; redundant delivery condition (`warehouse_env.py:268`). | Code read. | Minor. |

### What is *not* the main problem (so far)
* Obs scaling: values are bounded and informative (F12).
* "Small dataset": not a factor. Experience comes from the simulator. The budget (F5) and the reward/physics bugs (F1–F4) are.
