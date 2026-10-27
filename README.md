# Multi-Agent RL for Task Allocation and Collision-Free Routing of Autonomous Warehouse Robots

A grid-world warehouse simulator (PettingZoo `ParallelEnv`), a classical baseline (nearest-robot allocation +
Space-Time A* with a reservation table) and three learning methods:

| Method | What is learned | Movement |
|---|---|---|
| **IPPO** | per-robot movement policy (shared weights), local critic | RL |
| **MAPPO** | same actor, centralized critic V(s, o_i) ("agent-specific global state") | RL |
| **Hierarchical** | which pending order each idle robot takes (RL allocation) | Space-Time A* (unchanged) |

The comparison is run under a fixed protocol: training on own layouts with synthetic orders, tuning on dev
seeds, and a final **held-out test on layouts derived from public MAPF benchmark maps with order statistics
from a real online retailer** (a distribution-shift test). All numbers in this README come from the scripts
in this repository run on a 2-core laptop CPU; see `results/report/FINAL_REPORT.md` for the full write-up.

**Results in one table** (orders delivered per episode, mean ± 95% CI over 3 training seeds; Δ = paired
difference vs the baseline; full tables in `results/report/FINAL_REPORT.md`):

| | Original repo | Dev: own layouts, 6 scenarios | Held-out test: MAPF layouts + real order stats |
|---|---|---|---|
| Baseline (nearest + Space-Time A*) | 46.2 | 39.3 ± 1.9 | 12.1 ± 4.4 |
| IPPO | 4.4 | 32.6 ± 4.5 (Δ -6.8 ± 4.5) | 5.8 ± 1.3 (Δ -6.3 ± 1.3) |
| MAPPO | 3.1 | 28.2 ± 7.0 (Δ -11.2 ± 7.0) | 4.9 ± 4.3 (Δ -7.2 ± 4.3) |
| Hierarchical (RL allocation + A*) | - | 39.6 ± 0.3 (Δ +0.3 ± 0.3) | 12.0 ± 0.4 (Δ -0.1 ± 0.4) |

The original column used a different protocol (one layout, lambda 0.25, 300 steps) and is not comparable
row-for-row; it shows the starting point. **Honest summary:** the original RL results were broken (policies
learned to stand still; two simulator/GAE bugs). Fixed, RL ties the planner on the open layout and a
single-layout IPPO specialist beat it there (74.2 vs 61.3, one seed), but RL loses in the bottleneck layout and
does not survive the held-out distribution shift. RL allocation on top of A* matches but does not beat
greedy allocation.

---

## 1. Repository layout

```text
environment/   grid.py (layouts incl. MAPF-derived), robot.py, order.py (synthetic + empirical orders),
               warehouse_env.py (PettingZoo env: masks, collision resolution, rewards from YAML), gym_wrapper.py
baselines/     astar.py (Space-Time A* + reservation table), heuristic_agent.py (nearest allocation + A*)
agents/        models.py (masked actor, critics), ippo_agent.py, mappo_agent.py, hier_agent.py (RL allocator)
training/      trainer.py (shared PPO loop), train_ippo.py, train_mappo.py, train_hier.py, buffer.py (GAE),
               normalization.py, vec_env.py, config_utils.py (YAML + --set overrides), run_queue.py (Phase 5 jobs)
evaluation/    evaluate.py, scenario_matrix.py, stats.py (95% CIs), plot_learning_curves.py, plot_results.py,
               metrics.py, phase0_audit.py
data/          README.md (download + protocol), check_datasets.py, preprocess_online_retail.py, mapf_layouts.py,
               processed/ (small cached statistics), layouts/ (held-out layouts as JSON)
configs/       default_config.yaml - every setting, seed set and experiment
results/       phase0/ (audit), runs/ (training curves, summaries), eval/ (CSV + markdown tables), plots/, report/
tests/         61+ pytest tests
```

## 2. Setup

Tested with **Python 3.12.2** on Windows 11, CPU only (torch 2.8.0+cpu). Python 3.13 should work but was not
available on the test machine.

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

New dependencies added in this revision: `openpyxl` (read the Online Retail II spreadsheet once) and `scipy`
(t-distribution confidence intervals). Windows note: the project path on the development machine was ~200
characters long, close to Windows' 260-character limit; keep the repository in a short path (e.g. `C:\marl`).

## 3. Data (realism only - RL experience comes from the simulator)

See `data/README.md`. Raw files are never committed; preprocessing writes only small JSON/CSV files.

| Dataset | Used for |
|---|---|
| Online Retail II (UCI id 502, D. Chen, CC BY 4.0) | `empirical` order mode: SKU popularity (top-1000 SKUs = 68.9% of 1,003,214 clean order lines), invoices per hour, lines per invoice (mean 25.1) |
| MAPF benchmark `warehouse-*` maps (Stern et al., SoCS 2019) | 4 held-out test layouts (30x20 and 46x33 crops, shelves/stations placed by a fixed rule) |

## 4. Environment

* 10x10 own layouts `kiva_standard` (2 stations) and `narrow_corridor` (1 station, one-cell gap).
* Observation: 28 numbers per robot (own state, target, wall flags, 3 nearest robots, congestion).
* Actions: N, S, E, W, Wait, Interact. **Action masking** removes moves into shelves/walls/outside the grid and
  Interact when it cannot pick anything up (other robots are not masked; avoiding them is learned).
* A "collision" is a **blocked move**: the simulator never lets two robots share a cell or swap, it keeps the
  robot in place and counts the event (robot-robot and wall bumps are counted separately).
* Rewards (per robot, `environment.rewards` in the YAML): -0.02 per step, -5 per blocked move, +0.2 x progress
  (Manhattan), -0.05 for waiting with a task, +3 pickup, +10 delivery.

## 5. How to reproduce everything

```bash
# Phase 0: audit of the original code (reproduces the committed numbers, measures speed)
python evaluation/phase0_audit.py
# Phase 3: data (needs the raw files, see data/README.md)
python data/check_datasets.py
python data/preprocess_online_retail.py
python data/mapf_layouts.py --show
# Single training runs
python training/train_ippo.py --seed 1
python training/train_mappo.py --seed 1
python training/train_hier.py --seed 1
# Phase 5: all methods + ablations x 3 seeds (resumable; ~4-5 h on a 2-core i3)
python training/run_queue.py
# Phase 5 evaluation (see results/report/FINAL_REPORT.md for the exact list of commands used)
python evaluation/scenario_matrix.py --matrix own --split dev --runs results/runs/p5/* --output results/eval/p5_own_dev.csv
python evaluation/stats.py --input results/eval/p5_own_dev.csv --output results/eval/p5_own_dev_summary.csv
```

Any setting can be overridden from the command line, e.g. `--set training.action_masking=false`.

## 6. Evaluation protocol

| Split | Layouts | Orders | Seeds | Used for |
|---|---|---|---|---|
| Train | kiva_standard, narrow_corridor | synthetic, lambda ~ U(0.1, 0.3) | env seeds >= 10000 | learning |
| Dev | same, low/medium/high load | synthetic | 5001-5010 | curves, checkpoint selection, ablations, all design decisions |
| Test | 4 MAPF-derived layouts x {5, 10} robots | empirical (Online Retail II) | 101-120 | final comparison only, run once |

Confidence intervals: RL variants use the training seed as the unit (3 seeds, Student-t, so intervals are wide);
the baseline uses evaluation episodes. Differences to the baseline are paired on the same evaluation seed.
Test order rates follow a geometry rule fixed before testing (`evaluation/scenario_matrix.py`).

## 7. Results and figures

* Report with every table, the audit, design decisions and limitations: `results/report/FINAL_REPORT.md`
* Slide text: `results/report/SLIDES.md`
* Figures: `results/plots/p1_curves.png` (Phase 1 learning curves), `p2_curriculum_diag.png` (curriculum
  collapse), `p5_curves_main.png`, `p5_curves_ablation_ippo.png`, `p5_own_dev_main.png`, `p5_ablations.png`,
  `p5_test.png`
* Tables: `results/eval/p5_own_dev_summary.md` (dev + ablations), `results/eval/p5_test_summary.md` (test)

Ablations on dev (3 seeds, all 6 scenarios): IPPO 32.6 ± 4.5; without action masking 17.2 ± 13.0; without
observation/reward normalisation 4.7 ± 3.2; with the 2->3->5-robot curriculum 25.7 ± 9.0; MAPPO with the
original single fleet-value critic 7.0 ± 4.5 vs 28.2 ± 7.0 with the per-robot critic.

## 8. Limitations

Grid world with unit moves; a "collision" is a blocked move; 3 training seeds (wide CIs); 500k-step budget
(IPPO/MAPPO curves were still rising); no per-method hyperparameter search; one item per order; empirical
statistics from one retailer; 4 test maps from one benchmark family; tested on Python 3.12.
