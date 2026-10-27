# Multi-Agent RL for Task Allocation and Collision-Free Routing of Warehouse Robots: Final Report

Every number below was produced by the scripts in this repository on one laptop (Intel i3-1005G1,
2 cores / 4 threads, 7.8 GB RAM, Windows 11, Python 3.12.2, torch 2.8.0+cpu). The "Provenance" section
lists the file behind each table. 95% confidence intervals (CIs) for RL use the **training seed** as the
unit (3 seeds, Student-t with 2 degrees of freedom, so the intervals are wide by construction). The baseline
uses evaluation episodes. "Δ" is the paired difference to the baseline on the same evaluation seeds.

## 1. Executive summary

1. **The original RL results (IPPO 4.4, MAPPO 3.1 orders/episode vs baseline 46.2) were caused by bugs and
   design problems, not by RL being unsuitable.** The trained policies had learned to stand still: action 5
   ("Interact") was a penalty-free wait and was chosen 72-77% of the time. Two real bugs (two robots could
   share a cell; GAE leaked advantages across episodes) and a tiny budget (the CLI silently used 37.5 episodes)
   made it worse.
2. After the fixes, a single-seed IPPO **specialist** trained on one layout/load (1M env steps) delivered
   74.2 orders/episode on dev seeds vs the baseline's 61.3, with lower latency (28.4 vs 44.1 steps) but more
   blocked moves (23.1 vs 9.6). Single seed: indicative only.
3. With the full protocol (one policy for both own layouts and all loads, 500k env steps, 3 seeds), RL movement
   policies **do not beat the classical baseline**: IPPO ties it on the open `kiva_standard` layout
   (Δ -1.2 ± 4.2 at medium load) but loses clearly on the one-gap `narrow_corridor` layout (Δ -12.6 ± 2.5),
   and over all six dev scenarios IPPO is lower (32.6 ± 4.5 vs 39.3 ± 1.9; Δ -6.8 ± 4.5). MAPPO is lower
   still (28.2 ± 7.0). The learning curves were still rising at 500k steps, so this is a budget-limited result.
4. The **hierarchical method** (RL decides allocation, Space-Time A* moves) **matches** the baseline
   (39.6 ± 0.3 vs 39.3 ± 1.9; Δ +0.3 ± 0.3) with the same latency. It learned to reproduce greedy-nearest
   allocation; it did not find a better rule.
5. **Ablations** (3 seeds each, dev): every Phase 1/2 design choice matters. Removing reward/observation
   normalisation collapses IPPO (4.7 ± 3.2), removing action masking roughly halves it (17.2 ± 13.0), adding the
   2->3->5-robot curriculum hurts (25.7 ± 9.0), and MAPPO with the original single fleet-value critic
   collapses (7.0 ± 4.5 vs 28.2 ± 7.0 with the per-robot critic).
6. **Held-out test** (4 MAPF-benchmark layouts x {5, 10} robots + Online Retail II order statistics, run once):
   **no RL movement advantage survives the distribution shift.** IPPO delivers 5.8 ± 1.3 orders/episode vs the
   baseline's 12.1 ± 4.4 (Δ -6.3 ± 1.3) and MAPPO 4.9 ± 4.3 (Δ -7.2 ± 4.3), with about twice the blocked
   moves. The hierarchical method keeps baseline throughput (12.0 ± 0.4, Δ -0.1 ± 0.4) and has fewer
   robot-robot blocked moves (29.2 vs 39.9, Δ -10.7 ± 0.4, mostly on the 10-robot maps); on the dev set
   this difference was not present (Δ +1.7 ± 3.0), so it is a test-only observation.

## 2. Phase 0: audit of the original code

Reproduction of the committed evaluation (`results/phase0/repro_eval.csv`): the baseline reproduces exactly
(46.2 orders, 10.3 collisions); IPPO/MAPPO vary run to run (3.9-5.6 and 3.1-4.0) because policy sampling was
not seeded.

| # | Finding | Evidence |
|---|---|---|
| F1 | Action 5 was a penalty-free wait (idle penalty only on action 4) | trained IPPO/MAPPO chose it 77% / 72% of the time; only 34/11,478 and 29/10,732 uses had any effect |
| F2 | Random exploration is punished on ~17% of robot-steps (82% of them wall bumps): standing still is the rational first policy | random-policy diagnostics |
| F3 | **Bug**: vertex conflicts resolved in one pass, so two robots could end in one cell | 163 overlap steps in 10 IPPO episodes; unit test |
| F4 | **Bug**: GAE used `done[t+1]` with post-step done flags, leaking advantages across episodes | synthetic buffer: advantage 4.95 instead of 0 |
| F5 | Training budget: `--timesteps` defaulted to 15,000 and overrode the YAML (= 37.5 episodes) | code |
| F6 | Unnormalised returns (std 16-27) with a value clip of 0.2 | diagnostics |
| F7 | MAPPO critic outputs one fleet value but rewards are per robot | 87-88% of return variance is between robots at the same time step |
| F8 | Evaluation used different max_steps/lambda than training, unseeded sampling, and selected checkpoints on training reward | code |

Speed: ~45 env steps/s effective training speed (1M steps ≈ 6-7 h), dominated by per-step torch overhead.

## 3. Phase 1: training fixes

| Change | Why (RL reasoning) |
|---|---|
| Action masking (walls/shelves; Interact only when useful) | removes actions that can only be punished, so exploration is not dominated by -5 penalties (F2) |
| No-effect Interact gets the idle penalty | closes the reward loophole (F1) |
| Iterative collision resolution; separate wall/robot counters | fixes F3; honest collision metric |
| GAE with the post-step done convention; bootstrap V(s_final) on time-limit truncation | fixes F4; truncation is the clock, not a terminal state |
| Running observation normalisation; reward scaling by the std of discounted returns; value clipping off | well-conditioned critic targets (F6) |
| MAPPO critic V(s, o_i) ("agent-specific global state", Yu et al. 2022) | one value per robot for per-robot rewards (F7) |
| 16 envs in one process with batched inference, 1 torch thread | training speed 45 -> 400-700 env steps/s |
| YAML-driven trainer, `--set` overrides, disjoint train/dev/test seeds, seeded evaluation, dev-based checkpoint selection, learning-curve CSV | fair, reproducible protocol (F5, F8) |

Single-seed results on dev seeds (kiva_standard, lambda = 0.2, 1M env steps; `results/eval/p1_dev.csv`):

| Method | Orders / episode | Robot-robot blocked moves | Mean latency (steps) |
|---|---|---|---|
| Baseline (A*) | 61.3 | 9.6 | 44.1 |
| IPPO | 74.2 | 23.1 | 28.4 |
| MAPPO | 73.7 | 21.4 | 31.1 |

Action mix of the fixed IPPO policy: N 23%, S 23%, E 18%, W 19%, Wait 15%, Interact 2% (was 77% Interact).

## 4. Phase 2: curriculum and scenario matrix

Scenario matrix (dev seeds): 2 own layouts x 3 loads (lambda = 0.1 / 0.2 / 0.3 orders per step, 5 robots).
Training draws lambda ~ U(0.1, 0.3) every episode, and the 16 envs alternate the two layouts.

**The curriculum (2 -> 3 -> 5 robots, low -> high load) hurt.** Dev diagnostic (IPPO, seed 1, 250k steps,
`results/plots/p2_curriculum_diag.png`): with the curriculum, kiva dev throughput reached 47.5 at 50k steps and
then fell to 11.2 after the switch to 5 robots. Training collisions jumped to 112 per episode, the policy
switched to stopping, and entropy collapsed from 0.82 to 0.19 (it then barely explores and does not recover).
The reward scale did not change between stages (16-21), so a normalisation shock is not the explanation.
Without the curriculum: 52.2 (kiva only) and 37.8 (both layouts). Decision on dev: main configuration
without the curriculum; the curriculum became an ablation (Section 6), and its 3-seed result agrees.

## 5. Phase 3: real-data realism

* **Online Retail II** (1,067,371 rows -> 1,003,214 order lines after removing duplicates, cancellations,
  Quantity <= 0, missing / non-product StockCodes and Price <= 0; 39,516 invoices, 4,707 SKUs, Dec 2009 - Dec
  2011). Top 1,000 SKUs = 68.9% of order lines. Mean 25.1 lines per invoice (19.3 when capped at 50).
  Invoices peak at 12:00; 08:00-18:00 covers almost all of them.
* **Empirical order mode**: shelf popularity = top-N SKU shares mapped to shelves by a seeded permutation;
  invoices arrive as Poisson(lambda_t / E[lines]) and spawn one order per line (expected lines per step =
  lambda_t, so load matches the synthetic mode but is much burstier); lambda_t follows the hourly profile
  (episode = 08:00-18:00, 40 steps per hour).
* **MAPF layouts**: four warehouse benchmark maps cropped to 30x20 (two maps) and 46x33 (two maps) by one rule
  (keep 6 open columns before the first shelf block; obstacles next to free cells become shelves; border and
  interior obstacles become walls; largest connected region; 3 stations on the left edge): 180-777 shelves.
* Test order rate rule (fixed before testing): lambda = 0.2 x (robots / 5) x (8.0 / mean trip length of the
  layout), giving lambda = 0.042-0.133.

## 6. Phases 4-5: methods, multi-seed results and ablations (DEV seeds)

**Hierarchical method.** At exactly the moments the baseline's nearest-order rule runs (checked: plugging a
greedy-nearest allocator into the same hook reproduces the baseline's numbers exactly), a PPO policy picks
one of up to 8 candidate orders (4 nearest + 4 oldest) for each idle robot from 7 per-candidate features.
Space-Time A* does all movement. Two problems were found and fixed on dev before the final runs: a shared
gradient clip that let the critic's large gradient swamp the policy (approx_kl was 0), and too few optimizer
steps (decisions are rare). The fleet-level semi-MDP reward gave almost no learning signal (dev latency 100
-> 95 steps over training), so the final reward is per assignment: minus the time until that order is
delivered (a contextual bandit).

Own layouts, dev seeds, 3 training seeds per method (`results/eval/p5_own_dev_summary.md`):

| Scenario | Baseline | IPPO | MAPPO | Hierarchical |
|---|---|---|---|---|
| kiva / low | 34.8 ± 3.1 | 34.5 ± 7.6 (Δ -0.3 ± 7.6) | 34.9 ± 4.9 (Δ +0.1 ± 4.9) | 34.6 ± 0.6 (Δ -0.2 ± 0.6) |
| kiva / medium | 61.3 ± 5.1 | 60.1 ± 4.2 (Δ -1.2 ± 4.2) | 49.3 ± 19.8 (Δ -12.0 ± 19.8) | 61.6 ± 0.4 (Δ +0.3 ± 0.4) |
| kiva / high | 65.2 ± 2.7 | 56.9 ± 20.4 (Δ -8.3 ± 20.4) | 42.8 ± 23.6 (Δ -22.4 ± 23.6) | 67.3 ± 1.7 (Δ +2.1 ± 1.7) |
| corridor / low | 23.0 ± 1.5 | 16.1 ± 2.5 (Δ -6.9 ± 2.5) | 13.7 ± 5.1 (Δ -9.3 ± 5.1) | 23.2 ± 0.1 (Δ +0.2 ± 0.1) |
| corridor / medium | 26.5 ± 2.1 | 13.9 ± 2.5 (Δ -12.6 ± 2.5) | 14.6 ± 3.9 (Δ -11.9 ± 3.9) | 26.1 ± 0.4 (Δ -0.4 ± 0.4) |
| corridor / high | 25.3 ± 6.6 | 13.9 ± 2.2 (Δ -11.4 ± 2.2) | 13.8 ± 5.2 (Δ -11.5 ± 5.2) | 24.9 ± 0.4 (Δ -0.4 ± 0.4) |
| **all 6** | **39.3 ± 1.9** | **32.6 ± 4.5 (Δ -6.8 ± 4.5)** | **28.2 ± 7.0 (Δ -11.2 ± 7.0)** | **39.6 ± 0.3 (Δ +0.3 ± 0.3)** |

Orders delivered per episode. Over all six scenarios, robot-robot blocked moves were 23.2 ± 6.9 (baseline),
22.9 ± 0.9 (IPPO), 19.3 ± 8.9 (MAPPO), 24.9 ± 3.0 (hierarchical), and mean latency 66.3, 76.9, 66.8, 67.2
steps. Note that the baseline itself has 65 blocked moves per episode at kiva/low (its reservation table
deadlocks on some seeds), so "collisions" do not favour the planner everywhere.

Ablations (same protocol, all six dev scenarios; Δ vs baseline):

| Variant (one change) | Orders / episode | Δ vs baseline |
|---|---|---|
| IPPO (full) | 32.6 ± 4.5 | -6.8 ± 4.5 |
| IPPO without action masking | 17.2 ± 13.0 | -22.1 ± 13.0 |
| IPPO without obs/reward normalisation | 4.7 ± 3.2 | -34.7 ± 3.2 |
| IPPO with the 2->3->5 curriculum | 25.7 ± 9.0 | -13.7 ± 9.0 |
| MAPPO, per-robot critic V(s, o_i) | 28.2 ± 7.0 | -11.2 ± 7.0 |
| MAPPO, original single fleet value V(s) | 7.0 ± 4.5 | -32.4 ± 4.5 |

Learning curves: `results/plots/p5_curves_main.png`, `results/plots/p5_curves_ablation_ippo.png`.

## 7. Held-out TEST (MAPF layouts + empirical orders, test seeds 101-120)

Order rates come from the geometry rule in Section 5 (lambda = 0.042-0.133); 20 test seeds per scenario;
nothing was tuned on these numbers. Orders delivered per episode (`results/eval/p5_test_summary.md`,
`results/plots/p5_test.png`):

| Layout / robots | Baseline | IPPO | MAPPO | Hierarchical |
|---|---|---|---|---|
| 10-20-10-2-1 / 5 | 8.0 ± 4.9 | 2.7 ± 1.7 (Δ -5.3 ± 1.7) | 2.8 ± 1.4 (Δ -5.2 ± 1.4) | 7.5 ± 0.3 (Δ -0.5 ± 0.3) |
| 10-20-10-2-1 / 10 | 21.6 ± 6.2 | 8.1 ± 3.8 (Δ -13.6 ± 3.8) | 7.3 ± 6.1 (Δ -14.3 ± 6.1) | 21.9 ± 2.1 (Δ +0.2 ± 2.1) |
| 10-20-10-2-2 / 5 | 9.6 ± 5.4 | 4.5 ± 0.6 (Δ -5.0 ± 0.6) | 3.8 ± 2.0 (Δ -5.8 ± 2.0) | 9.1 ± 0.3 (Δ -0.4 ± 0.3) |
| 10-20-10-2-2 / 10 | 22.6 ± 6.4 | 13.1 ± 7.9 (Δ -9.5 ± 7.9) | 9.9 ± 10.5 (Δ -12.7 ± 10.5) | 23.4 ± 1.3 (Δ +0.8 ± 1.3) |
| 20-40-10-2-1 / 5 | 4.0 ± 3.6 | 1.4 ± 0.6 (Δ -2.6 ± 0.6) | 1.4 ± 1.8 (Δ -2.6 ± 1.8) | 3.9 ± 0.2 (Δ -0.0 ± 0.2) |
| 20-40-10-2-1 / 10 | 12.8 ± 5.0 | 5.7 ± 1.6 (Δ -7.1 ± 1.6) | 6.0 ± 5.0 (Δ -6.9 ± 5.0) | 12.6 ± 0.4 (Δ -0.3 ± 0.4) |
| 20-40-10-2-2 / 5 | 4.5 ± 3.7 | 2.6 ± 0.7 (Δ -2.0 ± 0.7) | 1.9 ± 2.2 (Δ -2.7 ± 2.2) | 4.4 ± 0.2 (Δ -0.1 ± 0.2) |
| 20-40-10-2-2 / 10 | 13.7 ± 5.7 | 8.1 ± 1.5 (Δ -5.5 ± 1.5) | 6.3 ± 7.5 (Δ -7.3 ± 7.5) | 13.0 ± 0.3 (Δ -0.7 ± 0.3) |
| **all 8** | **12.1 ± 4.4** | **5.8 ± 1.3 (Δ -6.3 ± 1.3)** | **4.9 ± 4.3 (Δ -7.2 ± 4.3)** | **12.0 ± 0.4 (Δ -0.1 ± 0.4)** |

Robot-robot blocked moves (all 8): baseline 39.9 ± 26.1, IPPO 94.8 ± 48.2 (Δ +54.9 ± 48.2), MAPPO
67.2 ± 40.0 (Δ +27.3 ± 40.0), hierarchical 29.2 ± 0.4 (Δ -10.7 ± 0.4; -43.7 and -33.2 on the two 30x20
maps with 10 robots). Mean latency (all 8): baseline 99.4, IPPO 88.3, MAPPO 99.7, hierarchical 101.5 steps.
**Latency is averaged over delivered orders only**, so IPPO's lower value reflects the few, mostly short
orders it completes (survivorship), not faster service.

Low absolute throughput on the test layouts is expected for every method: trips are 3-5x longer (mean trip
24-38 cells vs 8 on the training layout) and empirical orders arrive in bursts of ~19 lines per invoice.

**Answer to the research question: the RL movement policies' dev-set parity on open layouts does not survive
the shift to new layouts and real order statistics; the planner-based methods (baseline and hierarchical) do.**

## 8. Discussion: why RL does or does not win

* **Where the movement policies lose: bottlenecks.** In `narrow_corridor` all traffic passes one cell. A
  planner that reserves space-time slots serialises robots through the gap; a reactive policy that sees only
  its 3 nearest robots cannot, and the -5 blocked-move penalty pushes it towards waiting. A shared policy
  trained on both layouts also gives up some kiva performance (60.1 vs the 74.2 single-layout specialist).
* **Where they tie or win: open layouts.** On `kiva_standard` the reactive policies keep moving where the
  planner waits for reservations (it waits in 43% of robot-steps), which is why the specialist beat it.
* **Budget.** The 2-layout runs were still improving at 500k steps; the 500k budget was chosen from the
  kiva-only Phase 1 plateau. More budget may close part of the gap; it was not tested.
* **Hierarchical allocation** keeps the planner's routing, so it inherits its reliability; with only local
  features and a per-assignment reward it learned greedy-like assignments. Beating greedy-nearest needs
  information about future demand or other robots' plans, which these features do not contain.
* **Statistical caution.** Three training seeds give wide intervals (t = 4.30); differences with intervals
  that include 0 are not evidence of a difference.

## 9. Limitations

Grid world with unit moves and a blocked-move model of collisions; 3 training seeds; a single hyperparameter
setting per method (no per-method tuning budget); one item per order; a 10-hour shop day compressed into
400 steps; empirical statistics from one UK online retailer; test layouts are crops of 4 maps from one benchmark
family; Python 3.12 instead of 3.13.

## 10. Provenance

| Table / claim | File(s) | Command |
|---|---|---|
| Phase 0 | `results/phase0/audit_report.json`, `repro_eval.csv`, `PHASE0_AUDIT.md` | `python evaluation/phase0_audit.py` |
| Phase 1 table | `results/eval/p1_dev.csv`, `results/runs/{ippo,mappo}_seed1/` | `python evaluation/evaluate.py --split dev --ippo ... --mappo ...` |
| Curriculum diagnostic | `results/runs/diag/{A_cur_kiva,B_nocur_both,D_nocur_kiva}/` | `train_ippo.py` with the overrides in the run's `config.json` |
| Allocator diagnostics | `results/runs/diag/{hfix,hot2,hfl2}/` | `train_hier.py` |
| Data statistics | `data/processed/online_retail_stats.json`, `data/layouts/*.json` | `data/preprocess_online_retail.py`, `data/mapf_layouts.py` |
| Dev matrix + ablations | `results/eval/p5_own_dev_{rl,hier,mg3}.csv`, `p5_own_dev_summary.{csv,md}` | `scenario_matrix.py --matrix own --split dev`, `stats.py` |
| Test | `results/eval/p5_test_{base,rl,h1,h2,h3}.csv`, `p5_test_summary.{csv,md}` | `scenario_matrix.py --matrix test --split test`, `stats.py` |
| Training runs | `results/runs/p5/<variant>_s<seed>/` (curves, summaries, config) | `python training/run_queue.py` |
