# Slide text (one section per slide). Numbers come from results/report/FINAL_REPORT.md.

## 1. Title
Multi-Agent RL for Task Allocation and Collision-Free Routing of Autonomous Warehouse Robots.
Does learning beat a classical planner, and does it survive a distribution shift?

## 2. Problem
* 5-10 robots move items from shelves to packing stations on a grid; orders arrive randomly.
* Two decisions: which robot takes which order (allocation), and how robots move without blocking each other (routing).
* Baseline: nearest-robot allocation + Space-Time A* with a reservation table.

## 3. Methods
* IPPO: one shared movement policy, local critic.
* MAPPO: same policy, centralized per-robot critic V(s, o_i).
* Hierarchical: RL chooses the order for each idle robot; A* does the moving.

## 4. Audit of the original project (Phase 0)
* Original result: RL 4.4 / 3.1 orders vs baseline 46.2.
* Cause: the policies learned to stand still (a penalty-free "Interact" action, chosen 72-77% of the time).
* Two bugs: two robots could share a cell; GAE leaked advantages across episodes.
* Budget: 37.5 training episodes; 45 env steps/s.

## 5. Fixes (Phase 1)
* Action masking, loophole closed, collision and GAE bugs fixed, observation and reward normalisation,
  per-robot MAPPO critic, 16 parallel envs (≈10x faster), seeded config-driven evaluation, disjoint seed sets.
* Specialist IPPO (one layout, 1 seed): 74.2 orders vs baseline 61.3; latency 28 vs 44 steps; more blocked moves (23 vs 10).

## 6. Protocol (Phases 2-3)
* Train: 2 own layouts, synthetic orders. Dev: 6 scenarios (2 layouts x 3 loads) on separate seeds.
* Test, run once: 4 layouts cut from MAPF benchmark maps + order statistics from Online Retail II
  (1.0M order lines, 39.5k invoices), 5 and 10 robots.
* 3 training seeds per method, mean ± 95% CI, differences paired against the baseline.

## 7. Dev results (3 seeds)
* Baseline 39.3 ± 1.9 orders/episode over 6 scenarios.
* Hierarchical 39.6 ± 0.3: same as the baseline.
* IPPO 32.6 ± 4.5: ties on the open layout, loses in the one-gap corridor (Δ -12.6 ± 2.5).
* MAPPO 28.2 ± 7.0.

## 8. Ablations (3 seeds)
* No normalisation: 4.7 (collapse). No masking: 17.2. With curriculum: 25.7. Full IPPO: 32.6.
* MAPPO single fleet value: 7.0 vs per-robot critic 28.2.
* The curriculum caused entropy collapse when 5 robots arrived.

## 9. Held-out test
* 4 MAPF-derived layouts x 5/10 robots, Online Retail II order statistics, 20 test seeds, run once.
* Baseline 12.1 ± 4.4 orders/episode; hierarchical 12.0 ± 0.4 (tie); IPPO 5.8 ± 1.3 (Δ -6.3 ± 1.3);
  MAPPO 4.9 ± 4.3 (Δ -7.2 ± 4.3).
* Learned movement policies roughly halve throughput and double blocked moves under the shift.
* Hierarchical had fewer blocked moves than the baseline on test (29 vs 40), not seen on dev: treat as tentative.

## 10. Conclusions
* Most of the original "RL loses" was bugs and setup. Fixed RL can match or beat the planner where it was trained.
* Learned movement policies do not transfer to bottlenecks or new layouts; the planner does.
* RL for allocation on top of A* is safe (keeps the planner's routing) but did not beat greedy assignment.
* Honest limits: 3 seeds, 500k steps (curves still rising), grid world, one dataset per source.
