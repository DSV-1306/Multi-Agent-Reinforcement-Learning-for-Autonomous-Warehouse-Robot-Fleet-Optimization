"""
Phase 3 tests: Online Retail II cleaning/statistics, empirical order mode, MAPF map loading/derivation,
held-out layout loading, trip-length rule, A* horizon scaling. Uses tiny synthetic inputs, so the raw
datasets are not needed.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from baselines.heuristic_agent import HeuristicFleetController, BASE_TIME_HORIZON
from data.mapf_layouts import read_map, derive_layout
from data.preprocess_online_retail import clean, compute_stats
from environment.grid import WarehouseGrid
from environment.order import OrderGenerator
from environment.warehouse_env import WarehouseEnv
from training.config_utils import load_config, env_kwargs_from_cfg

OPTS = {"top_n_skus": 3, "max_lines_per_invoice": 2, "drop_non_product_codes": True, "drop_nonpositive_price": True}


def _retail_df():
    rows = [
        # Invoice, StockCode, Quantity, Price, InvoiceDate
        ("1", "85123A", 1, 1.0, "2010-01-04 09:10"),
        ("1", "22423", 2, 1.0, "2010-01-04 09:10"),
        ("1", "21232", 1, 1.0, "2010-01-04 09:10"),   # 3 lines -> capped to 2
        ("2", "85123A", 1, 1.0, "2010-01-04 13:00"),
        ("C3", "85123A", -1, 1.0, "2010-01-04 14:00"),  # cancellation
        ("4", "22423", 0, 1.0, "2010-01-04 15:00"),     # quantity <= 0
        ("5", None, 1, 1.0, "2010-01-04 15:00"),        # missing code
        ("6", "POST", 1, 18.0, "2010-01-04 15:00"),     # non-product code
        ("7", "22423", 1, 0.0, "2010-01-04 15:00"),     # price <= 0
        ("2", "85123A", 1, 1.0, "2010-01-04 13:00"),    # exact duplicate
    ]
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_online_retail_cleaning_rules():
    df, log = clean(_retail_df(), OPTS)
    assert set(df["Invoice"]) == {"1", "2"}
    assert log["rows_raw"] == 10 and log["rows_after_dedup"] == 9 and len(df) == 4


def test_online_retail_statistics():
    df, _ = clean(_retail_df(), OPTS)
    st = compute_stats(df, OPTS)
    assert st["sku_codes"][0] == "85123A" and st["sku_line_counts"][0] == 2
    assert pytest.approx(sum(st["sku_probs"])) == 1.0   # all 3 SKUs kept
    assert st["lines_pmf"] == [0.5, 0.5]                 # invoice 2: 1 line; invoice 1: 3 -> cap 2
    assert st["hourly_profile"][9] == 1 and st["hourly_profile"][13] == 1 and sum(st["hourly_profile"]) == 2


def _empirical(n_shelves=4):
    return {"sku_probs": [0.4, 0.3, 0.2, 0.1, 0.05][:n_shelves + 1], "lines_pmf": [0.5, 0.0, 0.5],
            "hourly_profile": [0.0] * 8 + [1.0, 3.0] + [0.0] * 14, "steps_per_hour": 10,
            "start_hour": 8, "shelf_assignment_seed": 0}


def test_empirical_order_generator_rates_and_shelves():
    grid = WarehouseGrid(10, 10, "kiva_standard")
    shelves = sorted(grid.shelves)[:4]
    gen = OrderGenerator(shelves, grid.stations, grid.get_shelf_access_cells, lambda_rate=0.5, seed=0,
                         max_steps=20, mode="empirical", empirical=_empirical())
    assert pytest.approx(gen.shelf_weights.sum()) == 1.0
    assert sorted(np.round(gen.shelf_weights, 4)) == sorted(np.round(np.array([0.4, 0.3, 0.2, 0.1]) / 1.0, 4))
    assert gen.mean_lines == pytest.approx(2.0)
    # orders are generated at steps 0..max_steps; the time-of-day profile averages to lambda over them
    assert np.mean([gen.current_lambda(t) for t in range(21)]) == pytest.approx(0.5)
    assert gen.current_lambda(15) == pytest.approx(3 * gen.current_lambda(0))  # hour 9 is 3x hour 8
    lam15 = gen.current_lambda(15)
    lines = []
    for _ in range(4000):
        gen.reset()
        lines.append(len(gen.generate_orders_for_step(15)))
    assert np.mean(lines) == pytest.approx(lam15, rel=0.1)  # expected order LINES per step = lambda_t
    invoice = [o for o in gen.generate_orders_for_step(15)]
    assert len({o.station_pos for o in invoice}) <= 1 or len(invoice) == 0


def test_empirical_mode_validates_inputs():
    grid = WarehouseGrid(10, 10, "kiva_standard")
    with pytest.raises(ValueError):
        OrderGenerator(sorted(grid.shelves), grid.stations, grid.get_shelf_access_cells, mode="empirical",
                       empirical=_empirical())  # 20 shelves but only 5 SKUs
    with pytest.raises(ValueError):
        OrderGenerator(sorted(grid.shelves), grid.stations, grid.get_shelf_access_cells, mode="bogus")


def test_synthetic_mode_unchanged_by_refactor():
    # same draws as before Phase 3: 80/20 weights and Poisson(lambda) arrivals
    grid = WarehouseGrid(10, 10, "kiva_standard")
    gen = OrderGenerator(list(grid.shelves), grid.stations, grid.get_shelf_access_cells, lambda_rate=0.2, seed=7)
    n_pop = int(0.2 * len(grid.shelves))
    assert gen.shelf_weights[:n_pop].sum() == pytest.approx(0.8)
    assert gen.current_lambda(123) == 0.2


MAP_TEXT = """type octile
height 7
width 12
map
TTTTTTTTTTTT
T..........T
T....TTT.T.T
T....TTT...T
T..........T
T.........TT
TTTTTTTTTTTT
"""


def test_mapf_read_and_derive(tmp_path):
    p = tmp_path / "m.map"
    p.write_text(MAP_TEXT)
    obst = read_map(str(p), "@T")
    assert obst.shape == (7, 12) and obst[0].all() and not obst[1, 1]
    lay = derive_layout(obst, "m", crop_w=11, crop_h=7, open_margin=2, num_stations=2)
    # first inner obstacle column is x=5 -> x0 = 5 - 2 = 3
    assert lay["source_crop"]["x0"] == 3 and lay["width"] == 9
    shelves = {tuple(c) for c in lay["shelves"]}
    walls = {tuple(c) for c in lay["walls"]}
    assert (3, 2) in shelves and (4, 2) in shelves      # obstacle block cells next to free cells
    assert all(y in (0, 6) or (x + 3) == 11 or (x, y) in walls for (x, y) in walls)
    assert not any(y in (0, 6) for _, y in shelves)     # border cells are never shelves
    assert len(lay["stations"]) == 2 and all(x == 0 for x, _ in lay["stations"])
    grid = WarehouseGrid(spec=lay)
    assert grid.mean_trip_length() > 0


def test_mapf_layout_loads_through_config(tmp_path):
    lay = derive_layout(read_map(_write(tmp_path, MAP_TEXT), "@T"), "tiny", 11, 7, 2, 2)
    (tmp_path / "tiny.json").write_text(json.dumps(lay))
    cfg = load_config("configs/default_config.yaml", [f"environment.layouts_dir={tmp_path.as_posix()}"])
    kw = env_kwargs_from_cfg(cfg["environment"], layout_type="mapf:tiny", num_robots=2)
    env = WarehouseEnv(**kw)
    env.reset(seed=0)
    assert env.grid_width == lay["width"] and env.grid.stations == [tuple(s) for s in lay["stations"]]
    env.step({a: 4 for a in env.agents})


def _write(tmp_path, text):
    p = tmp_path / "x.map"
    p.write_text(text)
    return str(p)


def test_bfs_trip_length_on_training_layout():
    g = WarehouseGrid(10, 10, "kiva_standard")
    d = g.bfs_distances(g.stations[0])
    assert d[g.stations[0]] == 0 and all(g.is_walkable(*c) for c in d)
    assert g.mean_trip_length() == pytest.approx(8.0)


def test_astar_horizon_scales_only_on_large_grids():
    small = HeuristicFleetController(WarehouseEnv(num_robots=2))
    assert small.max_time_horizon == BASE_TIME_HORIZON
    big = WarehouseEnv(num_robots=2, grid_width=60, grid_height=40)
    assert HeuristicFleetController(big).max_time_horizon == 200


@pytest.mark.skipif(not os.path.exists("data/processed/online_retail_stats.json"),
                    reason="run data/preprocess_online_retail.py first")
def test_cached_stats_and_empirical_env_from_config():
    cfg = load_config("configs/default_config.yaml")
    kw = env_kwargs_from_cfg(cfg["environment"], order_mode="empirical")
    env = WarehouseEnv(**kw)
    env.reset(seed=1)
    for _ in range(50):
        env.step({a: 4 for a in env.agents})
    assert env.order_generator.mode == "empirical"
