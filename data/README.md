# Data

Reinforcement learning gets its experience from the simulator, so these datasets do **not** train
anything. They only make the simulator more realistic and provide a **held-out test** with a
distribution shift. Raw files stay outside the repository; only small derived files are committed.

| Dataset | Used for | Licence / citation |
|---|---|---|
| Online Retail II (UCI ML Repository, id 502) | `empirical` order mode: SKU popularity, hourly arrival profile, lines per invoice | D. Chen, CC BY 4.0 |
| MAPF benchmark `warehouse-*` maps | Held-out test layouts (cropped, shelves/stations derived by rule) | Stern et al., "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks", SoCS 2019 |

## 1. Download (manual, once)

1. Online Retail II: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>. Download the zip and extract
   `online_retail_II.xlsx` (~43 MB, sheets `Year 2009-2010` and `Year 2010-2011`).
2. MAPF maps: <https://movingai.com/benchmarks/mapf.html>. From the warehouse section, download
   `warehouse-10-20-10-2-1.map`, `warehouse-10-20-10-2-2.map`, `warehouse-20-40-10-2-1.map`,
   `warehouse-20-40-10-2-2.map`.

Put them under one folder and set `data.raw_dir` in `configs/default_config.yaml`:

```
<raw_dir>/online_retail_ii/online_retail_II.xlsx
<raw_dir>/mapf_maps/warehouse-*.map
```

Check that everything is where the config expects it:

```bash
python data/check_datasets.py
```

## 2. Preprocess (writes small files only)

```bash
python data/preprocess_online_retail.py
python data/mapf_layouts.py --show
```

| Output | Content |
|---|---|
| `data/processed/online_retail_stats.json` | top-1000 SKU line shares, lines-per-invoice pmf (capped at 50), invoices per hour, cleaning log |
| `data/processed/*.csv` | the same statistics as readable tables |
| `data/layouts/warehouse-*.json` | held-out layouts (`layout_type: "mapf:<name>"`) |

Cleaning, the SKU-to-shelf mapping, the time-of-day mapping and the map derivation rule are described in
the docstrings of `preprocess_online_retail.py`, `environment/order.py` and `mapf_layouts.py`.

## 3. Protocol

* TRAIN: own layouts (`kiva_standard`, `narrow_corridor`) + synthetic orders, training seeds.
* DEV: own layouts, dev seeds (`seeds.dev_eval`), used for learning curves, checkpoint selection, ablations.
* TEST: MAPF layouts + empirical orders, test seeds (`evaluation.seeds`), run once at the end.
