"""
MAPF benchmark warehouse maps -> held-out TEST layouts (Phase 3b).

Source: Stern et al., "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks", SoCS 2019,
https://movingai.com/benchmarks/mapf.html. Map format: 4 header lines ("type octile", "height H",
"width W", "map") then H rows; '.' = free, '@'/'T' = obstacle.

Derivation rule (identical for every map; parameters in configs/default_config.yaml -> data.mapf):
  1. Crop. The maps have a large open area on the left, then 10x2 shelf blocks separated by 1-cell
     aisles. Keep `open_margin` free columns left of the first shelf column (x0 = first obstacle column
     inside the border - open_margin), y0 = 0, and a crop_width x crop_height window.
  2. Classify obstacles. Cells on the original map border, or obstacles with no free 4-neighbour inside
     the crop, become walls; obstacles with at least one free 4-neighbour become shelves (pickable from
     that neighbour).
  3. Connectivity. Keep the largest 4-connected free region; other free cells become walls; shelves with
     no access cell in that region become walls.
  4. Stations. `num_stations` free cells on the left-most free column of the region, evenly spaced
     from top to bottom (packing stations along the edge of the pick area).
Output: <environment.layouts_dir>/<map name>.json, loaded with layout_type "mapf:<map name>".

    python data/mapf_layouts.py
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_config

HEADER_LINES = 4
NEIGHBOURS = ((0, -1), (0, 1), (1, 0), (-1, 0))


def read_map(path: str, obstacle_chars: str) -> np.ndarray:
    """Returns a bool array [y, x], True = obstacle."""
    with open(path) as f:
        lines = f.read().splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    rows = lines[HEADER_LINES:HEADER_LINES + height]
    if len(rows) != height or any(len(r) < width for r in rows):
        raise ValueError(f"{path}: body does not match header {width}x{height}")
    return np.array([[c in obstacle_chars for c in r[:width]] for r in rows], dtype=bool)


def derive_layout(obst: np.ndarray, name: str, crop_w: int, crop_h: int, open_margin: int,
                  num_stations: int) -> Dict:
    H, W = obst.shape
    inner_cols = np.where(obst[1:H - 1, 1:W - 1].any(axis=0))[0]
    if inner_cols.size == 0:
        raise ValueError(f"{name}: no shelves found")
    x0 = max(1 + inner_cols[0] - open_margin, 0)
    y0 = 0
    crop = obst[y0:y0 + crop_h, x0:x0 + crop_w]
    h, w = crop.shape
    on_border = lambda x, y: (x + x0) in (0, W - 1) or (y + y0) in (0, H - 1)

    free = {(x, y) for y in range(h) for x in range(w) if not crop[y, x]}
    # largest 4-connected free component
    best: Set[Tuple[int, int]] = set()
    seen: Set[Tuple[int, int]] = set()
    for cell in sorted(free):
        if cell in seen:
            continue
        comp, stack = set(), [cell]
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            stack.extend((c[0] + dx, c[1] + dy) for dx, dy in NEIGHBOURS if (c[0] + dx, c[1] + dy) in free)
        seen |= comp
        if len(comp) > len(best):
            best = comp

    shelves, walls = [], []
    for y in range(h):
        for x in range(w):
            if (x, y) in best:
                continue
            is_obstacle = bool(crop[y, x])
            touches_region = any((x + dx, y + dy) in best for dx, dy in NEIGHBOURS)
            if is_obstacle and touches_region and not on_border(x, y):
                shelves.append([x, y])
            else:
                walls.append([x, y])  # border, interior obstacle, or disconnected free cell

    left_x = min(x for x, _ in best)
    column = sorted(y for x, y in best if x == left_x)
    picks = np.linspace(0, len(column) - 1, num_stations).round().astype(int)
    stations = [[left_x, column[i]] for i in sorted(set(picks.tolist()))]

    return {"name": name, "width": w, "height": h, "shelves": shelves, "stations": stations, "walls": walls,
            "source_crop": {"x0": int(x0), "y0": y0, "width": w, "height": h},
            "free_cells": len(best)}


def build_all(cfg: Dict) -> List[Dict]:
    dcfg, mcfg = cfg["data"], cfg["data"]["mapf"]
    out_dir = cfg["environment"]["layouts_dir"]
    os.makedirs(out_dir, exist_ok=True)
    layouts = []
    for m in mcfg["maps"]:
        path = os.path.join(dcfg["raw_dir"], dcfg["mapf_dir"], m["file"])
        if not os.path.exists(path):
            sys.exit(f"Map not found: {path}\nSee data/README.md for download instructions.")
        name = os.path.splitext(m["file"])[0]
        lay = derive_layout(read_map(path, mcfg["obstacle_chars"]), name, m["crop_width"], m["crop_height"],
                            mcfg["open_margin"], mcfg["num_stations"])
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(lay, f)
        layouts.append(lay)
    return layouts


def ascii_view(lay: Dict) -> str:
    grid = [["." for _ in range(lay["width"])] for _ in range(lay["height"])]
    for x, y in lay["walls"]:
        grid[y][x] = "#"
    for x, y in lay["shelves"]:
        grid[y][x] = "S"
    for x, y in lay["stations"]:
        grid[y][x] = "P"
    return "\n".join("".join(r) for r in grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--show", action="store_true", help="Print an ASCII view of each layout")
    args = parser.parse_args()
    for lay in build_all(load_config(args.config)):
        print(f"--> {lay['name']}: {lay['width']}x{lay['height']}, free={lay['free_cells']}, "
              f"shelves={len(lay['shelves'])}, stations={len(lay['stations'])}, crop={lay['source_crop']}")
        if args.show:
            print(ascii_view(lay))


if __name__ == "__main__":
    main()
