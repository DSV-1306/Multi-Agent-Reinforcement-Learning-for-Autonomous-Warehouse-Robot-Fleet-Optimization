"""
Checks that the raw datasets are where configs/default_config.yaml expects them and prints their
size and SHA-256, so a report can state exactly which files were used. Downloads nothing.

    python data/check_datasets.py
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_config

CHUNK_BYTES = 1 << 20


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_files(cfg) -> list:
    d = cfg["data"]
    files = [os.path.join(d["raw_dir"], d["online_retail_file"])]
    files += [os.path.join(d["raw_dir"], d["mapf_dir"], m["file"]) for m in d["mapf"]["maps"]]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()
    missing = 0
    for path in expected_files(load_config(args.config)):
        if os.path.exists(path):
            print(f"OK       {os.path.getsize(path):>11,} B  sha256={sha256(path)}  {path}")
        else:
            missing += 1
            print(f"MISSING  {path}")
    if missing:
        sys.exit(f"{missing} file(s) missing - see data/README.md")


if __name__ == "__main__":
    main()
