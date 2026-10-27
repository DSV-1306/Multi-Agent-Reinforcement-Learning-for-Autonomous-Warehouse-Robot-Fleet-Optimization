"""
Online Retail II -> small cached statistics for the "empirical" order mode (Phase 3a).

Source: UCI Machine Learning Repository, dataset id 502, "Online Retail II" (D. Chen), CC BY 4.0.
Raw file (NOT committed): <data.raw_dir>/<data.online_retail_file>, two sheets (2009-2010, 2010-2011).

Cleaning (in this order):
  1. Concatenate both sheets; drop exact duplicate rows (the sheets can overlap at the year boundary).
  2. Drop invoices starting with "C" (cancellations).
  3. Drop Quantity <= 0.
  4. Drop missing StockCode.
  5. (config) Drop non-product StockCodes = codes not starting with a digit (POST, M, DOT, BANK CHARGES, ...).
  6. (config) Drop Price <= 0 (adjustments, free samples).

Outputs (data.processed_dir):
  online_retail_stats.json   sku_probs (top-N SKUs by order-line count), lines_pmf, hourly_profile, metadata
  sku_popularity.csv, lines_per_invoice.csv, hourly_profile.csv   human-readable copies

    python data/preprocess_online_retail.py            (needs openpyxl; ~2-5 min on a laptop)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_config

HOURS_PER_DAY = 24


def clean(df: pd.DataFrame, opts: dict) -> (pd.DataFrame, dict):
    log = {"rows_raw": int(len(df))}
    df = df.drop_duplicates()
    log["rows_after_dedup"] = int(len(df))
    df = df[~df["Invoice"].astype(str).str.upper().str.startswith("C")]
    log["rows_after_drop_cancellations"] = int(len(df))
    df = df[df["Quantity"] > 0]
    log["rows_after_drop_nonpositive_quantity"] = int(len(df))
    df = df[df["StockCode"].notna() & (df["StockCode"].astype(str).str.strip() != "")]
    log["rows_after_drop_missing_stockcode"] = int(len(df))
    if opts["drop_non_product_codes"]:
        df = df[df["StockCode"].astype(str).str.match(r"^\d")]
        log["rows_after_drop_non_product_codes"] = int(len(df))
    if opts["drop_nonpositive_price"]:
        df = df[df["Price"] > 0]
        log["rows_after_drop_nonpositive_price"] = int(len(df))
    return df, log


def compute_stats(df: pd.DataFrame, opts: dict) -> dict:
    df = df.assign(StockCode=df["StockCode"].astype(str).str.strip().str.upper(),
                   Invoice=df["Invoice"].astype(str))
    # SKU popularity = number of order lines (an order line = one pick), not units
    lines_per_sku = df.groupby("StockCode").size().sort_values(ascending=False)
    top = lines_per_sku.head(opts["top_n_skus"])
    total_lines = int(lines_per_sku.sum())

    # Lines per invoice = distinct SKUs on the invoice, capped (tail folded into the last bucket)
    cap = opts["max_lines_per_invoice"]
    lines_per_inv = df.groupby("Invoice")["StockCode"].nunique().clip(upper=cap)
    pmf = np.bincount(lines_per_inv.to_numpy(), minlength=cap + 1)[1:].astype(float)
    pmf /= pmf.sum()

    # Hourly arrival profile = invoices started per hour of day (first timestamp of each invoice)
    inv_hour = pd.to_datetime(df.groupby("Invoice")["InvoiceDate"].min()).dt.hour
    hourly = np.bincount(inv_hour.to_numpy(), minlength=HOURS_PER_DAY).astype(float)

    return {
        "sku_codes": top.index.tolist(),
        "sku_line_counts": top.astype(int).tolist(),
        "sku_probs": (top / total_lines).tolist(),          # share of ALL order lines
        "top_n_share_of_lines": float(top.sum() / total_lines),
        "lines_pmf": pmf.tolist(),                           # index k -> k+1 lines
        "mean_lines_per_invoice_capped": float(np.sum(np.arange(1, cap + 1) * pmf)),
        "mean_lines_per_invoice_uncapped": float(df.groupby("Invoice")["StockCode"].nunique().mean()),
        "hourly_profile": hourly.tolist(),                   # invoice counts, hour 0..23
        "n_invoices": int(df["Invoice"].nunique()),
        "n_skus": int(lines_per_sku.size),
        "n_order_lines": total_lines,
        "date_min": str(pd.to_datetime(df["InvoiceDate"]).min()),
        "date_max": str(pd.to_datetime(df["InvoiceDate"]).max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["data"]
    opts = dcfg["online_retail"]
    src = os.path.join(dcfg["raw_dir"], dcfg["online_retail_file"])
    if not os.path.exists(src):
        sys.exit(f"Raw file not found: {src}\nSee data/README.md for download instructions.")

    print(f"--> Reading {src} (both sheets; this takes a few minutes) ...", flush=True)
    sheets = pd.read_excel(src, sheet_name=None, dtype={"Invoice": str, "StockCode": str}, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df, log = clean(df, opts)
    stats = compute_stats(df, opts)
    stats["cleaning"] = {**log, **{k: opts[k] for k in ("drop_non_product_codes", "drop_nonpositive_price")}}
    stats["source"] = ("UCI Machine Learning Repository id 502, Online Retail II (D. Chen), CC BY 4.0; "
                       f"sheets: {list(sheets)}")

    out_dir = dcfg["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "online_retail_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    pd.DataFrame({"rank": np.arange(1, len(stats["sku_codes"]) + 1), "stock_code": stats["sku_codes"],
                  "order_lines": stats["sku_line_counts"], "share_of_all_lines": stats["sku_probs"]}
                 ).to_csv(os.path.join(out_dir, "sku_popularity.csv"), index=False)
    pd.DataFrame({"lines": np.arange(1, len(stats["lines_pmf"]) + 1), "probability": stats["lines_pmf"]}
                 ).to_csv(os.path.join(out_dir, "lines_per_invoice.csv"), index=False)
    pd.DataFrame({"hour": np.arange(HOURS_PER_DAY), "invoices": stats["hourly_profile"]}
                 ).to_csv(os.path.join(out_dir, "hourly_profile.csv"), index=False)

    print(json.dumps(stats["cleaning"], indent=1))
    print(f"invoices={stats['n_invoices']:,}  skus={stats['n_skus']:,}  order lines={stats['n_order_lines']:,}  "
          f"dates {stats['date_min']} .. {stats['date_max']}")
    print(f"top-{opts['top_n_skus']} SKUs = {stats['top_n_share_of_lines']:.1%} of lines; "
          f"mean lines/invoice = {stats['mean_lines_per_invoice_uncapped']:.2f} "
          f"(capped at {opts['max_lines_per_invoice']}: {stats['mean_lines_per_invoice_capped']:.2f})")
    print("invoices per hour:", {h: int(c) for h, c in enumerate(stats["hourly_profile"]) if c})
    print(f"--> Wrote {out_dir}/online_retail_stats.json + 3 CSVs")


if __name__ == "__main__":
    main()
