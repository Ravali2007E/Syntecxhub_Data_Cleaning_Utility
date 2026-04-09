"""
Data Cleaning Utility
=====================
A comprehensive tool to clean and preprocess CSV/Excel datasets.
 
Features:
  - Detect and handle missing values (drop / fill / impute)
  - Fix incorrect dtypes (dates, numbers) and parse dates
  - Remove duplicates and standardize column names
  - Output a cleaned dataset and brief cleaning log
"""
 
import pandas as pd
import numpy as np
import argparse
import os
import json
import re
from datetime import datetime
 
 
# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
 
def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .xlsx/.xls")
    print(f"[✔] Loaded '{filepath}'  →  {df.shape[0]} rows × {df.shape[1]} columns")
    return df
 
 
# ─────────────────────────────────────────────
# 2. STANDARDIZE COLUMN NAMES
# ─────────────────────────────────────────────
 
def standardize_column_names(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    Normalize column names:
      - Strip whitespace
      - Lowercase
      - Replace spaces / special chars with underscores
    """
    original = list(df.columns)
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]
    renamed = {o: n for o, n in zip(original, df.columns) if o != n}
    if renamed:
        log.append({"step": "standardize_columns", "renamed": renamed})
        print(f"[✔] Renamed {len(renamed)} column(s): {renamed}")
    else:
        print("[✔] Column names already clean — no renaming needed.")
    return df
 
 
# ─────────────────────────────────────────────
# 3. REMOVE DUPLICATES
# ─────────────────────────────────────────────
 
def remove_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Drop fully duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    log.append({"step": "remove_duplicates", "rows_removed": removed})
    print(f"[✔] Removed {removed} duplicate row(s). Remaining: {len(df)}")
    return df
 
 
# ─────────────────────────────────────────────
# 4. FIX DTYPES — DATES & NUMBERS
# ─────────────────────────────────────────────
 
DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",           # 2024-01-15
    r"\d{2}/\d{2}/\d{4}",           # 15/01/2024  or  01/15/2024
    r"\d{2}-\d{2}-\d{4}",           # 15-01-2024
    r"\d{4}/\d{2}/\d{2}",           # 2024/01/15
]
DATE_RE = re.compile("|".join(DATE_PATTERNS))
 
 
def _looks_like_date(series: pd.Series) -> bool:
    """Return True if >50% of non-null string values look like dates."""
    sample = series.dropna().astype(str).head(50)
    hits = sample.apply(lambda v: bool(DATE_RE.search(v))).sum()
    return hits / max(len(sample), 1) > 0.5
 
 
def _looks_like_numeric(series: pd.Series) -> bool:
    """Return True if >70% of non-null values are coercible to float."""
    sample = series.dropna().astype(str).head(50)
    hits = pd.to_numeric(sample.str.replace(r"[,\$\s]", "", regex=True),
                         errors="coerce").notna().sum()
    return hits / max(len(sample), 1) > 0.7
 
 
def fix_dtypes(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    Iterate over object-typed columns and attempt:
      1. Parse to datetime
      2. Parse to numeric (strip currency / comma formatting)
    """
    dtype_changes = {}
    for col in df.select_dtypes(include="object").columns:
        if _looks_like_date(df[col]):
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
                dtype_changes[col] = "datetime64"
                print(f"  [date]    '{col}' → datetime")
        elif _looks_like_numeric(df[col]):
            cleaned = df[col].astype(str).str.replace(r"[,\$\s]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
                dtype_changes[col] = "numeric"
                print(f"  [numeric] '{col}' → numeric")
    if dtype_changes:
        log.append({"step": "fix_dtypes", "columns_converted": dtype_changes})
        print(f"[✔] Fixed dtypes for {len(dtype_changes)} column(s).")
    else:
        print("[✔] No dtype corrections needed.")
    return df
 
 
# ─────────────────────────────────────────────
# 5. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
 
def handle_missing(df: pd.DataFrame, strategy: str, log: list) -> pd.DataFrame:
    """
    Handle missing values according to `strategy`:
      - 'drop'   : drop rows that contain any NaN
      - 'fill'   : fill numeric cols with median, categorical with mode
      - 'impute' : same as fill (alias for clarity)
    """
    missing_before = df.isnull().sum().sum()
    missing_per_col = df.isnull().sum().to_dict()
 
    if strategy == "drop":
        df = df.dropna()
        print(f"[✔] Dropped rows with NaN. Missing before: {missing_before}, after: {df.isnull().sum().sum()}")
 
    elif strategy in ("fill", "impute"):
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                print(f"  [fill]  '{col}' numeric NaN → median ({fill_val:.4g})")
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    fill_val = mode_vals[0]
                    df[col] = df[col].fillna(fill_val)
                    print(f"  [fill]  '{col}' categorical NaN → mode ('{fill_val}')")
        print(f"[✔] Filled NaN values. Missing after: {df.isnull().sum().sum()}")
 
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: drop | fill | impute")
 
    log.append({
        "step": "handle_missing",
        "strategy": strategy,
        "missing_before": missing_before,
        "missing_after": int(df.isnull().sum().sum()),
        "per_column": {k: int(v) for k, v in missing_per_col.items() if v > 0},
    })
    return df
 
 
# ─────────────────────────────────────────────
# 6. SAVE OUTPUT
# ─────────────────────────────────────────────
 
def save_output(df: pd.DataFrame, input_path: str, output_dir: str) -> str:
    """Save the cleaned DataFrame as CSV next to the original (or in output_dir)."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"[✔] Cleaned dataset saved → '{out_path}'")
    return out_path
 
 
def save_log(log: list, output_dir: str, input_path: str) -> str:
    """Save the cleaning log as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    log_path = os.path.join(output_dir, f"{base}_cleaning_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"[✔] Cleaning log saved    → '{log_path}'")
    return log_path
 
 
# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
 
def clean(filepath: str, strategy: str = "fill", output_dir: str = "output") -> dict:
    """
    Full cleaning pipeline. Returns paths to cleaned CSV and log.
    """
    log = [{"step": "start", "file": filepath, "timestamp": str(datetime.now())}]
 
    print("\n" + "=" * 55)
    print("  DATA CLEANING UTILITY  —  SyntecxHub Project 3")
    print("=" * 55)
 
    # Step 1 — Load
    df = load_data(filepath)
    log.append({"step": "load", "shape": list(df.shape)})
 
    # Step 2 — Standardize columns
    print("\n[STEP 2] Standardizing column names …")
    df = standardize_column_names(df, log)
 
    # Step 3 — Remove duplicates
    print("\n[STEP 3] Removing duplicates …")
    df = remove_duplicates(df, log)
 
    # Step 4 — Fix dtypes
    print("\n[STEP 4] Fixing data types …")
    df = fix_dtypes(df, log)
 
    # Step 5 — Handle missing values
    print(f"\n[STEP 5] Handling missing values (strategy='{strategy}') …")
    df = handle_missing(df, strategy, log)
 
    # Step 6 — Save
    print("\n[STEP 6] Saving outputs …")
    cleaned_path = save_output(df, filepath, output_dir)
    log_path = save_log(log, output_dir, filepath)
 
    print("\n" + "=" * 55)
    print("  ✅  CLEANING COMPLETE")
    print(f"  Final shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 55 + "\n")
 
    return {"cleaned_csv": cleaned_path, "log_json": log_path}
 
 
# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Data Cleaning Utility — SyntecxHub Project 3"
    )
    parser.add_argument("filepath", help="Path to the raw CSV or Excel file")
    parser.add_argument(
        "--strategy",
        choices=["drop", "fill", "impute"],
        default="fill",
        help="How to handle missing values (default: fill)",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory to save cleaned file and log (default: output/)",
    )
    args = parser.parse_args()
    clean(args.filepath, args.strategy, args.output_dir)
 
 
if __name__ == "__main__":
    main()