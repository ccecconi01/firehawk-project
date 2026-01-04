"""
CSV to JSON Converter (strict JSON + optional merge for FireHawk).

Usage:
  python csv_to_json.py <file.csv> [--output file.json] [--pretty] [--ndjson]

FireHawk merge mode: ---- this is for when we had two separate CSVs (alerts + RF results), now pipelineactive.py already merges those results ----
  python csv_to_json.py dataset_final_clean.csv --merge resultados_rf_final.csv --output fires.json --pretty 

"""

import pandas as pd
import json
import sys
import os
import math
from pathlib import Path


def _to_str_int(x):
    """Normalize IDs: keep as string digits when possible; otherwise return None."""
    if x is None:
        return None
    # pandas missing values
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # numbers
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return str(int(x))

    # strings
    s = str(x).strip()
    if not s:
        return None
    # remove trailing .0 if present
    if s.endswith(".0"):
        s = s[:-2]
    # if it's numeric-ish, keep digits
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _make_json_safe(value):
    """
    Ensure strict JSON compatibility:
    - convert NaN/Inf to None
    - keep other values unchanged
    """
    # pandas missing
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _df_to_records_strict(df: pd.DataFrame):
    """Convert DataFrame to JSON-safe list of dicts (records)."""
    # replace NaN/Inf in the DF first (faster than per-cell recursion later)
    df = df.copy()
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # convert to records
    records = df.to_dict(orient="records")

    # ensure strict JSON values
    out = []
    for row in records:
        clean = {k: _make_json_safe(v) for k, v in row.items()}
        out.append(clean)
    return out


def csv_to_json(csv_file, output_file=None, pretty=False, ndjson=False, id_columns=None):
    """
    Convert a CSV file to strict JSON (valid JSON, no NaN/Infinity).

    Args:
        csv_file: Path to CSV file
        output_file: Path to JSON output file (optional)
        pretty: If True, formats JSON with indentation for readability
        ndjson: If True, also writes .ndjson (1 JSON object per line)
        id_columns: list of column names to force as string (keeps leading zeros)

    Returns:
        str: path to the generated JSON file
    """
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        return None

    try:
        print(f"→ Reading CSV: {csv_file}")

        # Read CSV (use pandas default NA handling)
        df = pd.read_csv(csv_file)

        # Force specific ID columns to string (FireHawk use-case)
        if id_columns:
            for col in id_columns:
                if col in df.columns:
                    df[col] = df[col].apply(_to_str_int)

        print(f"  ✓ {len(df)} rows read")

        json_data = _df_to_records_strict(df)

        if output_file is None:
            csv_path = Path(csv_file)
            output_file = csv_path.parent / f"{csv_path.stem}.json"
        output_file = str(output_file)

        print(f"\n→ Saving JSON: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(json_data, f, ensure_ascii=False)

        if ndjson:
            ndjson_path = str(Path(output_file).with_suffix(".ndjson"))
            print(f"→ Saving NDJSON: {ndjson_path}")
            with open(ndjson_path, "w", encoding="utf-8") as f:
                for rec in json_data:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        file_size_kb = os.path.getsize(output_file) / 1024
        print(f"\n✓ Conversion completed!")
        print(f"  File: {output_file}")
        print(f"  Size: {file_size_kb:.2f} KB")
        print(f"  Rows: {len(json_data)}")

        print(f"\n Preview (first row):")
        print("-" * 70)
        if json_data:
            print(json.dumps(json_data[0], indent=2, ensure_ascii=False))

        return output_file

    except Exception as e:
        print(f"ERROR processing file: {e}")
        return None


def merge_firehawk(alerts_csv, results_csv, output_file=None, pretty=False, ndjson=False):
    """
    Merge FireHawk alerts + RF results:
    - alerts key: ID_Incidente (or NCCO fallback)
    - results key: ID_Incendio
    Output: array of alerts where each alert contains `rf_results` object (or null).
    Produces strict JSON (no NaN/Inf).

    Example:
      python csv_to_json.py dataset_final_clean.csv --merge resultados_rf_final.csv --output fires.json --pretty
    """
    if not os.path.exists(alerts_csv):
        print(f"ERROR: File not found: {alerts_csv}")
        return None
    if not os.path.exists(results_csv):
        print(f"ERROR: File not found: {results_csv}")
        return None

    print(f"→ Reading alerts CSV: {alerts_csv}")
    df_alerts = pd.read_csv(alerts_csv)
    print(f"  ✓ {len(df_alerts)} alert rows read")

    print(f"→ Reading results CSV: {results_csv}")
    df_results = pd.read_csv(results_csv)
    print(f"  ✓ {len(df_results)} results rows read")

    # Normalize IDs as strings (safer in JS)
    if "ID_Incidente" in df_alerts.columns:
        df_alerts["ID_Incidente"] = df_alerts["ID_Incidente"].apply(_to_str_int)
    if "NCCO" in df_alerts.columns:
        df_alerts["NCCO"] = df_alerts["NCCO"].apply(_to_str_int)
    if "ID_Incendio" in df_results.columns:
        df_results["ID_Incendio"] = df_results["ID_Incendio"].apply(_to_str_int)

    alerts_records = _df_to_records_strict(df_alerts)
    results_records = _df_to_records_strict(df_results)

    # map results by ID_Incendio
    results_map = {}
    for r in results_records:
        rid = _to_str_int(r.get("ID_Incendio"))
        if rid:
            results_map[rid] = r

    merged = []
    matched = 0

    for a in alerts_records:
        aid = _to_str_int(a.get("ID_Incidente")) or _to_str_int(a.get("NCCO"))
        rf = results_map.get(aid)
        if rf is not None:
            matched += 1
        a2 = dict(a)
        a2["rf_results"] = rf  # nested results
        merged.append(a2)

    if output_file is None:
        output_file = str(Path(alerts_csv).with_suffix("")) + "_with_rf_results.json"

    output_file = str(output_file)

    out_obj = {
        "alerts": merged,
        "meta": {
            "alerts_count": len(merged),
            "results_count": len(results_records),
            "matched_alerts_with_results": matched,
            "results_key": "rf_results",
            "alerts_id_key": "ID_Incidente",
            "results_id_key": "ID_Incendio",
        },
    }

    print(f"\n→ Saving merged JSON: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(out_obj, f, ensure_ascii=False)

    if ndjson:
        ndjson_path = str(Path(output_file).with_suffix(".ndjson"))
        print(f"→ Saving merged NDJSON (alerts only): {ndjson_path}")
        with open(ndjson_path, "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ Merge completed!")
    print(f"  File: {output_file}")
    print(f"  Alerts: {len(merged)}")
    print(f"  Results: {len(results_records)}")
    print(f"  Matched: {matched}")

    return output_file


def main():
    print("=" * 70)
    print("  CSV to JSON Converter (strict JSON)")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\nUSAGE: python csv_to_json.py <file.csv> [--output file.json] [--pretty] [--ndjson] [--merge results.csv]")
        print("\nExamples:")
        print("  python csv_to_json.py data/test_latest_30_rows.csv")
        print("  python csv_to_json.py data/test_latest_30_rows.csv --pretty")
        print("  python csv_to_json.py dataset_final_clean.csv --merge resultados_rf_final.csv --output fires.json --pretty")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = None
    pretty = False
    ndjson = False
    merge_file = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--pretty":
            pretty = True
        elif arg == "--ndjson":
            ndjson = True
        elif arg == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 1
        elif arg == "--merge" and i + 1 < len(args):
            merge_file = args[i + 1]
            i += 1
        i += 1

    # If merge mode requested:
    if merge_file:
        result = merge_firehawk(
            alerts_csv=csv_file,
            results_csv=merge_file,
            output_file=output_file,
            pretty=pretty,
            ndjson=ndjson,
        )
        sys.exit(0 if result else 1)

    # Normal conversion mode (force common id columns to string if present)
    result = csv_to_json(
        csv_file,
        output_file=output_file,
        pretty=pretty,
        ndjson=ndjson,
        id_columns=["ID_Incidente", "ID_Incendio", "NCCO"],
    )

    if result:
        print(f"\nReady to use!")
        print(f"  JSON file: {result}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
