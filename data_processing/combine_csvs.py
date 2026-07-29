#!/usr/bin/env python3
"""Combine by-dendrite-separate style CSV files with slightly different columns.

Expected input layout per CSV:
1) metadata labels row
2) metadata values row
3) header row
4+) data rows

The script builds a union of columns, preserves the 3-row metadata/header layout,
and appends all data rows from each input file.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Sequence, Tuple


def _normalize_rows(rows: Sequence[List[str]], width: int) -> List[List[str]]:
    """Pad/truncate rows to a fixed width."""
    normalized: List[List[str]] = []
    for row in rows:
        if len(row) < width:
            normalized.append(row + [""] * (width - len(row)))
        elif len(row) > width:
            normalized.append(row[:width])
        else:
            normalized.append(list(row))
    return normalized


def _read_structured_csv(path: str) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    """Read one by-dendrite-separate CSV into (meta_labels, meta_values, header, data_rows)."""
    with open(path, "r", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise ValueError(f"File does not have expected 3+ row layout: {path}")

    header = list(rows[2])
    width = len(header)
    meta_labels = list(rows[0])
    meta_values = list(rows[1])
    data_rows = [list(r) for r in rows[3:]]

    meta_labels = _normalize_rows([meta_labels], width)[0]
    meta_values = _normalize_rows([meta_values], width)[0]
    data_rows = _normalize_rows(data_rows, width)
    return meta_labels, meta_values, header, data_rows


def _combine_csvs(input_paths: Sequence[str], output_path: str) -> None:
    """Combine multiple structured CSVs using a union of columns."""
    parsed = []
    for path in input_paths:
        meta_labels, meta_values, header, data_rows = _read_structured_csv(path)
        parsed.append({
            "path": path,
            "meta_labels": meta_labels,
            "meta_values": meta_values,
            "header": header,
            "data_rows": data_rows,
        })

    if not parsed:
        raise ValueError("No input CSV files provided.")

    union_columns: List[str] = []
    seen = set()
    for item in parsed:
        for col in item["header"]:
            if col not in seen:
                seen.add(col)
                union_columns.append(col)

    combined_meta_labels: List[str] = []
    combined_meta_values: List[str] = []
    for col in union_columns:
        chosen_label = ""
        chosen_value = ""
        for item in parsed:
            header = item["header"]
            if col not in header:
                continue
            idx = header.index(col)
            label = item["meta_labels"][idx].strip()
            value = item["meta_values"][idx].strip()
            if chosen_label == "" and label != "":
                chosen_label = label
            if chosen_value == "" and value != "":
                chosen_value = value
            if chosen_label != "" and chosen_value != "":
                break
        combined_meta_labels.append(chosen_label)
        combined_meta_values.append(chosen_value)

    combined_data_rows: List[List[str]] = []
    for item in parsed:
        header = item["header"]
        col_to_index: Dict[str, int] = {name: i for i, name in enumerate(header)}
        for row in item["data_rows"]:
            out_row = []
            for col in union_columns:
                idx = col_to_index.get(col)
                out_row.append(row[idx] if idx is not None else "")
            combined_data_rows.append(out_row)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(combined_meta_labels)
        writer.writerow(combined_meta_values)
        writer.writerow(union_columns)
        writer.writerows(combined_data_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine by-dendrite-separate CSV files with mismatched columns.",
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="Input CSV files to combine.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output combined CSV path.",
    )
    args = parser.parse_args()

    missing = [path for path in args.csvs if not os.path.exists(path)]
    if missing:
        print("Error: Missing input files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)

    try:
        _combine_csvs(args.csvs, args.output)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Created: {args.output}")


if __name__ == "__main__":
    main()