################################################################################
# Combine by-dendrite-separate CSV files whose columns differ.                 #
################################################################################

#
"""
Imports
"""
import os
import csv
import sys
import argparse

from typing import Dict, List, Sequence, Tuple

#
"""
Functions
"""
def normalize_rows(rows: Sequence[List[str]], width: int) -> List[List[str]]:
    '''
    Pad short rows and trim long ones to the header width

    Signature:
        rows (Sequence[List[str]]):
            - Rows straight out of csv.reader
        width (int):
            - How many columns every row should end up with
    '''
    normalized: List[List[str]] = []
    for row in rows:
        if len(row) < width:
            normalized.append(row + [''] * (width - len(row)))
        elif len(row) > width:
            normalized.append(row[:width])
        else:
            normalized.append(list(row))
    return normalized

def read_structured_csv(
    path: str,
) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    '''
    Split one by-dendrite-separate CSV into its four row groups

    Notes:
        - Row 1 is metadata labels, row 2 metadata values, row 3 the
          header, and everything after is data
        - Every row is padded or trimmed to the header width

    Signature:
        path (str):
            - The CSV to read
    '''
    with open(path, 'r', newline='') as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise ValueError(f'File does not have expected 3+ row layout: {path}')

    header      = list(rows[2])
    width       = len(header)
    meta_labels = normalize_rows([list(rows[0])], width)[0]
    meta_values = normalize_rows([list(rows[1])], width)[0]
    data_rows   = normalize_rows([list(r) for r in rows[3:]], width)
    return meta_labels, meta_values, header, data_rows

def combine_csvs(input_paths: Sequence[str], output_path: str) -> None:
    '''
    Merge several structured CSVs into one file

    Notes:
        - The output columns are the union of every input header, in
          the order we first see them. A row from a file that lacks a
          column gets an empty cell
        - Metadata labels and values come from the first file with a
          non-empty entry for that column

    Signature:
        input_paths (Sequence[str]):
            - The CSVs to merge
        output_path (str):
            - Where the merged CSV goes
    '''
    parsed = []
    for path in input_paths:
        meta_labels, meta_values, header, data_rows = read_structured_csv(path)
        parsed.append({
            'path'       : path,
            'meta_labels': meta_labels,
            'meta_values': meta_values,
            'header'     : header,
            'data_rows'  : data_rows,
        })

    if not parsed:
        raise ValueError('No input CSV files provided.')

    union_columns: List[str] = []
    seen = set()
    for item in parsed:
        for col in item['header']:
            if col not in seen:
                seen.add(col)
                union_columns.append(col)

    combined_meta_labels: List[str] = []
    combined_meta_values: List[str] = []
    for col in union_columns:
        chosen_label = ''
        chosen_value = ''
        for item in parsed:
            header = item['header']
            if col not in header:
                continue
            idx   = header.index(col)
            label = item['meta_labels'][idx].strip()
            value = item['meta_values'][idx].strip()
            if chosen_label == '' and label != '':
                chosen_label = label
            if chosen_value == '' and value != '':
                chosen_value = value
            if chosen_label != '' and chosen_value != '':
                break
        combined_meta_labels.append(chosen_label)
        combined_meta_values.append(chosen_value)

    combined_data_rows: List[List[str]] = []
    for item in parsed:
        header = item['header']
        col_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(header)
        }
        for row in item['data_rows']:
            out_row = []
            for col in union_columns:
                idx = col_to_index.get(col)
                out_row.append(row[idx] if idx is not None else '')
            combined_data_rows.append(out_row)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(combined_meta_labels)
        writer.writerow(combined_meta_values)
        writer.writerow(union_columns)
        writer.writerows(combined_data_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Combine by-dendrite-separate CSV files with '
                      'mismatched columns'
    )
    parser.add_argument(
        '--csvs',
        nargs    = '+',
        required = True,
        help     = 'input CSV files to combine',
    )
    parser.add_argument(
        '--output',
        required = True,
        help     = 'output combined CSV path',
    )
    args = parser.parse_args()

    missing = [path for path in args.csvs if not os.path.exists(path)]
    if missing:
        print('Error: Missing input files:', file=sys.stderr)
        for path in missing:
            print(f'  {path}', file=sys.stderr)
        sys.exit(1)

    try:
        combine_csvs(args.csvs, args.output)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    print(f'Created: {args.output}')
