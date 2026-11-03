"""Synthetic atomic cell location; no stored datasets are required."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import re

from common import Sample, read_json, write_json
from entity_pool import ENTITY_POOL


TEMPLATES = [
    "What is the {column_header} for {target_row_key}?",
    "According to the table, what is the {column_header} for {target_row_key}?",
    "In the table, what is the value of {column_header} for {target_row_key}?",
]


def table_text(columns, rows):
    return "\n".join(" | ".join(map(str, row)) for row in [columns, *rows])


def find_span(text, value, start, end):
    matches = list(re.finditer(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", text[start:end]))
    if len(matches) != 1:
        raise ValueError(f"Expected one query occurrence of {value!r}, found {len(matches)}")
    return (start + matches[0].start(), start + matches[0].end())


def sample_from_prompt(prompt, columns, rows, row_key, col_header, identifier):
    columns = list(map(str, columns))
    rows = [list(map(str, row)) for row in rows]
    serialized = table_text(columns, rows)
    start = prompt.find(serialized, prompt.rfind("Table:"))
    if start < 0:
        raise ValueError("Serialized table does not match the prompt.")
    cells = []
    cursor = start
    for r, row in enumerate([columns, *rows]):
        for c, value in enumerate(row):
            cells.append({"row": r, "col": c, "text": value,
                          "span": (cursor, cursor + len(value))})
            cursor += len(value) + (3 if c < len(row) - 1 else 1)
    q_start = prompt.rfind("Question:") + len("Question:")
    q_end = prompt.rfind("Answer:")
    if q_start < len("Question:") or q_end <= q_start:
        raise ValueError("Expected Table/Question/Answer prompt markers.")
    qr = find_span(prompt, row_key, q_start, q_end)
    qc = find_span(prompt, col_header, q_start, q_end)
    r = next(i + 1 for i, row in enumerate(rows) if row[0] == str(row_key))
    c = columns.index(str(col_header))
    return Sample(str(identifier), prompt, cells, qr, qc, (r, c), rows[r - 1][c],
                  columns, rows, (start, start + len(serialized)))


def load_samples(path, limit=None):
    obj = read_json(path)
    records = obj["samples"] if isinstance(obj, dict) else obj
    records = [x for x in records if x.get("position", "after") == "after"]
    samples = []
    for x in records[:limit]:
        tab = x["table_list"]
        prompt = x.get("whole_input", x.get("input"))
        samples.append(sample_from_prompt(prompt, tab["columns"], tab["rows"],
                                         x["row_key"], x["column_header"], x["id"]))
    return samples


def cell(sample, coord):
    return next(x for x in sample.cells if (x["row"], x["col"]) == tuple(coord))


def change_query(sample, row=None, col=None):
    r, c = sample.target
    row_key = sample.rows[(row if row is not None else r) - 1][0]
    header = sample.columns[col if col is not None else c]
    edits = [(sample.query_row, row_key), (sample.query_col, header)]
    prompt = sample.prompt
    for (a, b), value in sorted(edits, reverse=True):
        prompt = prompt[:a] + value + prompt[b:]
    return sample_from_prompt(prompt, sample.columns, sample.rows, row_key, header, sample.id)


def swap_table(sample, axis, other):
    r, c = sample.target
    columns, rows = list(sample.columns), [list(x) for x in sample.rows]
    row_key = rows[r - 1][0]
    header = columns[c]
    if axis == "row":
        rows[r - 1], rows[other - 1] = rows[other - 1], rows[r - 1]
    else:
        columns[c], columns[other] = columns[other], columns[c]
        for row in rows:
            row[c], row[other] = row[other], row[c]
    a, b = sample.table_span
    prompt = sample.prompt[:a] + table_text(columns, rows) + sample.prompt[b:]
    return sample_from_prompt(prompt, columns, rows, row_key, header, sample.id)


def counterfactual(sample, axis, seed, preceding=False):
    rng = random.Random(str(seed) + ":" + sample.id + ":" + axis)
    r, c = sample.target
    if axis == "column":
        choices = [i for i in range(1, len(sample.columns)) if i != c and (not preceding or i < c)]
    else:
        choices = [i for i in range(1, len(sample.rows) + 1) if i != r and (not preceding or i < r)]
    rng.shuffle(choices)
    if not choices:
        raise ValueError("No eligible counterfactual coordinate.")
    return change_query(sample, col=choices[0]) if axis == "column" else change_query(sample, row=choices[0])


def classify(sample, prediction):
    p = str(prediction).strip().casefold()
    r, c = sample.target
    if p == sample.answer.casefold():
        return "exact"
    if p == sample.rows[r - 1][0].casefold():
        return "row_header"
    if p == sample.columns[c].casefold():
        return "col_header"
    for row in sample.cells:
        if row["row"] == 0 or row["col"] == 0 or row["text"].casefold() != p:
            continue
        if row["row"] == r:
            return "wrong_column"
        if row["col"] == c:
            return "wrong_row"
        return "wrong_row_and_column"
    return "not_in_table"


def generate(n=500, seed=42, row_min=4, row_max=20, col_min=4, col_max=20):
    if n < 1 or not (4 <= row_min <= row_max) or not (4 <= col_min <= col_max <= len(ENTITY_POOL)):
        raise ValueError("Invalid dataset size or table dimensions.")
    rng = random.Random(seed)
    # Preserve the original category order while making pools globally disjoint.
    seen = set()
    pools = {}
    for name, values in ENTITY_POOL.items():
        pools[name] = []
        for value in values:
            key = str(value).casefold()
            if key not in seen:
                pools[name].append(str(value))
                seen.add(key)
    results = []
    attempts = 0
    while len(results) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError("Entity pools cannot satisfy the requested dimensions.")
        nr, nc = rng.randint(row_min, row_max), rng.randint(col_min, col_max)
        key = rng.choice(["name", "role", "id"])
        candidates = [x for x in pools if x != key and len(pools[x]) >= nr]
        if len(pools[key]) < nr or len(candidates) < nc - 1:
            continue
        columns = [key, *rng.sample(candidates, nc - 1)]
        vectors = [rng.sample(pools[x], nr) for x in columns]
        rows = [list(x) for x in zip(*vectors)]
        r, c = rng.randrange(nr), rng.randrange(1, nc)
        template = rng.choice(TEMPLATES)
        query = template.format(column_header=columns[c], target_row_key=rows[r][0])
        demo_q = template.format(column_header="category", target_row_key="Nike")
        demo = ("Table:\nbrand | category | price\nSamsung | electronics | high\n"
                "Nike | clothing | medium\nIkea | furniture | low\n\nQuestion:\n"
                + demo_q + "\n\nAnswer:\nclothing\n\n")
        text = table_text(columns, rows)
        prompt = "Table:\n" + text + "\n\nQuestion:\n" + query + "\n\nAnswer:\n"
        results.append({
            "id": str(len(results)), "position": "after", "query": query,
            "input": prompt, "whole_input": demo + prompt, "table": text,
            "table_list": {"columns": columns, "rows": rows},
            "row_key": rows[r][0], "column_header": columns[c], "answer": rows[r][c],
        })
    return {"seed": seed, "num_tables": n, "row_range": [row_min, row_max],
            "column_range": [col_min, col_max], "samples": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--row-min", type=int, default=4)
    p.add_argument("--row-max", type=int, default=20)
    p.add_argument("--col-min", type=int, default=4)
    p.add_argument("--col-max", type=int, default=20)
    a = p.parse_args()
    release = Path(__file__).resolve().parents[1]
    destination = Path(a.output).expanduser().resolve()
    if destination == release or release in destination.parents:
        raise ValueError("Generated data must be outside paper_code/.")
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, generate(a.samples, a.seed, a.row_min, a.row_max, a.col_min, a.col_max))


if __name__ == "__main__":
    main()
