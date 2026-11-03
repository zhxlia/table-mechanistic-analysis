"""Probe-selected head patching under row/column swaps (main-text Table 1)."""
import argparse
from common import resume_complete, finish
from contextlib import ExitStack
from pathlib import Path

from common import (Runner, add_model_args, effect, file_hash, heads_json, mean,
                    output_dir, read_json, token_indices, write_json)
from activation_patching import patch_z
from data import cell, load_samples, swap_table


def selected_heads(path, axis, topk):
    obj = read_json(Path(path) / "probes.json")
    if obj["mode"] != "head":
        raise ValueError("Causal selection requires individual-head probes.")
    rows = [x for x in obj["results"] if x["axis"] == axis and x["r2"] is not None]
    rows.sort(key=lambda x: (-x["r2"], x["layer"], x["head"]))
    if len(rows) < topk:
        raise ValueError("Not enough fitted head probes.")
    return [(x["layer"] - 1, x["head"] - 1) for x in rows[:topk]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    p.add_argument("--cell-probes", required=True)
    p.add_argument("--delimiter-probes", required=True)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    out = output_dir(args, "delimiter_causal_patching")
    if resume_complete(out):
        return
    runner = Runner(args)
    for kind, path in [("cell", args.cell_probes), ("delimiter", args.delimiter_probes)]:
        if read_json(Path(path) / "probes.json")["kind"] != kind:
            raise ValueError("Cell/delimiter probe files are interchanged.")
        config = read_json(Path(path) / "run.json")["config"]
        if (config["model"] != args.model or config["revision"] != args.revision
                or config["data_sha256"] != file_hash(args.data)):
            raise ValueError("Head probes use another checkpoint or dataset.")
    records, skipped = [], []
    heads = {}
    for axis in ["row", "column"]:
        for kind, path in [("cell", args.cell_probes), ("delimiter", args.delimiter_probes)]:
            hs = selected_heads(path, axis, args.top_k)
            heads[f"{axis}_{kind}"] = heads_json(hs)
            grouped = {}
            for l, h in hs:
                grouped.setdefault(l, []).append(h)
            for clean in load_samples(args.data, args.limit):
                r, c = clean.target
                index = r if axis == "row" else c
                other = index - 1 if index > 1 else index + 1
                try:
                    corrupt = swap_table(clean, axis, other)
                    foil = cell(corrupt, clean.target)["text"]
                    lc = runner.logit_difference(clean.prompt, clean.answer, foil)
                    lx = runner.logit_difference(corrupt.prompt, clean.answer, foil)
                    _, co = runner.encode(clean.prompt)
                    _, xo = runner.encode(corrupt.prompt)
                    cp = token_indices(co, cell(clean, clean.target)["span"])
                    xp = token_indices(xo, cell(corrupt, corrupt.target)["span"])
                    if len(cp) != len(xp):
                        raise ValueError("Target-cell token spans cannot be aligned.")
                    source = {l: runner.head_outputs(clean.prompt, l, cp) for l in grouped}
                    with ExitStack() as stack:
                        for l, hh in grouped.items():
                            stack.enter_context(patch_z(runner, l, source[l], xp, hh))
                        lp = runner.logit_difference(corrupt.prompt, clean.answer, foil)
                    records.append({"id": clean.id, "axis": axis, "kind": kind,
                                    "original_coordinate": clean.target, "swapped_coordinate": corrupt.target,
                                    **effect(lc, lx, lp)})
                except ValueError as e:
                    skipped.append({"id": clean.id, "axis": axis, "kind": kind, "reason": str(e)})
            write_json(out / "delimiter_patching.json",
                       {"metadata": runner.metadata(), "heads": heads,
                        "records": records, "skipped": skipped,
                        "summary": {f"{a}_{k}": mean(x["effect"] for x in records
                                    if x["axis"] == a and x["kind"] == k)
                                    for a in ["row", "column"] for k in ["cell", "delimiter"]}})

    finish(out)


if __name__ == "__main__":
    main()
