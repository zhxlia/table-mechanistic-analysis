"""Residual-region and final-token head patching for the three-stage analysis."""
import argparse
from common import resume_complete, finish
from collections import Counter
from contextlib import ExitStack

import numpy as np

from common import (Runner, add_model_args, effect, heads_json, mean, output_dir,
                    rank_heads, token_indices, write_json)
from data import cell, change_query, counterfactual, load_samples


def focus_spans(reference, other, sample, axis):
    r, c = reference.target
    rr, cc = other.target
    coords = [(0, c), (r, 0), (r, c)]
    if axis == "column":
        coords += [(0, cc), (r, cc)]
    else:
        coords += [(rr, 0), (rr, c)]
    return sorted([cell(sample, coord)["span"] for coord in coords]
                  + [sample.query_col, sample.query_row])


def regions(offsets, focus):
    seq = []
    cursor = 0
    for span in focus:
        ids = token_indices(offsets, span)
        if not ids or min(ids) < cursor:
            raise ValueError("Focus spans overlap after tokenization.")
        a, b = min(ids), max(ids) + 1
        seq += [list(range(cursor, a)), list(range(a, b))]
        cursor = b
    seq.append(list(range(cursor, len(offsets))))
    if len(seq) != 15 or any(not x for x in seq):
        raise ValueError("Sample does not yield the paper's 15 nonempty regions.")
    return seq


def aligned_regions(runner, clean, corrupt, axis):
    _, a = runner.encode(clean.prompt)
    _, b = runner.encode(corrupt.prompt)
    clean_regions = regions(a, focus_spans(clean, corrupt, clean, axis))
    corrupt_regions = regions(b, focus_spans(clean, corrupt, corrupt, axis))
    if any(len(x) != len(y) for x, y in zip(clean_regions, corrupt_regions)):
        raise ValueError("Clean/corrupt region token counts differ; patch alignment is undefined.")
    return clean_regions, corrupt_regions


def patch_residual(runner, layer, source, positions):
    def patch(hidden):
        hidden = hidden.clone()
        hidden[0, positions] = source.to(device=hidden.device, dtype=hidden.dtype)
        return hidden
    return runner.pre_hook(layer, patch)


def patch_z(runner, layer, source, positions, heads):
    def patch(z):
        z = z.clone()
        # Assign per position/head explicitly; advanced-index copies must not be patched in place.
        for i, pos in enumerate(positions):
            z[0, pos, heads, :] = source[i, heads, :].to(device=z.device, dtype=z.dtype)
        return z
    return runner.z_hook(layer, patch)


def residual(args, runner, samples, out):
    layers = list(range(len(runner.layers)))
    rows, skipped = [], []
    for clean in samples:
        try:
            corrupt = counterfactual(clean, args.axis, args.seed, preceding=True)
            cr, xr = aligned_regions(runner, clean, corrupt, args.axis)
            lc = runner.logit_difference(clean.prompt, clean.answer, corrupt.answer)
            lx = runner.logit_difference(corrupt.prompt, clean.answer, corrupt.answer)
        except ValueError as e:
            skipped.append({"id": clean.id, "reason": str(e)})
            continue
        positions = [i for region in cr for i in region]
        states = runner.residuals(clean.prompt, layers, positions)
        metrics = []
        cursor = np.cumsum([0] + [len(x) for x in cr])
        for layer in layers:
            for site, target in enumerate(xr):
                source = states[layer][cursor[site]:cursor[site + 1]]
                with patch_residual(runner, layer, source, target):
                    lp = runner.logit_difference(corrupt.prompt, clean.answer, corrupt.answer)
                metrics.append({"layer": layer + 1, "region": site + 1, **effect(lc, lx, lp)})
        rows.append({"id": clean.id, "metrics": metrics})
        del states
        write_json(out / "residual_patching.json",
                   {"metadata": runner.metadata(), "records": rows, "skipped": skipped,
                    "regions": list(range(1, 16)), "axis": args.axis})
    if not rows:
        raise ValueError(f"No aligned patching pairs; first skip: {skipped[:1]}")
    write_json(out / "residual_patching.json",
               {"metadata": runner.metadata(), "records": rows, "skipped": skipped,
                "regions": list(range(1, 16)), "axis": args.axis})


def final_heads(args, runner, samples, out):
    rows, skipped = [], []
    for clean in samples:
        try:
            row_changed = counterfactual(clean, "row", args.seed)
            corrupt = counterfactual(row_changed, "column", args.seed)
            lc = runner.logit_difference(clean.prompt, clean.answer, corrupt.answer)
            lx = runner.logit_difference(corrupt.prompt, clean.answer, corrupt.answer)
        except ValueError as e:
            skipped.append({"id": clean.id, "reason": str(e)})
            continue
        ci, _ = runner.encode(clean.prompt)
        xi, _ = runner.encode(corrupt.prompt)
        metrics = []
        for layer in range(len(runner.layers)):
            source = runner.head_outputs(clean.prompt, layer, [ci.shape[1] - 1])
            for head in range(runner.n_heads):
                with patch_z(runner, layer, source, [xi.shape[1] - 1], [head]):
                    lp = runner.logit_difference(corrupt.prompt, clean.answer, corrupt.answer)
                metrics.append({"layer": layer + 1, "head": head + 1, **effect(lc, lx, lp)})
        rows.append({"id": clean.id, "metrics": metrics})
        write_json(out / "final_head_patching.json",
                   {"metadata": runner.metadata(), "records": rows, "skipped": skipped})
    if not rows:
        raise ValueError("No valid clean/corrupt logit-difference pairs.")
    totals = np.zeros((len(runner.layers), runner.n_heads))
    counts = np.zeros_like(totals)
    for row in rows:
        for m in row["metrics"]:
            if m["effect"] is not None and np.isfinite(m["effect"]):
                index = (m["layer"] - 1, m["head"] - 1)
                totals[index] += m["effect"]
                counts[index] += 1
    scores = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    positive = [(l, h) for l, h in rank_heads(scores, min(3, scores.size)) if scores[l, h] > 0]
    wins = {head: [] for head in positive}
    for sample in samples:
        _, offsets = runner.encode(sample.prompt)
        targets = token_indices(offsets, cell(sample, sample.target)["span"])
        def reduce(layer, attn):
            for l, h in positive:
                if layer == l:
                    # Winner is compared against the complete prompt, not table tokens alone.
                    wins[(l, h)].append(int(int(attn[h, -1].argmax()) in targets))
        runner.attention_reduce(sample.prompt, reduce)
    write_json(out / "final_head_patching.json",
               {"metadata": runner.metadata(), "records": rows, "skipped": skipped,
                "mean_effect": scores, "top_positive_heads": heads_json(positive),
                "target_win_rates": [mean(wins[h]) for h in positive]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=["residual", "final-heads"], default="residual")
    p.add_argument("--axis", choices=["column", "row"], default="column")
    args = p.parse_args()
    out = output_dir(args, args.mode + "_patching")
    if resume_complete(out):
        return
    runner = Runner(args)
    samples = load_samples(args.data, args.limit)
    (residual if args.mode == "residual" else final_heads)(args, runner, samples, out)

    finish(out)


if __name__ == "__main__":
    main()
