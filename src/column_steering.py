"""Unit-shift and additive-composition steering of column coordinates (§5)."""
import argparse
from common import resume_complete, finish
from collections import defaultdict
from contextlib import ExitStack

import numpy as np

from common import (Runner, add_model_args, effect, mean, output_dir,
                    parse_layers, token_indices, write_json)
from data import cell, change_query, load_samples


def extract_directions(runner, samples, layers, offsets, radius):
    sums, counts = {}, defaultdict(int)
    for sample in samples:
        _, spans = runner.encode(sample.prompt)
        pos = {c: token_indices(spans, cell(sample, (0, c))["span"])[-1]
               for c in range(len(sample.columns))}
        positions = sorted({p + off for p in pos.values() for off in offsets})
        if min(positions) < 0:
            raise ValueError("Header neighborhood starts before the prompt.")
        index = {p: i for i, p in enumerate(positions)}
        states = runner.residuals(sample.prompt, layers, positions)
        for layer in layers:
            for col, p in pos.items():
                for off in offsets:
                    key = layer, off, col
                    value = states[layer][index[p + off]].numpy().astype(np.float64)
                    sums[key] = sums.get(key, np.zeros_like(value)) + value
                    counts[key] += 1
    means = {key: value / counts[key] for key, value in sums.items()}
    directions = {}
    for layer in layers:
        for off in offsets:
            cols = sorted(c for l, o, c in means if l == layer and o == off)
            for shift in range(-radius, radius + 1):
                if shift == 0:
                    continue
                values = [means[(layer, off, c + shift)] - means[(layer, off, c)]
                          for c in cols if (layer, off, c + shift) in means]
                if values:
                    directions[(layer, off, shift)] = np.mean(values, axis=0)
    return directions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=["unit", "composition", "both"], default="both")
    p.add_argument("--layers", default="17-23")
    p.add_argument("--offsets", default="-1,0")
    p.add_argument("--coefficient", type=float, default=8.0)
    p.add_argument("--extract-samples", type=int, default=100)
    p.add_argument("--component-radius", type=int, default=6)
    p.add_argument("--shifts", default="-4,-3,-2,-1,1,2,3,4")
    args = p.parse_args()
    out = output_dir(args, "column_steering")
    if resume_complete(out):
        return
    runner = Runner(args)
    layers = parse_layers(args.layers, len(runner.layers))
    offsets = [int(x) for x in args.offsets.split(",")]
    shifts = [int(x) for x in args.shifts.split(",")]
    if not shifts or 0 in shifts or args.component_radius < max(map(abs, shifts)):
        raise ValueError("Nonzero shifts must lie within the component radius.")
    samples = load_samples(args.data, args.limit)
    if not 0 < args.extract_samples < len(samples):
        raise ValueError("Need an extraction subset and a disjoint evaluation subset.")
    # A fixed seed partitions tables, not repeatedly decoded copies of a prompt.
    order = np.random.default_rng(args.seed).permutation(len(samples))
    extraction = [samples[i] for i in order[:args.extract_samples]]
    evaluation = [samples[i] for i in order[args.extract_samples:]]
    directions = extract_directions(runner, extraction, layers, offsets, args.component_radius)
    tasks = []
    for shift in shifts:
        if args.mode in ["unit", "both"]:
            tasks.append(("unit", shift, 1, 0))
        if args.mode in ["composition", "both"]:
            tasks.append(("baseline", shift, shift, 0))
            for a in range(-args.component_radius, args.component_radius + 1):
                b = shift - a
                if a and b and abs(b) <= args.component_radius:
                    tasks.append(("composition", shift, a, b))
    records, skipped = [], []
    for mode, shift, a, b in tasks:
        for sample in evaluation:
            r, c = sample.target
            target = c + shift
            if target < 1 or target >= len(sample.columns):
                skipped.append({"id": sample.id, "mode": mode, "shift": shift, "a": a, "b": b,
                                "reason": "shift leaves non-primary-key columns"})
                continue
            target_prompt = change_query(sample, col=target)
            try:
                lc = runner.logit_difference(target_prompt.prompt, target_prompt.answer, sample.answer)
                lx = runner.logit_difference(sample.prompt, target_prompt.answer, sample.answer)
                _, spans = runner.encode(sample.prompt)
                base = token_indices(spans, cell(sample, (0, c))["span"])[-1]
                with ExitStack() as stack:
                    for layer in layers:
                        patches = []
                        for off in offsets:
                            key = (layer, off, 1 if mode == "unit" else a)
                            if key not in directions or (b and (layer, off, b) not in directions):
                                raise ValueError("An extracted component direction has no support.")
                            vector = directions[key].copy()
                            if mode == "unit":
                                vector *= shift
                            elif b:
                                vector += directions[(layer, off, b)]
                            patches.append((base + off, vector * args.coefficient))
                        def add(hidden, patches=patches):
                            hidden = hidden.clone()
                            for pos, vector in patches:
                                hidden[0, pos] += runner.torch.as_tensor(
                                    vector, device=hidden.device, dtype=hidden.dtype)
                            return hidden
                        stack.enter_context(runner.pre_hook(layer, add))
                    lp = runner.logit_difference(sample.prompt, target_prompt.answer, sample.answer)
                records.append({"id": sample.id, "mode": mode, "shift": shift, "a": a, "b": b,
                                **effect(lc, lx, lp)})
            except ValueError as e:
                skipped.append({"id": sample.id, "mode": mode, "shift": shift,
                                "a": a, "b": b, "reason": str(e)})
        write_json(out / "column_steering.json", {
            "metadata": runner.metadata(), "layers": [l + 1 for l in layers],
            "offsets": offsets, "coefficient": args.coefficient,
            "extraction_ids": [s.id for s in extraction],
            "evaluation_ids": [s.id for s in evaluation], "records": records, "skipped": skipped,
        })

    finish(out)


if __name__ == "__main__":
    main()
