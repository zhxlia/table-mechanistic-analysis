"""Span-mean header retrieval, alignment scores, and top-20 zero ablation (§3)."""
import argparse
from common import resume_complete, finish
from collections import Counter
import random

import numpy as np

from common import (Runner, add_model_args, answer_text, heads_json, mean,
                    output_dir, rank_heads, token_indices, write_json)
from data import cell, classify, load_samples


def alignment_score(pattern, query, target, distractor):
    if not query or not target or not distractor:
        raise ValueError("Empty alignment span or distractor set.")
    # Sum target-header mass, subtract mean mass over distractor tokens,
    # then average over the complete query-constraint span (paper equation).
    x = pattern[:, query, :]
    return (x[:, :, target].sum(-1) - x[:, :, distractor].mean(-1)).mean(-1)


def spans(sample, axis):
    r, c = sample.target
    if axis == "row":
        candidates = [x for x in sample.cells if x["row"] > 0 and x["col"] == 0]
        target = cell(sample, (r, 0))
        query = sample.query_row
    else:
        candidates = [x for x in sample.cells if x["row"] == 0]
        target = cell(sample, (0, c))
        query = sample.query_col
    return query, target, candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    out = output_dir(args, "semantic_binding")
    if resume_complete(out):
        return
    runner = Runner(args)
    samples = load_samples(args.data, args.limit)
    shape = (len(runner.layers), runner.n_heads)
    scores = {axis: np.zeros(shape) for axis in ["row", "column"]}
    retrieval = {axis: np.zeros(len(runner.layers)) for axis in scores}
    for index, sample in enumerate(samples):
        _, offsets = runner.encode(sample.prompt)
        positions = sorted({t for axis in scores for span in [
            spans(sample, axis)[0], *[x["span"] for x in spans(sample, axis)[2]]]
                            for t in token_indices(offsets, span)})
        mapping = {t: i for i, t in enumerate(positions)}
        states = runner.residuals(sample.prompt, range(len(runner.layers)), positions, post=True)
        for axis in scores:
            query, target, candidates = spans(sample, axis)
            qi = [mapping[i] for i in token_indices(offsets, query)]
            target_idx = candidates.index(target)
            for layer, hidden in states.items():
                q = hidden[qi].mean(0)
                headers = runner.torch.stack([
                    hidden[[mapping[i] for i in token_indices(offsets, x["span"])]].mean(0)
                    for x in candidates])
                sims = runner.torch.nn.functional.cosine_similarity(q[None], headers)
                retrieval[axis][layer] += int(sims.argmax().item() == target_idx)
        del states
        def reduce(layer, attn):
            for axis in scores:
                query, target, candidates = spans(sample, axis)
                q = token_indices(offsets, query)
                t = token_indices(offsets, target["span"])
                d = [i for x in candidates if x != target for i in token_indices(offsets, x["span"])]
                scores[axis][layer] += alignment_score(attn, q, t, d).float().cpu().numpy()
        runner.attention_reduce(sample.prompt, reduce)
    if not samples:
        raise ValueError("Empty dataset.")
    heads = {axis: rank_heads(a / len(samples), args.top_k) for axis, a in scores.items()}
    rng = random.Random(args.seed)
    excluded = set(heads["row"]) | set(heads["column"])
    available = [(l, h) for l in range(shape[0]) for h in range(shape[1]) if (l, h) not in excluded]
    if len(available) < args.top_k:
        raise ValueError("Not enough heads for an independent random-head control.")
    heads["random"] = rng.sample(available, args.top_k)
    records = []
    for sample in samples:
        predictions = {}
        for condition in ["clean", "row", "column", "random"]:
            with runner.ablate([] if condition == "clean" else heads[condition]):
                text, info = runner.generate(sample.prompt)
            pred = answer_text(text)
            predictions[condition] = {"answer": pred, "category": classify(sample, pred), **info}
        records.append({"id": sample.id, "conditions": predictions})
        write_json(out / "semantic_binding.json", {
            "metadata": runner.metadata(), "sample_count": len(samples),
            "layers": list(range(1, shape[0] + 1)),
            "retrieval": {k: v / len(samples) for k, v in retrieval.items()},
            "alignment_scores": {k: v / len(samples) for k, v in scores.items()},
            "heads": {k: heads_json(v) for k, v in heads.items()},
            "records": records,
        })

    finish(out)


if __name__ == "__main__":
    main()
