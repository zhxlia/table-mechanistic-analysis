"""Logical-column Virtual-RoPE diagnostic (§3), not native attention logits."""
import argparse
from common import resume_complete, finish

import numpy as np

from common import (Runner, add_model_args, heads_json, output_dir, token_indices,
                    write_json)
from data import load_samples


def rotate(x, positions, theta):
    d = x.shape[-1]
    if d % 2:
        raise ValueError("Rotary head dimension must be even.")
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    freq = np.outer(np.asarray(positions), inv)
    emb = np.concatenate([freq, freq], axis=-1)
    other = np.concatenate([-x[..., d // 2:], x[..., :d // 2]], axis=-1)
    return x * np.cos(emb) + other * np.sin(emb)


class CoordinateAccumulator:
    def __init__(self, runner, layers):
        self.runner, self.layers = runner, layers
        self.q, self.k, self.qn, self.kn = {}, {}, {}, {}

    def add(self, prompt, targets, cells):
        _, offsets = self.runner.encode(prompt)
        valid = [(x["col"], token_indices(offsets, x["span"])) for x in cells]
        valid = [(c, ids) for c, ids in valid if ids]
        if not valid:
            raise ValueError("No token-aligned data cells.")
        indices = [ids[0] for _, ids in valid]
        states = self.runner.qk(prompt, self.layers, [len(offsets) - 1], indices)
        for layer in self.layers:
            q = states[layer]["q"][0].numpy().astype(np.float64)
            k = states[layer]["k"].numpy().astype(np.float64)
            # Contiguous groups of query heads share a KV head.
            k = np.repeat(k, self.runner.n_heads // self.runner.n_kv_heads, axis=1)
            for col in set(targets):
                key = (layer, col)
                self.q[key] = self.q.get(key, np.zeros_like(q)) + q
                self.qn[key] = self.qn.get(key, 0) + 1
            for (col, _), value in zip(valid, k):
                key = (layer, col)
                self.k[key] = self.k.get(key, np.zeros_like(value)) + value
                self.kn[key] = self.kn.get(key, 0) + 1

    def result(self, topk):
        theta = getattr(self.runner.config, "rope_theta", None)
        if theta is None:
            theta = getattr(self.runner.config, "rope_parameters", {}).get("rope_theta", 10000.0)
        candidates = []
        for layer in self.layers:
            cols = sorted({c for l, c in self.q if l == layer}
                          & {c for l, c in self.k if l == layer})
            if len(cols) < 2:
                continue
            q = np.stack([self.q[(layer, c)] / self.qn[(layer, c)] for c in cols])
            k = np.stack([self.k[(layer, c)] / self.kn[(layer, c)] for c in cols])
            for head in range(self.runner.n_heads):
                qr = rotate(q[:, head], cols, theta)
                kr = rotate(k[:, head], cols, theta)
                matrix = qr @ kr.T
                diag = np.eye(len(cols), dtype=bool)
                contrast = float(matrix[diag].mean() - matrix[~diag].mean())
                candidates.append({"layer": layer + 1, "head": head + 1,
                                   "columns": cols, "matrix": matrix, "contrast": contrast})
        candidates.sort(key=lambda x: (-x["contrast"], x["layer"], x["head"]))
        if len(candidates) < topk:
            raise ValueError("Insufficient supported heads/columns for the requested top-k.")
        selected = candidates[:topk]
        columns = sorted({c for x in selected for c in x["columns"]})
        index = {c: i for i, c in enumerate(columns)}
        total = np.zeros((len(columns), len(columns)))
        count = np.zeros_like(total)
        for x in selected:
            ids = [index[c] for c in x["columns"]]
            total[np.ix_(ids, ids)] += x["matrix"]
            count[np.ix_(ids, ids)] += 1
        matrix = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        predictions, gold = [], []
        for i, c in enumerate(columns):
            valid = np.isfinite(matrix[i])
            if not valid.any():
                continue
            prediction = columns[int(np.nanargmax(matrix[i]))]
            predictions.append(prediction)
            gold.append(c)
        pearson = (float(np.corrcoef(gold, predictions)[0, 1])
                   if len(set(gold)) > 1 and len(set(predictions)) > 1 else None)
        return {"top_k": topk, "heads": selected, "columns": columns, "matrix": matrix,
                "gold_columns": gold, "predicted_columns": predictions, "pearson": pearson,
                "top1": float(np.mean(np.equal(gold, predictions))),
                "definition": "RoPE(mean_Q_j,j)^T RoPE(mean_K_l,l); logical indices, no scale fit"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    out = output_dir(args, "coordinate_localization")
    if resume_complete(out):
        return
    runner = Runner(args)
    acc = CoordinateAccumulator(runner, list(range(len(runner.layers))))
    samples = load_samples(args.data, args.limit)
    for sample in samples:
        acc.add(sample.prompt, [sample.target[1]], [x for x in sample.cells if x["row"] > 0])
    write_json(out / "coordinate_localization.json",
               {"metadata": runner.metadata(), "sample_count": len(samples), **acc.result(args.top_k)})

    finish(out)


if __name__ == "__main__":
    main()
