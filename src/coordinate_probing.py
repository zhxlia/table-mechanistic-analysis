"""Residual and individual-head coordinate Ridge probes (§4)."""
import argparse
from common import resume_complete, finish
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from common import Runner, add_model_args, output_dir, parse_layers, token_indices, write_json
from data import load_samples


def labeled_tokens(sample, offsets, kind, axis):
    """Content tokens for cells; pipes for column delimiters, newlines for rows."""
    labels = {}
    max_row, max_col = len(sample.rows), len(sample.columns) - 1
    if kind == "cell":
        for x in sample.cells:
            value = x["row"] / max_row if axis == "row" else x["col"] / max_col
            for t in token_indices(offsets, x["span"]):
                if t in labels and labels[t] != value:
                    raise ValueError("A tokenizer token spans cells with different coordinates.")
                labels[t] = value
    else:
        for x in sample.cells:
            r, c = x["row"], x["col"]
            end = x["span"][1]
            if axis == "column" and c < max_col:
                marker = sample.prompt.find("|", end, end + 3)
                value = c / max_col
            elif axis == "row" and c == max_col:
                marker = end if sample.prompt[end:end + 1] == "\n" else -1
                value = r / max_row
            else:
                continue
            if marker < 0:
                raise ValueError("Missing structural delimiter.")
            for t in token_indices(offsets, (marker, marker + 1)):
                if t in labels and labels[t] != value:
                    raise ValueError("A tokenizer token spans different delimiter coordinates.")
                labels[t] = value
    ids = sorted(labels)
    return ids, np.array([labels[t] for t in ids], dtype=np.float64)


def fit_probe(x, y, seed):
    if len(y) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Probe requires >=10 finite, aligned token-coordinate pairs.")
    if np.std(y) == 0:
        raise ValueError("Coordinate target has no variance.")
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    if np.std(y[test]) == 0:
        raise ValueError("Evaluation coordinates have no variance.")
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(x[train], y[train])
    return model, float(r2_score(y[test], model.predict(x[test]))), train, test



def project_head_features(z, output_weight):
    # H = Z W_O^T. The right singular coordinates preserve H H^T, hence
    # Ridge(alpha=1) predictions, while avoiding a full d_model feature matrix.
    gram = output_weight.T.astype(np.float64) @ output_weight.astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]
    return z @ transform, transform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    p.add_argument("--kind", choices=["cell", "delimiter"], required=True)
    p.add_argument("--mode", choices=["residual", "head"], default="residual")
    p.add_argument("--layers", default="all", help="One-based, inclusive; e.g. 17 or 1-36.")
    args = p.parse_args()
    out = output_dir(args, "coordinate_probes")
    if resume_complete(out):
        return
    runner = Runner(args)
    layers = parse_layers(args.layers, len(runner.layers))
    samples = load_samples(args.data, args.limit)
    results = []
    for layer in layers:
        for axis in ["row", "column"]:
            xs, ys, refs = [], [], []
            for sample in samples:
                _, offsets = runner.encode(sample.prompt)
                positions, y = labeled_tokens(sample, offsets, args.kind, axis)
                if not positions:
                    raise ValueError(f"No {args.kind} tokens for {sample.id}")
                if args.mode == "residual":
                    x = runner.residuals(sample.prompt, [layer], positions, post=True)[layer].numpy()
                else:
                    x = runner.head_outputs(sample.prompt, layer, positions).numpy()
                xs.append(x)
                ys.append(y)
                refs.extend({"id": sample.id, "token": t} for t in positions)
            if not xs:
                raise ValueError("Empty probe dataset.")
            x, y = np.concatenate(xs), np.concatenate(ys)
            for head in ([None] if args.mode == "residual" else range(runner.n_heads)):
                features = x
                transform = None
                if head is not None:
                    # Probe H_h = W_O,h z_h in residual space, not the entire MHA output.
                    weight = runner.layers[layer].self_attn.o_proj.weight.detach().float().cpu().numpy()
                    lo, hi = head * runner.head_dim, (head + 1) * runner.head_dim
                    features, transform = project_head_features(x[:, head, :], weight[:, lo:hi])
                model, score, train, test = fit_probe(features, y, args.seed)
                name = f"{args.mode}-{args.kind}-L{layer + 1}-{axis}"
                if head is not None:
                    name += f"-H{head + 1}"
                joblib.dump(model if transform is None else {"regressor": model, "z_transform": transform},
                            out / (name + ".joblib"))
                row = {"layer": layer + 1, "head": head + 1 if head is not None else None,
                       "axis": axis, "r2": score, "train_tokens": len(train), "test_tokens": len(test)}
                results.append(row)
                if head is None and layer == 16:
                    prediction = model.predict(features[test])
                    write_json(out / f"layer17-{axis}-predictions.json",
                               {"axis": axis, "normalized": True,
                                "records": [{**refs[int(t)], "gold": float(y[t]), "predicted": float(v)}
                                            for t, v in zip(test, prediction)]})
            write_json(out / "probes.json",
                       {"metadata": runner.metadata(), "kind": args.kind,
                        "mode": args.mode, "results": results, "seed": args.seed})

    finish(out)


if __name__ == "__main__":
    main()
