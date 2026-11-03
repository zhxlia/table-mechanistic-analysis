"""Generate main-text figures from runtime results without embedded measurements."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import numpy as np

from common import file_hash, mean, read_json, write_json


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def heatmap(matrix, path, xlabel, ylabel, heads=None, labels=None):
    a = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.ma.masked_invalid(a), aspect="auto", origin="lower", cmap="RdBu_r")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
    else:
        ax.set_xticks(range(a.shape[1]), range(1, a.shape[1] + 1))
        ax.set_yticks(range(0, a.shape[0], 4), range(1, a.shape[0] + 1, 4))
    if heads:
        from matplotlib.patches import Rectangle
        for h in heads:
            ax.add_patch(Rectangle((h["head"] - 1.5, h["layer"] - 1.5), 1, 1,
                                   fill=False, edgecolor="red", linewidth=1.5))
    save(fig, path)


def transitions(pairs, path):
    counts = Counter(pairs)
    if not counts:
        return
    n = sum(counts.values())
    left, right = sorted({a for a, _ in pairs}), sorted({b for _, b in pairs})
    def layout(labels, side):
        heights = {x: sum(v for k, v in counts.items() if k[side] == x) / n for x in labels}
        start, positions = 0.0, {}
        for x in labels:
            positions[x] = start
            start += heights[x] + 0.025
        return positions, heights, start
    lp, lh, lt = layout(left, 0)
    rp, rh, rt = layout(right, 1)
    cl, cr = lp.copy(), rp.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab10")
    for (a, b), number in sorted(counts.items()):
        h, lo, ro = number / n, cl[a], cr[b]
        vertices = [(0.05, lo), (0.4, lo), (0.6, ro), (0.95, ro),
                    (0.95, ro + h), (0.6, ro + h), (0.4, lo + h),
                    (0.05, lo + h), (0.05, lo)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(vertices, codes), facecolor=colors(left.index(a) % 10),
                               alpha=0.55, edgecolor="none"))
        cl[a] += h
        cr[b] += h
    for labels, pos, heights, x, align in [(left, lp, lh, 0, "right"), (right, rp, rh, 1, "left")]:
        for name in labels:
            ax.text(x, pos[name] + heights[name] / 2,
                    f"{name}\n{100 * heights[name]:.1f}%", ha=align, va="center", fontsize=8)
    ax.text(0, -0.05, "Clean")
    ax.text(0.85, -0.05, "Ablated")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.08, max(lt, rt))
    ax.axis("off")
    save(fig, path)


def metric_matrix(records, second):
    groups = defaultdict(list)
    for row in records:
        for x in row["metrics"]:
            groups[(x["layer"], x[second])].append(x["effect"])
    if not groups:
        raise ValueError("No patching measurements to plot.")
    a = np.full((max(k[0] for k in groups), max(k[1] for k in groups)), np.nan)
    for (l, c), values in groups.items():
        value = mean(values)
        if value is not None:
            a[l - 1, c - 1] = value
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output).expanduser().resolve()
    release = Path(__file__).resolve().parents[1]
    if out == release or release in out.parents:
        raise ValueError("Figures must be outside paper_code/.")
    out.mkdir(parents=True, exist_ok=True)
    sources, probes = [], []
    for run in args.runs:
        for path in sorted(Path(run).rglob("*.json")):
            if out == path.parent or out in path.parents or path.name == "run.json":
                continue
            obj, name, prefix = read_json(path), path.name, path.parent.name + "-"
            if name == "residual_patching.json":
                heatmap(metric_matrix(obj["records"], "region"), out / (prefix + "residual.pdf"),
                        "Region (1–15)", "Layer (1-based)")
            elif name == "final_head_patching.json":
                heatmap(obj["mean_effect"] if "mean_effect" in obj else metric_matrix(obj["records"], "head"),
                        out / (prefix + "final-heads.pdf"), "Head (1-based)", "Layer (1-based)",
                        obj.get("top_positive_heads"))
            elif name == "semantic_binding.json":
                if "retrieval" in obj:
                    fig, ax = plt.subplots()
                    for axis, y in obj["retrieval"].items():
                        ax.plot(obj["layers"], y, label=axis)
                    ax.set(xlabel="Layer", ylabel="Top-1 header retrieval", ylim=(0, 1.02))
                    ax.legend()
                    save(fig, out / (prefix + "header-retrieval.pdf"))
                    pairs = [(r["conditions"]["clean"]["category"], r["conditions"]["row"]["category"])
                             for r in obj["records"]]
                else:
                    pairs = [(r["baseline"], r["ablated"]) for r in obj["records"]]
                transitions(pairs, out / (prefix + "row-ablation.pdf"))
            elif name == "coordinate_localization.json":
                heatmap(obj["matrix"], out / (prefix + "virtual-rope.pdf"),
                        "Key column (0-based)", "Query column (0-based)", labels=obj["columns"])
            elif name == "information_propagation.json":
                heatmap(obj["focus_rate"], out / (prefix + "information-movers.pdf"),
                        "Head (1-based)", "Layer (1-based)", obj["heads"])
            elif name == "probes.json":
                if obj["mode"] == "residual":
                    probes.append(obj)
            elif name.startswith("layer17-") and name.endswith("-predictions.json"):
                rows = obj["records"]
                sid = rows[0]["id"]
                rows = sorted([x for x in rows if x["id"] == sid], key=lambda x: x["token"])
                fig, ax = plt.subplots(figsize=(9, 3))
                for key, label in [("predicted", "Prediction"), ("gold", "Gold")]:
                    ax.plot([x["token"] for x in rows], [x[key] for x in rows], label=label)
                ax.set(xlabel="Token position", ylabel="Normalized " + obj["axis"] + " index")
                ax.legend()
                save(fig, out / (prefix + name.replace(".json", ".pdf")))
            elif name == "delimiter_patching.json":
                fig, ax = plt.subplots()
                labels = ["row_cell", "row_delimiter", "column_cell", "column_delimiter"]
                ax.bar(labels, [np.nan if obj["summary"][k] is None else obj["summary"][k] for k in labels])
                ax.tick_params(axis="x", rotation=30)
                ax.set_ylabel("Effect Score")
                save(fig, out / (prefix + "delimiter-causal.pdf"))
            elif name == "column_steering.json":
                groups = defaultdict(list)
                for row in obj["records"]:
                    groups[(row["mode"], row["shift"], row["a"], row["b"])].append(row["effect"])
                for mode in ["unit", "composition"]:
                    fig, ax = plt.subplots()
                    for (kind, shift, a, b), values in groups.items():
                        if kind != mode and not (mode == "composition" and kind == "baseline"):
                            continue
                        value = mean(values)
                        if value is not None:
                            ax.scatter(shift, value, marker="*" if kind == "baseline" else "o",
                                       color="crimson" if kind == "baseline" else "steelblue")
                    ax.set(xlabel="Column shift", ylabel="Effect Score")
                    save(fig, out / (prefix + mode + "-steering.pdf"))
            else:
                continue
            sources.append({"path": str(path.resolve()), "sha256": file_hash(path)})
    if probes:
        fig, ax = plt.subplots()
        for obj in probes:
            for axis, color in [("row", "tab:orange"), ("column", "tab:blue")]:
                rows = sorted([x for x in obj["results"] if x["axis"] == axis], key=lambda x: x["layer"])
                ax.plot([x["layer"] for x in rows], [x["r2"] for x in rows],
                        "--" if obj["kind"] == "cell" else "-", color=color,
                        label=f"{obj['kind']} {axis}")
        ax.set(xlabel="Layer", ylabel="Held-out R²")
        ax.legend()
        save(fig, out / "coordinate-probes.pdf")
    if not sources:
        raise ValueError("No supported result files found.")
    write_json(out / "figure_sources.json", sources)


if __name__ == "__main__":
    main()
