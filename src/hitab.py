"""Stage-conditioned analysis of real-world HiTab tables (§6)."""
import argparse
from common import resume_complete, finish
from collections import Counter
from pathlib import Path

import numpy as np

from common import (Runner, add_model_args, file_hash, heads_json, output_dir,
                    rank_heads, read_json, spans_indices, write_json)
from coordinate_localization import CoordinateAccumulator
from semantic_binding import alignment_score
from hitab_data import (HiTabRepository, build_hitab_prompt, canonicalize_answer,
                        classify_prediction, gold_answer_strings, parse_model_prediction)


def identifier(example):
    return str(example.sample["id"])


def data_coords(example):
    return {c for c in example.gold_support_coords
            if c in example.table_cells and example.table_cells[c].region == "data"}


def grounding(example, prediction):
    evidence = {tuple(x) for x in prediction.get("evidence_cells", [])}
    if prediction.get("malformed") or any(c not in example.table_cells for c in evidence):
        return False, False, set()
    gold = data_coords(example)
    pred = {c for c in evidence if example.table_cells[c].region == "data"}
    row_evidence = pred or {c for c in evidence if example.table_cells[c].region == "left"}
    row_ok = bool(gold and row_evidence) and {r for r, _ in row_evidence} == {r for r, _ in gold}
    col_ok = bool(gold and pred) and {c for _, c in pred} == {c for _, c in gold}
    answers = set(gold_answer_strings(example.sample))
    targets = {c for c in example.gold_answer_coords
               if c in example.table_cells and example.table_cells[c].region == "data"
               and canonicalize_answer(example.table_cells[c].text) in answers}
    correct = canonicalize_answer(prediction.get("answer", "")) in answers
    # Aggregations, coincidentally repeated values, and header tokens are not cell-answer targets.
    propagation_targets = (targets if correct and targets and targets <= evidence
                           and pred <= gold else set())
    return row_ok, col_ok, propagation_targets


def evaluate(runner, examples, out, signature, initial=None):
    records = list(initial or [])
    for ex in examples:
        text, info = runner.generate(ex.prompt)
        prediction = parse_model_prediction(text)
        category = classify_prediction(ex, prediction)
        if not prediction["answer"] and not prediction["malformed"]:
            category = "empty_answer"
        records.append({"id": identifier(ex), "prediction": prediction,
                        "category": category, **info})
        write_json(out / "evaluation.json",
                   {"metadata": runner.metadata(), "signature": signature, "records": records,
                    "counts": dict(Counter(x["category"] for x in records))})
    return records


def semantic(runner, examples, baseline, args, out):
    eligible, excluded = [], []
    for ex in examples:
        pred = baseline[identifier(ex)]["prediction"]
        if grounding(ex, pred)[0] and ex.gold_row_header_coords and ex.query_row_char_spans:
            eligible.append(ex)
        else:
            excluded.append(identifier(ex))
    scores = np.zeros((len(runner.layers), runner.n_heads))
    used, skipped = [], []
    for ex in eligible:
        _, offsets = runner.encode(ex.prompt)
        q = spans_indices(offsets, ex.query_row_char_spans)
        t = spans_indices(offsets, [ex.coord_to_char_span[c] for c in ex.gold_row_header_coords
                                   if c in ex.coord_to_char_span])
        d = spans_indices(offsets, [ex.coord_to_char_span[c] for c in ex.other_row_header_coords
                                   if c in ex.coord_to_char_span])
        if not q or not t or not d:
            skipped.append({"id": identifier(ex), "reason": "missing query/target/distractor span"})
            continue
        def reduce(layer, pattern):
            scores[layer] += alignment_score(pattern, q, t, d).float().cpu().numpy()
        runner.attention_reduce(ex.prompt, reduce)
        used.append(ex)
    if not used:
        raise ValueError("No row-correct examples with aligned query/header spans.")
    scores /= len(used)
    heads = rank_heads(scores, args.top_k)
    records = []
    for ex in used:
        with runner.ablate(heads):
            text, info = runner.generate(ex.prompt)
        pred = parse_model_prediction(text)
        category = classify_prediction(ex, pred)
        if not pred["answer"] and not pred["malformed"]:
            category = "empty_answer"
        records.append({"id": identifier(ex), "baseline": baseline[identifier(ex)]["category"],
                        "ablated": category, "prediction": pred, **info})
        write_json(out / "semantic_binding.json",
                   {"metadata": runner.metadata(), "heads": heads_json(heads),
                    "alignment_scores": scores, "records": records,
                    "eligible_ids": [identifier(x) for x in used],
                    "excluded_ids": excluded, "skipped": skipped,
                    "denominator": len(used), "ablation_counts": dict(Counter(x["ablated"] for x in records))})


def coordinate(runner, examples, baseline, args, out):
    acc = CoordinateAccumulator(runner, list(range(len(runner.layers))))
    selected, excluded = [], []
    for ex in examples:
        if not grounding(ex, baseline[identifier(ex)]["prediction"])[1]:
            excluded.append(identifier(ex))
            continue
        cells = [{"row": r, "col": c, "span": span}
                 for (r, c), span in ex.coord_to_char_span.items()
                 if ex.table_cells[(r, c)].region == "data"]
        targets = {c for _, c in data_coords(ex)}
        acc.add(ex.prompt, targets, cells)
        selected.append(identifier(ex))
    if not selected:
        raise ValueError("No column-correct examples.")
    write_json(out / "coordinate_localization.json",
               {"metadata": runner.metadata(), "eligible_ids": selected,
                "excluded_ids": excluded, "denominator": len(selected), **acc.result(args.top_k)})


def propagation(runner, examples, baseline, args, out):
    sums = np.zeros((len(runner.layers), runner.n_heads))
    used, excluded = [], []
    for ex in examples:
        targets = grounding(ex, baseline[identifier(ex)]["prediction"])[2]
        if not targets:
            excluded.append(identifier(ex))
            continue
        _, offsets = runner.encode(ex.prompt)
        positions = spans_indices(offsets, [ex.coord_to_char_span[c] for c in targets])
        if not positions:
            raise ValueError("Eligible answer cell has no aligned content tokens.")
        def reduce(layer, pattern):
            last = pattern[:, -1, :]
            mask = runner.torch.ones(last.shape[-1], dtype=runner.torch.bool, device=last.device)
            mask[positions] = False
            wins = last[:, positions].max(-1).values > last[:, mask].max(-1).values
            sums[layer] += wins.float().cpu().numpy()
        runner.attention_reduce(ex.prompt, reduce)
        used.append(identifier(ex))
    if not used:
        raise ValueError("No correctly localized, correct cell-answer examples.")
    rates = sums / len(used)
    heads = rank_heads(rates, args.mover_top_k)
    write_json(out / "information_propagation.json",
               {"metadata": runner.metadata(), "focus_rate": rates, "heads": heads_json(heads),
                "eligible_ids": used, "excluded_ids": excluded, "denominator": len(used),
                "target": "answer-cell content only; maximum compared with all other input tokens"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--hitab-root", required=True, help="External HiTab checkout with data/tables.")
    p.add_argument("--split", choices=["train", "dev", "test"], default="test")
    p.add_argument("--stage", choices=["evaluate", "semantic", "coordinate", "propagation", "all"],
                   default="all")
    p.add_argument("--baseline", default=None, help="Evaluation JSON generated by this release.")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--mover-top-k", type=int, default=10)
    args = p.parse_args()
    out = output_dir(args, "hitab_" + args.stage)
    if resume_complete(out):
        return
    repository = HiTabRepository(args.hitab_root)
    samples = repository.load_split(args.split)
    if args.limit:
        samples = samples[:args.limit]
    examples = [build_hitab_prompt(s, repository.load_table(s["table_id"])) for s in samples]
    runner = Runner(args)
    signature = {
        "model": args.model, "revision": args.revision, "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
        "sample_file_sha256": file_hash(repository.data_dir / f"{args.split}_samples.jsonl"),
        "prompt_protocol": "full coordinate-labeled table; content-only cell spans; no gold-driven cropping",
    }
    if args.baseline:
        obj = read_json(args.baseline)
        if obj["signature"] != signature:
            raise ValueError("Baseline checkpoint, split, or prompt/generation settings differ.")
        records = obj["records"]
    elif (out / "evaluation.json").exists():
        obj = read_json(out / "evaluation.json")
        if obj["signature"] != signature:
            raise ValueError("Existing baseline signature differs.")
        completed = {x["id"] for x in obj["records"]}
        missing = [x for x in examples if identifier(x) not in completed]
        records = obj["records"]
        if missing:
            # Keep partial results while completing the same deterministic baseline.
            records = evaluate(runner, missing, out, signature, initial=records)
    else:
        records = evaluate(runner, examples, out, signature)
    baseline = {x["id"]: x for x in records}
    if any(identifier(ex) not in baseline for ex in examples):
        raise ValueError("Baseline is incomplete for the requested examples.")
    for name, function in [("semantic", semantic), ("coordinate", coordinate), ("propagation", propagation)]:
        if args.stage in [name, "all"]:
            function(runner, examples, baseline, args, out)

    finish(out)


if __name__ == "__main__":
    main()
