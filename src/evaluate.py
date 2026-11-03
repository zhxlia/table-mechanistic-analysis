"""Greedy atomic-cell evaluation and the error taxonomy used in the main text."""
import argparse
from common import resume_complete, finish
from collections import Counter

from common import Runner, add_model_args, answer_text, output_dir, write_json
from data import classify, load_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--data", required=True)
    args = p.parse_args()
    out = output_dir(args, "atomic_evaluation")
    if resume_complete(out):
        return
    runner = Runner(args)
    records = []
    for sample in load_samples(args.data, args.limit):
        text, info = runner.generate(sample.prompt)
        answer = answer_text(text)
        records.append({"id": sample.id, "answer": answer, "continuation": text,
                        "category": classify(sample, answer), **info})
        write_json(out / "evaluation.json", {
            "metadata": runner.metadata(), "records": records,
            "counts": dict(Counter(x["category"] for x in records)),
            "em": sum(x["category"] == "exact" for x in records) / len(records),
        })

    finish(out)


if __name__ == "__main__":
    main()
