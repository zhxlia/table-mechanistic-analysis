"""End-to-end wiring checks on a tiny random CPU checkpoint, never paper results."""
import contextlib
import io
import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_methods import toy_runner
from common import read_json, write_json, file_hash
from data import table_text, sample_from_prompt
import activation_patching
import column_steering
import coordinate_localization
import coordinate_probing
import delimiter_patching
import evaluate
import hitab
from hitab_data import build_hitab_prompt
import plot
import semantic_binding


def invoke(module, args):
    old = sys.argv
    sys.argv = [module.__name__, *map(str, args)]
    try:
        module.main()
    finally:
        sys.argv = old


class WorkflowCheck(unittest.TestCase):
    def test_cpu_pipeline_and_figures(self):
        # All temporary checkpoints, data and outputs stay under the project root
        # and are removed on success or failure.
        release = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix=".paper_code_check_", dir=release.parent) as tmp:
            tmp = Path(tmp)
            runner = toy_runner()
            checkpoint = tmp / "model"
            runner.model.save_pretrained(checkpoint)
            runner.tokenizer.save_pretrained(checkpoint)
            rows = [["alpha", "gamma", "x"], ["beta", "omega", "y"],
                    ["delta", "phi", "z"], ["theta", "eta", "w"]]
            columns = ["name", "score", "other"]
            records = []
            for i, order in enumerate(itertools.islice(itertools.permutations(rows), 6)):
                c = 1 if i % 2 == 0 else 2
                prompt = ("Table:\n" + table_text(columns, order) +
                          f"\n\nQuestion:\nWhat is the {columns[c]} for alpha?\n\nAnswer:\n")
                records.append({"id": str(i), "position": "after", "whole_input": prompt,
                                "table_list": {"columns": columns, "rows": list(order)},
                                "row_key": "alpha", "column_header": columns[c]})
            data = tmp / "data.json"
            write_json(data, records)
            common = ["--model", checkpoint, "--data", data, "--device", "cpu",
                      "--dtype", "float32", "--max-new-tokens", "4"]
            runs = tmp / "runs"
            with contextlib.redirect_stdout(io.StringIO()):
                invoke(evaluate, [*common, "--output", runs / "evaluation"])
                before = file_hash(runs / "evaluation/evaluation.json")
                invoke(evaluate, [*common, "--output", runs / "evaluation"])
                self.assertEqual(before, file_hash(runs / "evaluation/evaluation.json"))
                self.assertTrue((runs / "evaluation/complete.json").exists())
                invoke(semantic_binding, [*common, "--top-k", "2", "--output", runs / "semantic"])
                invoke(coordinate_localization, [*common, "--top-k", "2", "--output", runs / "coordinate"])
                invoke(activation_patching, [*common, "--mode", "residual", "--output", runs / "residual"])
                invoke(activation_patching, [*common, "--mode", "final-heads", "--output", runs / "final"])
                for kind in ["cell", "delimiter"]:
                    for mode in ["residual", "head"]:
                        invoke(coordinate_probing, [*common, "--kind", kind, "--mode", mode,
                               "--layers", "1-2", "--output", runs / f"probes-{mode}-{kind}"])
                invoke(delimiter_patching, [*common, "--top-k", "2",
                       "--cell-probes", runs / "probes-head-cell",
                       "--delimiter-probes", runs / "probes-head-delimiter",
                       "--output", runs / "delimiter"])
                invoke(column_steering, [*common, "--layers", "1-2", "--extract-samples", "2",
                       "--shifts=-1,1", "--component-radius", "2", "--output", runs / "steering"])
                # Exercise HiTab stages with an explicitly constructed parser/filter fixture.
                table = {"texts": [["name", "score", "other"], ["alpha", "10", "x"],
                                   ["beta", "20", "y"]],
                         "top_header_rows_num": 1, "left_header_columns_num": 1}
                examples, baseline = [], {}
                for i, col in enumerate([1, 2]):
                    header, answer = table["texts"][0][col], table["texts"][1][col]
                    sample = {"id": f"hitab-{i}", "question": f"What is the {header} for alpha?",
                              "answer": [answer], "linked_cells": {
                                  "entity_link": {"left": {"alpha": {"(1,0)": "alpha"}},
                                                  "top": {header: {f"(0,{col})": header}}},
                                  "quantity_link": {"[ANSWER]": {f"(1,{col})": answer}}}}
                    ex = build_hitab_prompt(sample, table)
                    examples.append(ex)
                    baseline[sample["id"]] = {
                        "category": "exact",
                        "prediction": {"answer": answer, "malformed": False,
                                       "evidence_cells": [list(c) for c in ex.gold_support_coords]}}
                args = type("Args", (), {"top_k": 2, "mover_top_k": 2})()
                hout = runs / "hitab"
                hout.mkdir()
                hitab.semantic(runner, examples, baseline, args, hout)
                hitab.coordinate(runner, examples, baseline, args, hout)
                hitab.propagation(runner, examples, baseline, args, hout)
                invoke(plot, ["--runs", runs, "--output", tmp / "figures"])
            self.assertGreater(len(read_json(runs / "residual/residual_patching.json")["records"]), 0)
            self.assertGreater(len(read_json(runs / "steering/column_steering.json")["records"]), 0)
            causal = read_json(runs / "delimiter/delimiter_patching.json")
            for axis in ["row", "column"]:
                for kind in ["cell", "delimiter"]:
                    self.assertTrue(any(x["axis"] == axis and x["kind"] == kind for x in causal["records"]))
            self.assertEqual(read_json(hout / "information_propagation.json")["denominator"], 2)
            self.assertGreaterEqual(len(list((tmp / "figures").glob("*.pdf"))), 10)
            for f in (tmp / "figures").glob("*.pdf"):
                self.assertTrue(f.read_bytes().startswith(b"%PDF"))
            self.assertFalse(any(release.rglob("*.json")))
            self.assertFalse(any(release.rglob("*.joblib")))


if __name__ == "__main__":
    unittest.main()
