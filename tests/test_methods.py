"""CPU-only checks of the mechanisms; no checkpoint weights or paper results."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import Runner, effect, token_indices
from activation_patching import patch_residual, patch_z
from coordinate_localization import CoordinateAccumulator, rotate
from coordinate_probing import fit_probe, labeled_tokens, project_head_features
from data import generate, sample_from_prompt, swap_table, change_query
from hitab import grounding
from hitab_data import build_hitab_prompt, parse_model_prediction


def toy_runner():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    corpus = [" alpha beta gamma omega delta phi theta eta x y z w name score other Table: Question: Answer: \n",
              "Table:\nname | score\nalpha | gamma\nbeta | omega\n\nQuestion:\nWhat is the score for alpha?\n\nAnswer:\n"]
    tokenizer.train_from_iterator(corpus * 20, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["[UNK]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]")
    cfg = Qwen3Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=2048,
                      pad_token_id=fast.eos_token_id, eos_token_id=fast.eos_token_id,
                      attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    r = Runner.__new__(Runner)
    r.torch = torch
    r.tokenizer = fast
    r.model = Qwen3ForCausalLM(cfg).eval()
    r.config, r.layers = cfg, r.model.model.layers
    r.n_heads, r.n_kv_heads, r.head_dim = 4, 2, 16
    r.max_input_tokens, r.max_new_tokens = 2048, 4
    return r


def synthetic():
    columns = ["name", "score", "other"]
    rows = [["alpha", "gamma", "x"], ["beta", "omega", "y"]]
    prompt = ("Table:\nname | score | other\nalpha | gamma | x\nbeta | omega | y\n\n"
              "Question:\nWhat is the score for alpha?\n\nAnswer:\n")
    return sample_from_prompt(prompt, columns, rows, "alpha", "score", "toy")


def hitab_example():
    table = {"texts": [["name", "score", "other"], ["alpha", "10", "x"], ["beta", "20", "y"]],
             "top_header_rows_num": 1, "left_header_columns_num": 1}
    sample = {"id": "toy", "question": "What is the score for alpha?", "answer": ["10"],
              "linked_cells": {"entity_link": {"left": {"alpha": {"(1,0)": "alpha"}},
                                               "top": {"score": {"(0,1)": "score"}}},
                               "quantity_link": {"[ANSWER]": {"(1,1)": "10"}}}}
    return build_hitab_prompt(sample, table)


class MethodChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = toy_runner()

    def test_effect_keeps_negative_and_zero_denominators(self):
        self.assertEqual(effect(1, 3, 2)["effect"], 0.5)
        self.assertIsNone(effect(2, 2, 4)["effect"])
        self.assertEqual(effect(2, 2, 4)["raw_recovery"], 2)

    def test_query_change_does_not_change_table(self):
        s = synthetic()
        changed = change_query(s, col=2)
        self.assertEqual(changed.rows, s.rows)
        self.assertEqual(changed.answer, "x")
        self.assertEqual(changed.prompt[changed.table_span[0]:changed.table_span[1]],
                         s.prompt[s.table_span[0]:s.table_span[1]])

    def test_swaps_preserve_query_and_move_target(self):
        s = synthetic()
        for axis, other in [("row", 2), ("column", 2)]:
            x = swap_table(s, axis, other)
            self.assertEqual(x.answer, s.answer)
            self.assertNotEqual(x.target, s.target)
            self.assertEqual(x.prompt[x.query_col[0]:x.query_col[1]], "score")
            self.assertEqual(x.prompt[x.query_row[0]:x.query_row[1]], "alpha")

    def test_seeded_generation_and_global_uniqueness(self):
        a, b = generate(5, seed=19), generate(5, seed=19)
        self.assertEqual(a, b)
        self.assertEqual(len(a["samples"]), 5)
        for x in a["samples"]:
            values = [v.casefold() for row in x["table_list"]["rows"] for v in row]
            self.assertEqual(len(values), len(set(values)))
            self.assertGreaterEqual(len(x["table_list"]["rows"]), 4)
            self.assertLessEqual(len(x["table_list"]["columns"]), 20)

    def test_full_layer_zero_patch_recovers_clean_run(self):
        r = self.runner
        clean, corrupt = " alpha gamma", " beta gamma"
        ci, _ = r.encode(clean)
        xi, _ = r.encode(corrupt)
        self.assertEqual(ci.shape, xi.shape)
        positions = list(range(ci.shape[-1]))
        expected, _ = r.forward(clean)
        source = r.residuals(clean, [0], positions)[0]
        before, _ = r.forward(corrupt)
        with patch_residual(r, 0, source, positions):
            restored, _ = r.forward(corrupt)
        after, _ = r.forward(corrupt)
        torch.testing.assert_close(restored.logits, expected.logits, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(before.logits, after.logits)

    def test_ablation_zeros_only_selected_head_and_removes_hooks(self):
        r = self.runner
        seen = []
        with r.ablate([(0, 1)]):
            handle = r.layers[0].self_attn.o_proj.register_forward_pre_hook(
                lambda m, args: seen.append(args[0].detach().clone()))
            try:
                r.forward(" alpha gamma")
            finally:
                handle.remove()
        z = seen[0].reshape(1, -1, r.n_heads, r.head_dim)
        self.assertTrue(torch.equal(z[:, :, 1], torch.zeros_like(z[:, :, 1])))
        self.assertGreater(z[:, :, 0].abs().sum().item(), 0)
        self.assertEqual(len(r.layers[0].self_attn.o_proj._forward_pre_hooks), 0)

    def test_head_patch_uses_native_head_width(self):
        r = self.runner
        text = " alpha gamma"
        ids, _ = r.encode(text)
        pos = ids.shape[-1] - 1
        source = torch.full((1, r.n_heads, r.head_dim), 0.125)
        seen = []
        with patch_z(r, 0, source, [pos], [2]):
            h = r.layers[0].self_attn.o_proj.register_forward_pre_hook(
                lambda m, args: seen.append(args[0].detach().clone()))
            try:
                r.forward(text)
            finally:
                h.remove()
        z = seen[0].reshape(1, -1, r.n_heads, r.head_dim)
        torch.testing.assert_close(z[0, pos, 2], source[0, 2])
        self.assertEqual(r.head_dim, 16)
        self.assertNotEqual(r.head_dim, r.config.hidden_size // r.n_heads)

    def test_native_qk_and_gqa(self):
        r = self.runner
        text = " alpha gamma"
        ids, _ = r.encode(text)
        got = r.qk(text, [0], [ids.shape[-1] - 1], [0])[0]
        with torch.inference_mode():
            h = r.layers[0].input_layernorm(r.model.get_input_embeddings()(ids))
            attn = r.layers[0].self_attn
            q = attn.q_norm(attn.q_proj(h).reshape(1, -1, r.n_heads, r.head_dim))
            k = attn.k_norm(attn.k_proj(h).reshape(1, -1, r.n_kv_heads, r.head_dim))
        torch.testing.assert_close(got["q"][0], q[0, -1])
        torch.testing.assert_close(got["k"][0], k[0, 0])

    def test_attention_reducer_releases_hooks(self):
        r = self.runner
        seen = []
        def reduce(layer, pattern):
            torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
            seen.append(layer)
        r.attention_reduce(" alpha gamma", reduce)
        self.assertEqual(seen, [0, 1])
        self.assertTrue(all(not x.self_attn._forward_hooks for x in r.layers))

    def test_virtual_rotary_identity_and_norm(self):
        x = np.arange(16, dtype=float).reshape(2, 8)
        y = rotate(x, [0, 2], 10000)
        np.testing.assert_allclose(y[0], x[0])
        np.testing.assert_allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1))

    def test_probe_targets_and_split(self):
        s, r = synthetic(), self.runner
        _, offsets = r.encode(s.prompt)
        ids, y = labeled_tokens(s, offsets, "cell", "column")
        self.assertEqual(len(ids), len(y))
        self.assertEqual(y.min(), 0)
        self.assertEqual(y.max(), 1)
        for kind in ["cell", "delimiter"]:
            for axis in ["row", "column"]:
                ids, y = labeled_tokens(s, offsets, kind, axis)
                self.assertGreater(len(ids), 0)
                self.assertTrue(np.isfinite(y).all())
        x = np.arange(40, dtype=float).reshape(20, 2)
        model, score, train, test = fit_probe(x, x[:, 0], 42)
        self.assertEqual((len(train), len(test)), (16, 4))
        self.assertGreater(score, 0.99)
        self.assertEqual(model.alpha, 1.0)

    def test_head_probe_reduction_preserves_ridge_predictions(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(11)
        z = rng.normal(size=(60, 8))
        w = rng.normal(size=(32, 8))
        y = rng.normal(size=60)
        reduced, transform = project_head_features(z, w)
        full = z @ w.T
        a = Ridge(alpha=1.0, solver="lsqr", tol=1e-10).fit(full[:40], y[:40])
        b = Ridge(alpha=1.0, solver="lsqr", tol=1e-10).fit(reduced[:40], y[:40])
        np.testing.assert_allclose(a.predict(full[40:]), b.predict(reduced[40:]), atol=1e-7)

    def test_hitab_fractional_coordinates_are_malformed(self):
        p = parse_model_prediction('{"answer":"10","evidence_cells":[[1.5,1]]}')
        self.assertTrue(p["malformed"])

    def test_hitab_content_spans_and_stage_filters(self):
        ex = hitab_example()
        for c, span in ex.coord_to_char_span.items():
            self.assertEqual(ex.prompt[slice(*span)], ex.table_cells[c].text)
        def pred(evidence, answer="wrong"):
            return {"answer": answer, "evidence_cells": evidence, "malformed": False}
        self.assertEqual(grounding(ex, pred([[1, 2]]))[:2], (True, False))
        self.assertEqual(grounding(ex, pred([[2, 1]]))[:2], (False, True))
        self.assertEqual(grounding(ex, pred([[1, 1]], "10"))[2], {(1, 1)})
        self.assertFalse(grounding(ex, pred([[2, 1]], "10"))[2])

    def test_hitab_json_parser(self):
        p = parse_model_prediction('{"answer":"10","evidence_cells":[[1,1]],"operation":"lookup"}')
        self.assertFalse(p["malformed"])
        self.assertEqual(p["evidence_cells"], [[1, 1]])


    def test_model_family_native_qk_and_loading(self):
        import tempfile
        from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig,
                                  MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                                  Gemma3Config, Gemma3ForConditionalGeneration)
        from transformers import Gemma3TextConfig, SiglipVisionConfig
        base = self.runner
        values = dict(vocab_size=len(base.tokenizer), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=2048,
                      pad_token_id=base.tokenizer.eos_token_id,
                      eos_token_id=base.tokenizer.eos_token_id, attention_dropout=0.0)
        text = Gemma3TextConfig(**values, sliding_window=32, sliding_window_pattern=2)
        vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=1, num_attention_heads=2,
                                    image_size=8, patch_size=4)
        families = [(LlamaConfig(**values), LlamaForCausalLM),
                    (MistralConfig(**values, sliding_window=32), MistralForCausalLM),
                    (Qwen2Config(**{k: v for k, v in values.items() if k != "head_dim"}),
                     Qwen2ForCausalLM),
                    (Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                                  mm_tokens_per_image=4, boi_token_index=2,
                                  eoi_token_index=3, image_token_index=4),
                     Gemma3ForConditionalGeneration)]
        release = Path(__file__).resolve().parents[1]
        for cfg, cls in families:
            with self.subTest(model=cfg.model_type), tempfile.TemporaryDirectory(
                    prefix=".paper_code_check_", dir=release.parent) as tmp:
                model = cls(cfg).eval()
                model.save_pretrained(tmp)
                base.tokenizer.save_pretrained(tmp)
                r = Runner(argparse.Namespace(model=tmp, revision=None, seed=42,
                           dtype="float32", device="cpu", max_input_tokens=2048,
                           max_new_tokens=2))
                ids, _ = r.encode("Table: name score alpha gamma")
                positions = [0, ids.shape[1] - 1]
                captures = r.qk("Table: name score alpha gamma", [0], positions, positions)
                self.assertEqual(captures[0]["q"].shape, (2, r.n_heads, r.head_dim))
                self.assertEqual(captures[0]["k"].shape, (2, r.n_kv_heads, r.head_dim))
                patterns = []
                r.attention_reduce("Table: name score alpha gamma",
                                   lambda layer, pattern: patterns.append(tuple(pattern.shape)))
                self.assertEqual(len(patterns), 2)
                with r.ablate([(0, 0)]):
                    out, _ = r.forward("Table: name score alpha gamma")
                self.assertTrue(torch.isfinite(out.logits).all())


if __name__ == "__main__":
    unittest.main()
