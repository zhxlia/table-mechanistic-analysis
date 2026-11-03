"""Shared model access, span alignment, interventions, and paper metrics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re


@dataclass
class Sample:
    id: str
    prompt: str
    cells: list
    query_row: tuple
    query_col: tuple
    target: tuple
    answer: str
    columns: list
    rows: list
    table_span: tuple
    # Each cell has matrix row/column indices and a content-only character span.


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if hasattr(x, "tolist"):
            return clean(x.tolist())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def effect(clean, corrupt, patched):
    denominator = float(clean - corrupt)
    return {
        "ld_clean": float(clean), "ld_corrupt": float(corrupt),
        "ld_patched": float(patched), "denominator": denominator,
        "raw_recovery": float(patched - corrupt),
        # Do not replace negative or zero denominators by a positive epsilon.
        "effect": float((patched - corrupt) / denominator) if denominator != 0 else None,
    }


def mean(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def add_model_args(p):
    p.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID.")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input-tokens", type=int, default=None,
                   help="Fail on overlength inputs; never silently truncate a table.")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True, help="Runtime output directory outside paper_code/.")


def output_dir(args, experiment):
    release = Path(__file__).resolve().parents[1]
    out = Path(args.output).expanduser().resolve()
    if out == release or release in out.parents:
        raise ValueError("Write generated data/results outside paper_code/.")
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    if config.get("data"):
        config["data_sha256"] = file_hash(config["data"])
    # A different configuration must never silently overwrite an existing run.
    digest = hashlib.sha256()
    for source in sorted((release / "src").glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    signature = {"experiment": experiment, "config": config, "code_sha256": digest.hexdigest()}
    path = out / "run.json"
    if path.exists() and read_json(path) != signature:
        raise FileExistsError("Output contains another configuration; select a new output directory.")
    write_json(path, signature)
    return out


def resume_complete(out):
    path = out / "complete.json"
    if not path.exists():
        return False
    for name, digest in read_json(path)["files"].items():
        item = out / name
        if not item.is_file() or file_hash(item) != digest:
            raise RuntimeError("Completed output changed or is missing; use a new output directory.")
    print(f"Reusing completed run: {out}")
    return True


def finish(out):
    files = {f.name: file_hash(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name not in {"run.json", "complete.json"}
             and not f.name.endswith(".tmp")}
    if not files:
        raise RuntimeError("Experiment produced no output.")
    write_json(out / "complete.json", {"files": files})


def parse_layers(text, count):
    """CLI layer numbers are one-based and ranges are inclusive."""
    if text == "all":
        return list(range(count))
    result = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            result.extend(range(lo - 1, hi))
        else:
            result.append(int(part) - 1)
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"Layers must be between 1 and {count}")
    return sorted(set(result))


def token_indices(offsets, span):
    lo, hi = span
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]


def spans_indices(offsets, spans):
    return sorted({i for span in spans for i in token_indices(offsets, span)})


def rank_heads(scores, topk):
    import numpy as np
    a = np.asarray(scores)
    candidates = [(l, h) for l in range(a.shape[0]) for h in range(a.shape[1])
                  if np.isfinite(a[l, h])]
    candidates.sort(key=lambda x: (-a[x], x))
    if len(candidates) < topk:
        raise ValueError(f"Only {len(candidates)} valid heads; requested {topk}")
    return candidates[:topk]


def heads_json(heads):
    return [{"layer": l + 1, "head": h + 1} for l, h in heads]


def decode_heads(heads):
    return [(int(x["layer"]) - 1, int(x["head"]) - 1) for x in heads]


def answer_text(text):
    # Greedy decoding may include a thinking block; it is excluded from EM.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"^\s*(?:answer|final answer)\s*:\s*", "", text, flags=re.I)
    return text.strip().splitlines()[0].strip() if text.strip() else ""


class Runner:
    """One HF implementation for all experiments; native per-head dimensions."""
    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offset mappings is required.")
        cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
        cls = AutoModelForCausalLM
        if cfg.model_type == "gemma3":
            from transformers import AutoModelForImageTextToText
            cls = AutoModelForImageTextToText
        self.model = cls.from_pretrained(
            args.model, revision=args.revision, dtype=getattr(torch, args.dtype),
            attn_implementation="eager").to(args.device).eval()
        core = self.model.model
        if hasattr(core, "language_model"):
            core = core.language_model
        self.layers = core.layers
        self.config = getattr(cfg, "text_config", cfg)
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = getattr(self.config, "num_key_value_heads", self.n_heads)
        self.head_dim = self.layers[0].self_attn.o_proj.in_features // self.n_heads
        if self.n_heads % self.n_kv_heads:
            raise ValueError("Invalid grouped-query head mapping.")
        if self.head_dim != getattr(self.config, "head_dim", self.head_dim):
            raise ValueError("Projection width disagrees with config.head_dim.")
        self.max_input_tokens = args.max_input_tokens
        self.max_new_tokens = args.max_new_tokens

    def encode(self, text):
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        maximum = self.max_input_tokens or getattr(self.config, "max_position_embeddings", None)
        if maximum and len(ids) > maximum:
            raise ValueError(f"Input length {len(ids)} exceeds {maximum}; no truncation applied.")
        return self.torch.tensor([ids], device=self.model.get_input_embeddings().weight.device), enc["offset_mapping"]

    def metadata(self):
        return {
            "model_class": type(self.model).__name__, "head_dim": self.head_dim,
            "num_heads": self.n_heads, "num_kv_heads": self.n_kv_heads,
            "num_layers": len(self.layers),
            "checkpoint_revision": getattr(self.model.config, "_commit_hash", None),
            "prompt_protocol": "raw one-shot prompt; no extra chat template",
        }

    def forward(self, text, attentions=False):
        ids, offsets = self.encode(text)
        with self.torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False, output_attentions=attentions)
        return out, offsets

    def logit_difference(self, text, target, foil):
        # Paper LD uses the first answer token, with the same leading-space convention.
        y = self.tokenizer.encode(" " + str(target), add_special_tokens=False)
        yp = self.tokenizer.encode(" " + str(foil), add_special_tokens=False)
        if not y or not yp or y[0] == yp[0]:
            raise ValueError("Target/foil first answer token must differ for logit difference.")
        out, _ = self.forward(text)
        value = (out.logits[0, -1, y[0]] - out.logits[0, -1, yp[0]]).float().item()
        del out
        return value

    def generate(self, text):
        ids, _ = self.encode(text)
        with self.torch.inference_mode():
            result = self.model.generate(
                ids, attention_mask=self.torch.ones_like(ids),
                do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        continuation = result[0, ids.shape[-1]:]
        # GenerationConfig from a checkpoint must not override greedy decoding.
        return self.tokenizer.decode(continuation, skip_special_tokens=True), {
            "generated_tokens": len(continuation),
            "hit_token_limit": len(continuation) >= self.max_new_tokens,
        }

    @contextmanager
    def pre_hook(self, layer, callback):
        def wrapped(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            modified = callback(hidden)
            if args:
                return (modified, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": modified}
        handle = self.layers[layer].register_forward_pre_hook(wrapped, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def z_hook(self, layer, callback):
        def wrapped(module, args):
            z = args[0]
            shape = z.shape
            view = z.reshape(*shape[:-1], self.n_heads, self.head_dim)
            return (callback(view).reshape(shape), *args[1:])
        handle = self.layers[layer].self_attn.o_proj.register_forward_pre_hook(wrapped)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def ablate(self, heads):
        grouped = {}
        for layer, head in heads:
            grouped.setdefault(layer, []).append(head)
        with ExitStack() as stack:
            for layer, hs in grouped.items():
                def zero(z, hs=hs):
                    z = z.clone()
                    z[:, :, hs, :] = 0
                    return z
                stack.enter_context(self.z_hook(layer, zero))
            yield

    def residuals(self, text, layers, positions, post=False):
        saved = {}
        with ExitStack() as stack:
            for layer in layers:
                if not post:
                    def capture(h, layer=layer):
                        saved[layer] = h[0, positions].detach().float().cpu()
                        return h
                    stack.enter_context(self.pre_hook(layer, capture))
                else:
                    def capture(module, args, output, layer=layer):
                        h = output[0] if isinstance(output, tuple) else output
                        saved[layer] = h[0, positions].detach().float().cpu()
                    handle = self.layers[layer].register_forward_hook(capture)
                    stack.callback(handle.remove)
            out, _ = self.forward(text)
            del out
        return saved

    def head_outputs(self, text, layer, positions):
        saved = {}
        def capture(z):
            saved["z"] = z[0, positions].detach().float().cpu()
            return z
        with self.z_hook(layer, capture):
            out, _ = self.forward(text)
            del out
        return saved["z"]

    def attention_reduce(self, text, callback):
        """Reduce one layer at a time, releasing quadratic attention arrays."""
        handles = []
        try:
            for layer, block in enumerate(self.layers):
                def capture(module, args, output, layer=layer):
                    if output[1] is None:
                        raise RuntimeError("Attention weights unavailable; eager attention is required.")
                    callback(layer, output[1][0].detach())
                    return (output[0], None, *output[2:])
                handles.append(block.self_attn.register_forward_hook(capture))
            out, _ = self.forward(text, attentions=True)
            del out
        finally:
            for h in handles:
                h.remove()

    def qk(self, text, layers, q_positions, k_positions):
        """Capture native Q/K after any head norm and before rotary positions."""
        saved = {layer: {} for layer in layers}
        handles = []
        try:
            for layer in layers:
                attn = self.layers[layer].self_attn
                for kind, positions, n_heads in [
                    ("q", q_positions, self.n_heads), ("k", k_positions, self.n_kv_heads)
                ]:
                    module = getattr(attn, kind + "_proj")
                    normalizer = getattr(attn, kind + "_norm", None)
                    def capture(module, args, output, layer=layer, kind=kind,
                                positions=positions, n_heads=n_heads, normalizer=normalizer):
                        z = output.reshape(output.shape[0], output.shape[1], n_heads, self.head_dim)
                        z = z[:, positions]
                        if normalizer is not None:
                            z = normalizer(z)
                        saved[layer][kind] = z[0].detach().float().cpu()
                    handles.append(module.register_forward_hook(capture))
            out, _ = self.forward(text)
            del out
        finally:
            for h in handles:
                h.remove()
        return saved
