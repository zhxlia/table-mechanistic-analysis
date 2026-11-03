"""HiTab loading, prompts, and evaluation adapted from the paper pipeline."""

from __future__ import annotations

import json, math, re

from pathlib import Path

from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]

_WS_RE = re.compile(r"\s+")

_COORD_RE = re.compile(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")

_CODE_FENCE_RE = re.compile(r"```(?:json)?|```", re.I)

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s.strip())
    return s

def canonicalize_numeric_string(text: str) -> Optional[str]:
    s = normalize_text(text).lower()
    if not s:
        return None
    s = s.replace(",", "")
    # Remove trivial leading labels.
    s = re.sub(r"^(answer|final answer|result)\s*[:=]\s*", "", s)
    # Keep percentages as text; they still parse numerically after removing %.
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    # Strip paired quotes.
    s = s.strip("\"'")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        val = float(s)
        if math.isfinite(val):
            if abs(val - round(val)) < 1e-9:
                out = str(int(round(val)))
            else:
                out = ("%.12f" % val).rstrip("0").rstrip(".")
            return out + ("%" if pct else "")
    return None

def canonicalize_answer(text: Any) -> str:
    s = normalize_text(text)
    if not s:
        return ""
    num = canonicalize_numeric_string(s)
    if num is not None:
        return num
    s = s.lower()
    s = s.strip("\"'")
    s = _WS_RE.sub(" ", s)
    return s

@dataclass

class TableCell:
    coord: Coord
    text: str
    region: str

@dataclass

class HiTabExample:
    sample: Dict[str, Any]
    table: Dict[str, Any]
    prompt: str
    question_char_span: Tuple[int, int]
    query_row_char_spans: List[Tuple[int, int]]
    query_col_char_spans: List[Tuple[int, int]]
    gold_row_header_coords: Set[Coord]
    gold_col_header_coords: Set[Coord]
    gold_corner_header_coords: Set[Coord]
    gold_answer_coords: Set[Coord]
    gold_support_coords: Set[Coord]
    other_row_header_coords: Set[Coord]
    other_col_header_coords: Set[Coord]
    coord_to_char_span: Dict[Coord, Tuple[int, int]]
    table_cells: Dict[Coord, TableCell]

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

class HiTabRepository:
    """Resolve HiTab sample files and table files from a local clone/unzipped repo."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Cannot find HiTab data dir: {self.data_dir}")
        self._table_index: Optional[Dict[str, List[Path]]] = None
        self._table_cache: Dict[str, Dict[str, Any]] = {}

    def load_split(self, split: str) -> List[Dict[str, Any]]:
        path = self.data_dir / f"{split}_samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find split file: {path}")
        return load_jsonl(path)

    @staticmethod
    def _path_priority(path: Path) -> Tuple[int, int, str]:
        """
        Prefer full table jsons under data/tables/ over adapted files under data/tables/hmt/.
        If multiple files share the same stem, we later merge them in this order.
        """
        posix = path.as_posix().lower()
        is_hmt = "/tables/hmt/" in posix or posix.endswith("/tables/hmt")
        return (1 if is_hmt else 0, len(path.parts), posix)

    def _build_table_index(self) -> Dict[str, List[Path]]:
        candidates: List[Path] = []
        search_dirs = [self.data_dir / "tables"]
        for d in search_dirs:
            if d.exists():
                candidates.extend(sorted(d.rglob("*.json")))

        # Fallback: search under data recursively.
        if not candidates:
            candidates.extend(sorted(self.data_dir.rglob("*.json")))

        # Deduplicate paths while preserving order.
        seen: Set[str] = set()
        uniq_candidates: List[Path] = []
        for p in candidates:
            s = str(p)
            if s not in seen:
                seen.add(s)
                uniq_candidates.append(p)

        index: Dict[str, List[Path]] = {}
        for p in uniq_candidates:
            index.setdefault(p.stem, []).append(p)
        for stem, paths in index.items():
            index[stem] = sorted(paths, key=self._path_priority)
        if not index:
            raise FileNotFoundError(
                "No table json files found. Make sure you have unzipped data/tables.zip into data/."
            )
        return index

    def load_table(self, table_id: str | int) -> Dict[str, Any]:
        tid = str(table_id)
        if tid in self._table_cache:
            return self._table_cache[tid]
        if self._table_index is None:
            self._table_index = self._build_table_index()
        if tid not in self._table_index:
            raise FileNotFoundError(f"Cannot resolve table file for table_id={tid}")

        merged: Dict[str, Any] = {}
        loaded_paths: List[str] = []
        for path in self._table_index[tid]:
            with path.open("r", encoding="utf-8") as f:
                cur = json.load(f)
            loaded_paths.append(str(path))
            if not isinstance(cur, dict):
                continue
            for k, v in cur.items():
                if k not in merged:
                    merged[k] = v
                elif isinstance(merged.get(k), dict) and isinstance(v, dict):
                    merged[k] = {**v, **merged[k]}

        if not merged:
            raise FileNotFoundError(f"Resolved table_id={tid}, but failed to load any json content.")
        merged.setdefault("__loaded_paths__", loaded_paths)
        self._table_cache[tid] = merged
        return merged

def _get_table_texts(table: Dict[str, Any]) -> List[List[Any]]:
    candidate_paths = [
        ("texts", table.get("texts")),
        ("table.texts", table.get("table", {}).get("texts") if isinstance(table.get("table"), dict) else None),
        ("matrix.texts", table.get("matrix", {}).get("texts") if isinstance(table.get("matrix"), dict) else None),
        ("table_data.texts", table.get("table_data", {}).get("texts") if isinstance(table.get("table_data"), dict) else None),
        ("table_matrix", table.get("table_matrix")),
        ("matrix", table.get("matrix") if isinstance(table.get("matrix"), list) else None),
    ]
    for _, value in candidate_paths:
        if isinstance(value, list) and (not value or isinstance(value[0], list)):
            return value

    loaded_paths = table.get("__loaded_paths__", [])
    raise KeyError(
        "Cannot find a table text matrix in table json. "
        f"Tried keys texts / table.texts / matrix.texts / table_data.texts / table_matrix. "
        f"Loaded from: {loaded_paths}"
    )

def _get_top_header_rows_num(table: Dict[str, Any]) -> int:
    if "top_header_rows_num" in table:
        return int(table["top_header_rows_num"])
    if "table" in table and "top_header_rows_num" in table["table"]:
        return int(table["table"]["top_header_rows_num"])
    raise KeyError("Cannot find 'top_header_rows_num' in table json")

def _get_left_header_columns_num(table: Dict[str, Any]) -> int:
    if "left_header_columns_num" in table:
        return int(table["left_header_columns_num"])
    if "table" in table and "left_header_columns_num" in table["table"]:
        return int(table["table"]["left_header_columns_num"])
    raise KeyError("Cannot find 'left_header_columns_num' in table json")

def parse_coord(coord_str: str) -> Coord:
    m = _COORD_RE.fullmatch(coord_str.strip())
    if not m:
        raise ValueError(f"Bad coord string: {coord_str}")
    return int(m.group(1)), int(m.group(2))

def collect_coords_from_mapping(mapping: Dict[str, Any]) -> Set[Coord]:
    coords: Set[Coord] = set()
    for coord_str in mapping.keys():
        try:
            coords.add(parse_coord(coord_str))
        except Exception:
            continue
    return coords

def collect_all_linked_coords(sample: Dict[str, Any]) -> Tuple[Set[Coord], Set[Coord], Set[Coord], Set[Coord], Set[Coord]]:
    linked = sample.get("linked_cells", {}) or {}
    entity = linked.get("entity_link", {}) or {}
    quantity = linked.get("quantity_link", {}) or {}

    row_headers: Set[Coord] = set()
    col_headers: Set[Coord] = set()
    corner_headers: Set[Coord] = set()
    answer_coords: Set[Coord] = set()
    support_coords: Set[Coord] = set()

    for area_name, bucket in (("left", row_headers), ("top", col_headers), ("top_left_corner", corner_headers)):
        area = entity.get(area_name, {}) or {}
        for phrase, coord_map in area.items():
            coords = collect_coords_from_mapping(coord_map)
            bucket |= coords
            support_coords |= coords
            if phrase == "[ANSWER]":
                answer_coords |= coords

    for phrase, coord_map in (quantity or {}).items():
        coords = collect_coords_from_mapping(coord_map)
        support_coords |= coords
        if phrase == "[ANSWER]":
            answer_coords |= coords

    return row_headers, col_headers, corner_headers, answer_coords, support_coords

def build_table_cells(table: Dict[str, Any]) -> Dict[Coord, TableCell]:
    texts = _get_table_texts(table)
    top_rows = _get_top_header_rows_num(table)
    left_cols = _get_left_header_columns_num(table)
    cells: Dict[Coord, TableCell] = {}

    for r, row in enumerate(texts):
        for c, value in enumerate(row):
            text = normalize_text(value)
            if r < top_rows and c < left_cols:
                region = "corner"
            elif r < top_rows:
                region = "top"
            elif c < left_cols:
                region = "left"
            else:
                region = "data"
            if not text:
                region = "empty"
            cells[(r, c)] = TableCell(coord=(r, c), text=text, region=region)
    return cells

def collect_other_header_coords(table: Dict[str, Any], gold_row: Set[Coord], gold_col: Set[Coord]) -> Tuple[Set[Coord], Set[Coord]]:
    cells = build_table_cells(table)
    other_row = {coord for coord, cell in cells.items() if cell.region == "left" and cell.text and coord not in gold_row}
    other_col = {coord for coord, cell in cells.items() if cell.region == "top" and cell.text and coord not in gold_col}
    return other_row, other_col

def _phrase_spans_in_text(text: str, phrase: str) -> List[Tuple[int, int]]:
    phrase_n = normalize_text(phrase)
    if not phrase_n:
        return []

    spans: List[Tuple[int, int]] = []
    lower_text = text.lower()
    lower_phrase = phrase_n.lower()
    start = 0
    while True:
        idx = lower_text.find(lower_phrase, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(lower_phrase)))
        start = idx + len(lower_phrase)
    return spans

class PromptBuilder:
    def __init__(self) -> None:
        self.parts: List[str] = []
        self.coord_to_char_span: Dict[Coord, Tuple[int, int]] = {}
        self.question_char_span: Tuple[int, int] = (0, 0)

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def add(self, s: str) -> Tuple[int, int]:
        start = len(self.text)
        self.parts.append(s)
        end = start + len(s)
        return start, end

    def add_cell(self, coord: Coord, text: str) -> Tuple[int, int]:
        self.add(f"({coord[0]},{coord[1]})=")
        span = self.add(text)
        self.coord_to_char_span[coord] = span
        return span

def build_hitab_prompt(
    sample: Dict[str, Any],
    table: Dict[str, Any],
    include_question: bool = True,
    max_rows: Optional[int] = None,
) -> HiTabExample:
    cells = build_table_cells(table)
    gold_row, gold_col, gold_corner, gold_answer, gold_support = collect_all_linked_coords(sample)
    other_row, other_col = collect_other_header_coords(table, gold_row, gold_col)

    question = normalize_text(sample.get("question", ""))
    sub_sentence = normalize_text(sample.get("sub_sentence", ""))
    top_rows = _get_top_header_rows_num(table)
    left_cols = _get_left_header_columns_num(table)

    pb = PromptBuilder()
    pb.add("You are given a hierarchical table from HiTab.\n")
    pb.add(
        "Return exactly one JSON object with keys: "
        "answer (string), evidence_cells (list of [row, col]), operation (string).\n"
    )
    pb.add(
        "Rules: no markdown, no extra text, no explanation; evidence_cells must contain the cells actually used.\n\n"
    )

    pb.add(f"Top header rows: {top_rows}\n")
    pb.add(f"Left header columns: {left_cols}\n")
    pb.add("Table:\n")

    texts = _get_table_texts(table)
    row_limit = len(texts) if max_rows is None else min(max_rows, len(texts))
    for r in range(row_limit):
        pb.add(f"row {r}: ")
        first = True
        for c, raw in enumerate(texts[r]):
            text = normalize_text(raw)
            if not text:
                continue
            if not first:
                pb.add(" | ")
            pb.add_cell((r, c), text)
            first = False
        pb.add("\n")

    pb.add("\n")
    if include_question:
        pb.add("Question: ")
        pb.question_char_span = pb.add(question)
        pb.add("\n")
    else:
        pb.question_char_span = (len(pb.text), len(pb.text))

    pb.add("Output JSON:\n")

    # Build question phrase spans from linked phrases. If a linked phrase is not found in the
    # question text, fall back to the whole question span for that axis.
    linked = sample.get("linked_cells", {}) or {}
    entity = linked.get("entity_link", {}) or {}

    row_phrase_spans: List[Tuple[int, int]] = []
    col_phrase_spans: List[Tuple[int, int]] = []

    for phrase in (entity.get("left", {}) or {}).keys():
        if phrase == "[ANSWER]":
            continue
        spans = _phrase_spans_in_text(question, phrase)
        row_phrase_spans.extend(
            [(pb.question_char_span[0] + s, pb.question_char_span[0] + e) for s, e in spans]
        )

    for phrase in (entity.get("top", {}) or {}).keys():
        if phrase == "[ANSWER]":
            continue
        spans = _phrase_spans_in_text(question, phrase)
        col_phrase_spans.extend(
            [(pb.question_char_span[0] + s, pb.question_char_span[0] + e) for s, e in spans]
        )

    if not row_phrase_spans and question:
        row_phrase_spans = [pb.question_char_span]
    if not col_phrase_spans and question:
        col_phrase_spans = [pb.question_char_span]

    return HiTabExample(
        sample=sample,
        table=table,
        prompt=pb.text,
        question_char_span=pb.question_char_span,
        query_row_char_spans=row_phrase_spans,
        query_col_char_spans=col_phrase_spans,
        gold_row_header_coords=gold_row,
        gold_col_header_coords=gold_col,
        gold_corner_header_coords=gold_corner,
        gold_answer_coords=gold_answer,
        gold_support_coords=gold_support,
        other_row_header_coords=other_row,
        other_col_header_coords=other_col,
        coord_to_char_span=pb.coord_to_char_span,
        table_cells=cells,
    )

def _extract_last_json_object(text: str) -> Optional[str]:
    text = normalize_text(_CODE_FENCE_RE.sub("", text))
    if not text:
        return None
    start_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(start_positions):
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break
    return None

def _coerce_evidence_cells(value: Any) -> List[List[int]]:
    if not isinstance(value, list):
        raise ValueError("evidence_cells must be a list")
    cells = []
    for item in value:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or int(x) != x or x < 0 for x in item)):
            raise ValueError("Evidence coordinates must be pairs of nonnegative integers")
        cells.append([int(item[0]), int(item[1])])
    return cells


def parse_model_prediction(raw_text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "raw_text": raw_text,
        "malformed": False,
        "answer": "",
        "evidence_cells": [],
        "operation": "unknown",
    }

    obj_text = _extract_last_json_object(raw_text)
    if obj_text is not None:
        try:
            obj = json.loads(obj_text)
            parsed["answer"] = normalize_text(obj.get("answer", ""))
            parsed["evidence_cells"] = _coerce_evidence_cells(obj.get("evidence_cells", []))
            parsed["operation"] = normalize_text(obj.get("operation", "unknown")) or "unknown"
            return parsed
        except Exception:
            pass

    # Fallback: try to recover a plain answer.
    lines = [normalize_text(x) for x in raw_text.splitlines() if normalize_text(x)]
    if lines:
        first = lines[0]
        first = re.sub(r"^(answer|final answer)\s*[:=]\s*", "", first, flags=re.IGNORECASE)
        parsed["answer"] = normalize_text(first)
    parsed["malformed"] = True
    return parsed

def gold_answer_strings(sample: Dict[str, Any]) -> List[str]:
    ans = sample.get("answer", [])
    if isinstance(ans, list):
        variants = [canonicalize_answer(x) for x in ans]
        return [x for x in variants if x]
    one = canonicalize_answer(ans)
    return [one] if one else []

def table_value_index(table_cells: Dict[Coord, TableCell]) -> Dict[str, Set[Coord]]:
    idx: Dict[str, Set[Coord]] = {}
    for coord, cell in table_cells.items():
        if not cell.text:
            continue
        key = canonicalize_answer(cell.text)
        idx.setdefault(key, set()).add(coord)
    return idx

def classify_prediction(example: HiTabExample, pred: Dict[str, Any]) -> str:
    gold_answers = set(gold_answer_strings(example.sample))
    pred_answer = canonicalize_answer(pred.get("answer", ""))
    pred_evidence = {tuple(x) for x in pred.get("evidence_cells", []) if isinstance(x, list) and len(x) == 2}
    gold_support = set(example.gold_support_coords)
    gold_answer_coords = set(example.gold_answer_coords)

    if pred.get("malformed", False):
        if pred_answer and pred_answer in gold_answers:
            return "answer_correct_but_malformed_output"
        return "malformed_output"

    if pred_answer in gold_answers:
        if pred_evidence == gold_support:
            return "exact"
        if pred_evidence and pred_evidence <= gold_support:
            return "answer_correct_evidence_incomplete"
        if not pred_evidence:
            return "answer_correct_without_evidence"
        return "answer_correct_wrong_evidence"

    # Wrong answer but maybe evidence is revealing the failure mode.
    if pred_evidence:
        if pred_evidence and pred_evidence <= (example.gold_row_header_coords | example.gold_col_header_coords | example.gold_corner_header_coords):
            return "header_only_confusion"
        if pred_evidence & gold_answer_coords:
            return "wrong_aggregation_on_gold_cells"
        if pred_evidence & gold_support:
            return "partial_evidence_but_wrong_answer"

    value_idx = table_value_index(example.table_cells)
    matched_coords = value_idx.get(pred_answer, set()) if pred_answer else set()

    if matched_coords:
        gold_rows = {r for r, _ in gold_answer_coords}
        gold_cols = {c for _, c in gold_answer_coords}
        pred_rows = {r for r, _ in matched_coords}
        pred_cols = {c for _, c in matched_coords}

        if pred_rows & gold_rows and not (pred_cols & gold_cols):
            return "same_row_wrong_col"
        if pred_cols & gold_cols and not (pred_rows & gold_rows):
            return "same_col_wrong_row"
        if all(example.table_cells.get(coord, TableCell(coord, "", "empty")).region in {"top", "left", "corner"} for coord in matched_coords):
            return "header_value_confusion"
        return "other_table_value_confusion"

    if pred_answer:
        return "unsupported_answer"
    return "empty_answer"
