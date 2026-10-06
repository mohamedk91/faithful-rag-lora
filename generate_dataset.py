#!/usr/bin/env python3
"""Generate grounded-RAG QA fine-tuning data (train.jsonl / val.jsonl).

Input : chunks.jsonl, one JSON object per line with chunk_id, doc_id, text
        (aliases accepted: id / document_id, arxiv_id / chunk_text, content).
Output: <output-dir>/train.jsonl, val.jsonl      ({"messages": [...]} only)
        train_meta.jsonl, val_meta.jsonl         (type, doc, gold ids; line-aligned)
        splits.json                              (doc ids per split; reuse it to build the test set)
        stats.json, review_sample.md, candidates.jsonl (resume file), cache/

Design (what each piece protects against):
  * Documents are split 70/10/20 by doc_id; only train+val docs are used. Distractors for a
    split come only from that split's own documents (no val text inside train prompts).
  * The question is ALWAYS written from the gold chunk(s), and the generator is told which
    chunk to use. Hard-unanswerable questions are written from the gold chunk, which is
    then removed from the context.
  * Targets are built in code: refusals are hard-coded, citations are the gold labels.
    The LLM only supplies the question and the answer text.
  * Separate system prompts for generator and verifier (never the answering prompt).
  * An independent verifier checks that answers are supported and that unanswerable
    questions really are not answerable from the remaining chunks.
  * Chunks are relabeled c1..cN after shuffling, so labels are per-prompt.
  * Generation runs until each (split, type) quota is met, with an attempt cap.
  * Everything is cached on disk and candidates are appended to candidates.jsonl, so
    re-running the same command resumes where it stopped.

LLM access uses an OpenAI-compatible API: set API_KEY (and optionally API_BASE_URL,
GENERATOR_MODEL, VERIFIER_MODEL). Use a DIFFERENT model for the verifier if you can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install numpy: pip install numpy") from exc

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# System prompt used in the TRAINING RECORDS only (must match what you use at inference).
ANSWER_SYSTEM = (
    "You answer questions using only the supplied context chunks. Return exactly one "
    "compact JSON object with keys answer, citations, and supported. answer is a string "
    "or null; citations is an array of chunk labels such as c1; supported is a boolean. "
    "If the answer is not supported by the chunks, refuse with "
    '{"answer":null,"citations":[],"supported":false}.'
)
# System prompts for the data-generation roles (never the answering prompt).
GEN_SYSTEM = (
    "You are a careful data generator for a retrieval-augmented QA dataset. Follow the "
    "task exactly and return a single JSON object with exactly the requested keys and "
    "nothing else."
)
VERIFY_SYSTEM = (
    "You are a strict fact-checking judge. Use only the text provided, never outside "
    "knowledge. Return a single JSON object with exactly the requested keys and nothing else."
)

REFUSAL = {"answer": None, "citations": [], "supported": False}
TYPES = ("single", "multi", "hard_unanswerable", "easy_unanswerable", "counterfactual")
MIX = {
    "single": 0.45,
    "multi": 0.15,
    "hard_unanswerable": 0.20,
    "easy_unanswerable": 0.05,
    "counterfactual": 0.15,
}
ANSWERABLE = {"single", "multi", "counterfactual"}
UNANSWERABLE = {"hard_unanswerable", "easy_unanswerable"}
STYLES = [
    "a factoid question (who / what / which / when / where)",
    "a how or why question",
    "a numeric question (how many / how much / what value)",
    "a yes/no question whose answer includes a short justification",
    "a definition-style question",
    "a comparison or difference question",
]
FIELD_ALIASES = {
    "chunk_id": ("chunk_id", "id"),
    "doc_id": ("doc_id", "document_id", "arxiv_id"),
    "text": ("text", "chunk_text", "content"),
}
MIN_CHUNKS = 3
BAD_QUESTION = re.compile(
    r"\b(the|this|that|these|those)\s+(passage|text|document|chunk|context|excerpt|"
    r"paragraph|article|section|author|authors|paper|study|work)\b|according to|"
    r"as (described|mentioned|stated)|the proposed (method|approach|model|framework|system)",
    re.IGNORECASE,
)


class ApiError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield value


def normalize_chunk(raw: dict[str, Any], number: int) -> dict[str, str]:
    out = {}
    for field, aliases in FIELD_ALIASES.items():
        value = next((raw.get(a) for a in aliases if raw.get(a) is not None), None)
        if value is None or not str(value).strip():
            if field == "chunk_id":
                out[field] = ""  # generated below once doc_id is known
                continue
            raise ValueError(f"chunk #{number} has no {field}; accepted keys: {', '.join(aliases)}")
        out[field] = str(value).strip() if field == "text" else str(value)
    if not out["chunk_id"]:
        out["chunk_id"] = f"{out['doc_id']}::{number}"
    return out


YEAR = re.compile(r"\b(19|20)\d{2}\b")
VENUE = re.compile(r"arxiv|proceedings|journal|conference|\bpp\.|doi|preprint|advances in|neurips|"
                   r"icml|iclr|\bacl\b|emnlp|cvpr|trans\.", re.IGNORECASE)
REF_HEADING = re.compile(r"^\s*#{0,6}\s*\**\s*(references|bibliography)\s*\**\s*(\n|$)", re.IGNORECASE)


def noise_reason(text: str, min_alpha: float) -> str | None:
    """Why a chunk is a poor GOLD chunk (reference lists, tables, numeric dumps)."""
    bib_lines = sum(1 for line in text.split("\n")
                    if len(line) > 40 and YEAR.search(line) and VENUE.search(line))
    if REF_HEADING.match(text) or bib_lines >= 4:
        return "references"
    if text.count("<br>") >= 3 or text.count("|") >= 10:
        return "table"
    if sum(ch.isalpha() for ch in text) / max(1, len(text)) < min_alpha:
        return "low_alpha"
    return None


def overlap_neighbors(chunks: list[dict[str, str]], probe: int = 60) -> dict[str, set[str]]:
    """Consecutive chunks of one document whose text overlaps (sliding-window chunking).
    Such neighbours must not be used as distractors or as the 'second' multi-chunk."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for a, b in zip(chunks, chunks[1:]):
        if a["doc_id"] != b["doc_id"] or len(a["text"]) < probe or len(b["text"]) < probe:
            continue
        if a["text"][-probe:] in b["text"] or b["text"][:probe] in a["text"]:
            neighbors[a["chunk_id"]].add(b["chunk_id"])
            neighbors[b["chunk_id"]].add(a["chunk_id"])
    return neighbors


def parse_json(content: str) -> dict[str, Any]:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON object")
    value = json.loads(content[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def text_field(obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def contains(text: str, span: str) -> bool:
    """Case-insensitive whole-token containment (so '5' does not match '1.5')."""
    pattern = r"(?<![\w.])" + re.escape(span.strip().casefold()) + r"(?![\w])"
    return re.search(pattern, text.casefold()) is not None


# --------------------------------------------------------------------------------------
# LLM client (OpenAI-compatible) with disk cache, retries, thread safety
# --------------------------------------------------------------------------------------

class LLMClient:
    def __init__(self, cache_dir: Path, models: dict[str, str], api_key: str,
                 base_url: str | None, retries: int, timeout: float,
                 gen_temperature: float, json_mode: bool,
                 verifier_key: str | None = None, verifier_base_url: str | None = None,
                 rpm: float = 0.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install the client: pip install openai") from exc
        self.clients = {
            "generator": OpenAI(api_key=api_key, base_url=base_url, timeout=timeout),
            "verifier": OpenAI(api_key=verifier_key or api_key,
                               base_url=verifier_base_url or base_url, timeout=timeout),
        }
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.rate_lock = threading.Lock()
        self.next_slot = 0.0
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = models
        self.retries = retries
        self.gen_temperature = gen_temperature
        self.json_mode = json_mode
        self.lock = threading.Lock()
        self.stats = Counter()

    def _throttle(self) -> None:
        """Global requests-per-minute limiter shared by all worker threads."""
        if not self.min_interval:
            return
        with self.rate_lock:
            now = time.monotonic()
            wait = self.next_slot - now
            self.next_slot = max(now, self.next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def call(self, role: str, system: str, prompt: str) -> dict[str, Any]:
        model = self.models[role]
        key = hashlib.sha256(json.dumps(
            [role, model, system, prompt, self.json_mode], ensure_ascii=False).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                with self.lock:
                    self.stats["cache_hits"] += 1
                return value
            except json.JSONDecodeError:
                pass  # corrupt cache entry: regenerate
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model,
                    temperature=0.0 if role == "verifier" else self.gen_temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                )
                if self.json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                self._throttle()
                response = self.clients[role].chat.completions.create(**kwargs)
                parsed = parse_json(response.choices[0].message.content or "")
                tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
                with self.lock:
                    self.stats[f"api_calls_{role}"] += 1
                return parsed
            except Exception as exc:  # transient provider errors, bad JSON, rate limits
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 30) + random.random())
        raise ApiError(f"{role} call failed after {self.retries + 1} attempts: {last!r}")


# --------------------------------------------------------------------------------------
# Retriever (one shared embedding matrix, filtered per split)
# --------------------------------------------------------------------------------------

def load_encoder(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install: pip install sentence-transformers") from exc
    return SentenceTransformer(name)


class TfidfEncoder:
    """Lexical TF-IDF 'embeddings': no PyTorch or model download, language-agnostic.
    Fitted on the corpus; rows are L2-normalised so dot product = cosine similarity."""

    def __init__(self, texts: list[str]) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise SystemExit("Install scikit-learn: pip install scikit-learn") from exc
        self.vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True,
                                          strip_accents="unicode", ngram_range=(1, 2))
        self.vectorizer.fit(texts)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return self.vectorizer.transform(texts).toarray().astype("float32")


class Retriever:
    def __init__(self, chunks: list[dict[str, str]], split_docs: dict[str, set[str]],
                 model_name: str) -> None:
        self.chunks = chunks
        self.split_docs = split_docs
        if model_name == "tfidf":
            self.model = TfidfEncoder([c["text"] for c in chunks])
        else:
            self.model = load_encoder(model_name)
        self.lock = threading.Lock()
        self.vectors = self.model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)

    def _scored(self, query: str, split: str, exclude_ids: set[str], exclude_docs: set[str],
                only_doc: str | None) -> list[tuple[float, dict[str, str]]]:
        with self.lock:
            q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.vectors @ q
        allowed = self.split_docs[split]
        rows = [(float(scores[i]), c) for i, c in enumerate(self.chunks)
                if c["doc_id"] in allowed and c["chunk_id"] not in exclude_ids
                and c["doc_id"] not in exclude_docs and (only_doc is None or c["doc_id"] == only_doc)]
        rows.sort(key=lambda r: -r[0])
        return rows

    def nearest(self, query: str, count: int, split: str, exclude_ids: set[str] | None = None,
                exclude_docs: set[str] | None = None, only_doc: str | None = None
                ) -> list[dict[str, str]]:
        rows = self._scored(query, split, exclude_ids or set(), exclude_docs or set(), only_doc)
        return [c for _, c in rows[:count]]

    def far(self, query: str, count: int, split: str, rng: random.Random,
            exclude_docs: set[str] | None = None) -> list[dict[str, str]]:
        """Random chunks from the least-similar half of the split (off-topic distractors)."""
        rows = self._scored(query, split, set(), exclude_docs or set(), None)
        pool = [c for _, c in rows[len(rows) // 2:]]
        if len(pool) < count:
            pool = [c for _, c in rows]
        return rng.sample(pool, min(count, len(pool)))


# --------------------------------------------------------------------------------------
# Prompt formatting (single source of truth for the training/inference template)
# --------------------------------------------------------------------------------------

def format_user_prompt(chunks: list[dict[str, Any]], question: str) -> str:
    context = "\n".join(f"[{c['label']}] {c['text']}" for c in chunks)
    return f"{context}\nQuestion: {question}"


def label_key(label: str) -> int:
    return int(label[1:])


def build_context(rng: random.Random, items: list[dict[str, Any]], max_chars: int
                  ) -> dict[str, Any] | None:
    """items: dicts with chunk_id, text, gold(bool). Drops distractors if too long,
    shuffles, then relabels c1..cN. Returns None if it cannot fit / is too small."""
    items = list(items)
    total = lambda xs: sum(len(x["text"]) for x in xs)  # noqa: E731
    while len(items) > MIN_CHUNKS and total(items) > max_chars:
        drop = [i for i, x in enumerate(items) if not x["gold"]]
        if not drop:
            break
        items.pop(drop[-1])
    if len(items) < MIN_CHUNKS or total(items) > max_chars:
        return None
    rng.shuffle(items)
    chunks = [{"label": f"c{i}", "chunk_id": x["chunk_id"], "text": x["text"], "gold": x["gold"]}
              for i, x in enumerate(items, 1)]
    return {"chunks": chunks, "gold_labels": [c["label"] for c in chunks if c["gold"]]}


def valid_target(target: Any, labels: set[str]) -> tuple[bool, str]:
    if not isinstance(target, dict) or set(target) != {"answer", "citations", "supported"}:
        return False, "schema"
    answer, cites, supported = target["answer"], target["citations"], target["supported"]
    if not isinstance(supported, bool) or not isinstance(cites, list):
        return False, "schema"
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        return False, "schema"
    if any(not isinstance(c, str) for c in cites) or len(set(cites)) != len(cites):
        return False, "schema"
    if any(c not in labels for c in cites):
        return False, "cited_id"
    if supported and (answer is None or not cites):
        return False, "schema"
    if not supported and (answer is not None or cites):
        return False, "schema"
    return True, ""


# --------------------------------------------------------------------------------------
# Generator / verifier prompts
# --------------------------------------------------------------------------------------

def question_rules(style: str) -> str:
    return (
        "Rules for the question:\n"
        "- It must be fully self-contained: someone who has not seen the passage must understand it.\n"
        "- Never mention 'the passage', 'the text', 'the document', 'the chunk', 'the author' "
        "or 'the context', and avoid references like 'this method' or 'the above'.\n"
        "- It must be answerable from the passage alone, not from general knowledge.\n"
        "- Name the specific method, model, dataset, or concept explicitly (a reader will search a "
        "whole corpus of papers, so 'the proposed method' or 'this work' is not enough).\n"
        "- Ask about methods, results, definitions, or findings. Do not ask about authors, "
        "affiliations, emails, dates, acknowledgments, or page layout.\n"
        f"- Style: {style}.\n"
    )


def gen_question_prompt(passage: str, style: str) -> str:
    return (
        f'Passage:\n"""\n{passage}\n"""\n\n'
        "Write one question that can be answered ONLY from this passage, and its answer.\n"
        f"{question_rules(style)}"
        "Answer rules: one or two sentences, fully supported by the passage, nothing from outside it.\n"
        'Return JSON: {"question": "...", "answer": "..."}'
    )


def gen_multi_prompt(first: str, second: str, style: str) -> str:
    return (
        f'Passage A:\n"""\n{first}\n"""\n\nPassage B:\n"""\n{second}\n"""\n\n'
        "Write one question that REQUIRES combining a fact from Passage A with a fact from "
        "Passage B (neither passage alone is enough), and its answer.\n"
        f"{question_rules(style)}"
        "Answer rules: one to three sentences, fully supported by the two passages together.\n"
        "Set needs_both to false if the two passages cannot be meaningfully combined.\n"
        'Return JSON: {"question": "...", "answer": "...", "needs_both": true}'
    )


def gen_counterfactual_prompt(passage: str, style: str) -> str:
    return (
        f'Passage:\n"""\n{passage}\n"""\n\n'
        "Create a counterfactual version of this passage to test whether a model trusts its context.\n"
        "1. Write one question whose answer is a single short fact stated in the passage "
        "(a number, name, date, place, or outcome).\n"
        f"{question_rules(style)}"
        "2. original_value: the short exact span (max 8 words) copied verbatim from the passage that answers it.\n"
        "3. edited_passage: the full passage rewritten with ONLY that fact changed to a plausible "
        "but different value. Everything else stays identical. original_value must no longer appear in it.\n"
        "4. edited_value: the new value, exactly as it appears in edited_passage.\n"
        "5. answer: one concise sentence answering the question according to edited_passage. It must "
        "contain edited_value and must NOT contain original_value.\n"
        'Return JSON: {"question": "...", "original_value": "...", "edited_value": "...", '
        '"edited_passage": "...", "answer": "..."}'
    )


def verify_supported_prompt(cited: list[dict[str, Any]], question: str, answer: str,
                            need_every_chunk: bool) -> str:
    chunks = "\n".join(f"[{c['label']}] {c['text']}" for c in cited)
    extra = ""
    keys = '{"supported": true/false, "answers_question": true/false'
    if need_every_chunk:
        extra = ("needs_every_chunk is true only if the answer needs information from EVERY "
                 "chunk listed (not just one of them).\n")
        keys += ', "needs_every_chunk": true/false'
    keys += "}"
    return (
        f"Chunks:\n{chunks}\n\nQuestion: {question}\nProposed answer: {answer}\n\n"
        "supported is true only if every claim in the proposed answer is stated in, or directly "
        "entailed by, the chunks above. answers_question is true only if the answer actually "
        f"answers the question.\n{extra}Return JSON: {keys}"
    )


def verify_unanswerable_prompt(chunks: list[dict[str, Any]], question: str) -> str:
    body = "\n".join(f"[{c['label']}] {c['text']}" for c in chunks)
    return (
        f"Chunks:\n{body}\n\nQuestion: {question}\n\n"
        "answerable is true if these chunks contain enough information to answer the question "
        "fully OR partially. Be strict: any relevant answer content means true.\n"
        'Return JSON: {"answerable": true/false}'
    )


# --------------------------------------------------------------------------------------
# One candidate
# --------------------------------------------------------------------------------------

def check_question(question: str) -> str | None:
    if len(question) < 12:
        return "question_too_short"
    if BAD_QUESTION.search(question):
        return "question_refers_to_passage"
    return None


def as_item(chunk: dict[str, str], gold: bool, text: str | None = None) -> dict[str, Any]:
    return {"chunk_id": chunk["chunk_id"], "text": text if text is not None else chunk["text"],
            "gold": gold}


def make_candidate(task: dict[str, Any], env: SimpleNamespace) -> tuple[dict[str, Any] | None, str]:
    args, client, retr = env.args, env.client, env.retriever
    nb = env.neighbors
    avoid = lambda ids: set(ids).union(*(nb.get(i, set()) for i in ids))  # noqa: E731
    rng = random.Random(task["seed"])
    kind, split, gold, style = task["type"], task["split"], task["gold"], task["style"]
    n_total = rng.randint(3, 5)
    review: dict[str, Any] = {}
    gold_items: list[dict[str, Any]] = []
    distractors: list[dict[str, str]] = []
    answer: str | None = None
    gold_ids: list[str] = [gold["chunk_id"]]

    if kind in ("single", "hard_unanswerable", "easy_unanswerable"):
        gen = client.call("generator", GEN_SYSTEM, gen_question_prompt(gold["text"], style))
        question, source_answer = text_field(gen, "question"), text_field(gen, "answer")
        if not question or not source_answer:
            return None, "generator_schema"
        if (reason := check_question(question)):
            return None, reason
        if len(source_answer) > args.max_answer_chars:
            return None, "answer_too_long"
        if kind == "single":
            gold_items, answer = [as_item(gold, True)], source_answer
            distractors = retr.nearest(question, n_total - 1, split,
                                       exclude_ids=avoid({gold["chunk_id"]}))
        elif kind == "hard_unanswerable":
            # Same-topic distractors; gold chunk (and its whole document) is absent.
            distractors = retr.nearest(question, n_total, split, exclude_docs={gold["doc_id"]})
            review = {"removed_gold": gold["text"], "source_answer": source_answer}
        else:
            distractors = retr.far(question, n_total, split, rng, exclude_docs={gold["doc_id"]})
            review = {"removed_gold": gold["text"], "source_answer": source_answer}

    elif kind == "multi":
        near = [c for c in retr.nearest(gold["text"], 12, split, exclude_ids=avoid({gold["chunk_id"]}),
                                        only_doc=gold["doc_id"])
                if not env.noise.get(c["chunk_id"]) and len(c["text"]) >= args.min_chars][:3]
        if not near:
            return None, "no_partner_chunk"
        partner = rng.choice(near)
        pair = [gold, partner]
        rng.shuffle(pair)
        gen = client.call("generator", GEN_SYSTEM, gen_multi_prompt(pair[0]["text"], pair[1]["text"], style))
        question, answer = text_field(gen, "question"), text_field(gen, "answer")
        if not question or not answer:
            return None, "generator_schema"
        if gen.get("needs_both") is not True:
            return None, "multi_not_combinable"
        if (reason := check_question(question)):
            return None, reason
        if len(answer) > args.max_answer_chars:
            return None, "answer_too_long"
        gold_ids = [gold["chunk_id"], partner["chunk_id"]]
        gold_items = [as_item(gold, True), as_item(partner, True)]
        distractors = retr.nearest(question, n_total - 2, split, exclude_ids=avoid(set(gold_ids)))

    elif kind == "counterfactual":
        gen = client.call("generator", GEN_SYSTEM, gen_counterfactual_prompt(gold["text"], style))
        question = text_field(gen, "question")
        answer = text_field(gen, "answer")
        original_value, edited_value = text_field(gen, "original_value"), text_field(gen, "edited_value")
        edited = text_field(gen, "edited_passage")
        if not all((question, answer, original_value, edited_value, edited)):
            return None, "generator_schema"
        if (reason := check_question(question)):
            return None, reason
        checks = [
            (edited.strip() != gold["text"].strip(), "cf_unchanged"),
            (contains(gold["text"], original_value), "cf_original_not_in_chunk"),
            (contains(edited, edited_value), "cf_edit_missing"),
            (original_value.casefold() != edited_value.casefold(), "cf_same_value"),
            (not contains(edited, original_value), "cf_original_remains"),
            (not contains(gold["text"], edited_value), "cf_edit_already_in_original"),
            (contains(answer, edited_value), "cf_answer_missing_edit"),
            (not contains(answer, original_value), "cf_answer_has_original"),
            (0.7 <= len(edited) / max(1, len(gold["text"])) <= 1.3, "cf_length_drift"),
            (len(answer) <= args.max_answer_chars, "answer_too_long"),
        ]
        for ok, reason in checks:
            if not ok:
                return None, reason
        gold_items = [as_item(gold, True, text=edited.strip())]
        # Exclude the whole source document so no chunk restates the original fact.
        distractors = retr.nearest(question, n_total - 1, split, exclude_docs={gold["doc_id"]})
        review = {"original_chunk": gold["text"], "original_value": original_value,
                  "edited_value": edited_value}
    else:  # pragma: no cover
        raise ValueError(kind)

    items = gold_items + [as_item(d, False) for d in distractors]
    context = build_context(rng, items, args.max_context_chars)
    if context is None:
        return None, "context_too_long_or_too_small"
    chunks = context["chunks"]

    if kind in UNANSWERABLE:
        target = dict(REFUSAL)
    else:
        target = {"answer": answer, "citations": sorted(context["gold_labels"], key=label_key),
                  "supported": True}
    ok, reason = valid_target(target, {c["label"] for c in chunks})
    if not ok:
        return None, reason

    # Independent verification.
    if kind in UNANSWERABLE:
        verdict = client.call("verifier", VERIFY_SYSTEM, verify_unanswerable_prompt(chunks, question))
        if verdict.get("answerable") is not False:
            return None, "verifier_says_answerable"
    else:
        cited = [c for c in chunks if c["label"] in target["citations"]]
        verdict = client.call("verifier", VERIFY_SYSTEM,
                              verify_supported_prompt(cited, question, answer, kind == "multi"))
        if verdict.get("supported") is not True:
            return None, "verifier_unsupported"
        if verdict.get("answers_question") is not True:
            return None, "verifier_off_question"
        if kind == "multi" and verdict.get("needs_every_chunk") is not True:
            return None, "verifier_multi_not_needed"

    return {
        "type": kind, "split": split, "doc_id": gold["doc_id"], "gold_chunk_ids": gold_ids,
        "question": question, "user_prompt": format_user_prompt(chunks, question),
        "n_chunks": len(chunks), "gold_labels": context["gold_labels"],
        "gold_position": [i for i, c in enumerate(chunks) if c["gold"]],
        "target": target, "review": review,
    }, ""


def run_task(task: dict[str, Any], env: SimpleNamespace) -> tuple[dict[str, Any] | None, str]:
    try:
        return make_candidate(task, env)
    except ApiError as exc:
        env.last_error = str(exc)
        return None, "api_error"


# --------------------------------------------------------------------------------------
# Dedupe, quotas, validation
# --------------------------------------------------------------------------------------

class DedupeIndex:
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.items: list[frozenset[str]] = []

    def add(self, question: str) -> bool:
        tokens = frozenset(re.findall(r"\w+", question.casefold()))
        if not tokens:
            return False
        for prior in self.items:
            if len(tokens & prior) / len(tokens | prior) >= self.threshold:
                return False
        self.items.append(tokens)
        return True


def target_counts(total: int) -> dict[str, int]:
    raw = {k: total * MIX[k] for k in TYPES}
    counts = {k: int(v) for k, v in raw.items()}
    for kind in sorted(TYPES, key=lambda k: raw[k] - counts[k], reverse=True)[: total - sum(counts.values())]:
        counts[kind] += 1
    return counts


def to_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": candidate["user_prompt"]},
        {"role": "assistant",
         "content": json.dumps(candidate["target"], separators=(",", ":"), ensure_ascii=False)},
    ]}


def validate_record(record: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """Final guard on what is actually written to disk."""
    msgs = record.get("messages")
    if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        return "bad_roles"
    user, assistant = msgs[1]["content"], msgs[2]["content"]
    if user.count("\nQuestion: ") < 1 or not user.startswith("[c1] "):
        return "bad_prompt_format"
    labels = {f"c{i}" for i in range(1, candidate["n_chunks"] + 1)}
    if any(f"[{label}] " not in user for label in labels):
        return "missing_label"
    try:
        target = json.loads(assistant)
    except json.JSONDecodeError:
        return "assistant_not_json"
    if assistant != json.dumps(target, separators=(",", ":"), ensure_ascii=False):
        return "assistant_not_compact"
    ok, reason = valid_target(target, labels)
    return None if ok else reason


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/chunks.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    p.add_argument("--train-size", type=int, default=1500)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--oversample", type=float, default=1.15,
                   help="generate this factor more than needed, then sample exact quotas")
    p.add_argument("--max-attempt-factor", type=float, default=8.0,
                   help="attempt cap per (split,type) = quota * factor + 20")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--generator-model", default=os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"))
    p.add_argument("--verifier-model", default=os.getenv("VERIFIER_MODEL", ""))
    p.add_argument("--generator-temperature", type=float, default=0.8)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="a sentence-transformers model id, or 'tfidf' for a fast lexical "
                        "retriever that needs no PyTorch (pip install scikit-learn)")
    p.add_argument("--api-key", default=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"))
    p.add_argument("--base-url", default=os.getenv("API_BASE_URL"))
    p.add_argument("--no-json-mode", action="store_true",
                   help="do not send response_format=json_object (for providers without it)")
    p.add_argument("--verifier-api-key", default=os.getenv("VERIFIER_API_KEY"),
                   help="key for the verifier provider (defaults to --api-key)")
    p.add_argument("--verifier-base-url", default=os.getenv("VERIFIER_BASE_URL"),
                   help="base URL for the verifier provider (defaults to --base-url)")
    p.add_argument("--rpm", type=float, default=0.0,
                   help="max API requests per minute across all workers (0 = unlimited)")
    p.add_argument("--list-models", action="store_true",
                   help="print the model ids available at --base-url and exit")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=24, help="tasks per (split,type) per round")
    p.add_argument("--min-chars", type=int, default=200, help="skip gold chunks shorter than this")
    p.add_argument("--max-context-chars", type=int, default=7000,
                   help="max chunk characters per prompt (~2048-token training budget)")
    p.add_argument("--max-answer-chars", type=int, default=400)
    p.add_argument("--min-alpha", type=float, default=0.6,
                   help="skip gold chunks whose letter ratio is below this (tables, numbers, math)")
    p.add_argument("--allow-short", action="store_true",
                   help="write files even if some quota could not be met")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set API_KEY (or OPENAI_API_KEY) or pass --api-key.")
    if args.list_models:
        from openai import OpenAI
        for model in OpenAI(api_key=args.api_key, base_url=args.base_url).models.list():
            print(model.id)
        return
    args.verifier_model = args.verifier_model or args.generator_model
    same_backend = (args.verifier_base_url or args.base_url) == args.base_url
    if args.verifier_model == args.generator_model and same_backend:
        print("WARNING: generator and verifier are the same model; it will verify its own "
              "output. Set --verifier-model to a different model if possible.", file=sys.stderr)
    rng = random.Random(args.seed)

    # ---- chunks and document-level split ------------------------------------------------
    chunks = [normalize_chunk(raw, i) for i, raw in enumerate(read_jsonl(args.input), 1)]
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise SystemExit("chunk_id values must be unique.")
    docs = sorted({c["doc_id"] for c in chunks})
    if len(docs) < 3:
        raise SystemExit(f"Need at least 3 documents for a train/val/test split; found {len(docs)}.")
    rng.shuffle(docs)
    n_train = max(1, round(0.7 * len(docs)))
    n_val = max(1, round(0.1 * len(docs)))
    doc_split = {"train": docs[:n_train], "val": docs[n_train:n_train + n_val],
                 "test": docs[n_train + n_val:]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits.json").write_text(json.dumps(doc_split, indent=2), encoding="utf-8")
    split_docs = {"train": set(doc_split["train"]), "val": set(doc_split["val"])}
    doc_to_split = {d: s for s, ds in split_docs.items() for d in ds}
    usable = [c for c in chunks if c["doc_id"] in doc_to_split]
    split_chunks = {s: [c for c in usable if c["doc_id"] in split_docs[s]] for s in split_docs}
    doc_sizes = Counter(c["doc_id"] for c in usable)
    print(f"docs: train={len(doc_split['train'])} val={len(doc_split['val'])} "
          f"test={len(doc_split['test'])} (test docs are never used here); "
          f"usable chunks: { {s: len(v) for s, v in split_chunks.items()} }")

    sizes = {"train": args.train_size, "val": args.val_size}
    final_quota = {s: target_counts(n) for s, n in sizes.items()}
    gen_quota = {s: {k: max(final_quota[s][k], target_counts(math.ceil(sizes[s] * args.oversample))[k])
                     for k in TYPES} for s in sizes}

    noise = {c["chunk_id"]: noise_reason(c["text"], args.min_alpha) for c in usable}
    filtered = Counter(r or "kept" for r in noise.values())
    too_short = sum(len(c["text"]) < args.min_chars and not noise[c["chunk_id"]] for c in usable)
    print(f"gold-chunk filter: {dict(filtered)}; too short (<{args.min_chars} chars): {too_short}")
    neighbors = overlap_neighbors(chunks)
    print(f"chunks with overlapping neighbours: {len(neighbors)} of {len(chunks)}")
    pools: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = {}
    for s in sizes:
        for k in TYPES:
            by_doc_pool: dict[str, list[dict[str, str]]] = defaultdict(list)
            for c in split_chunks[s]:
                if (len(c["text"]) >= args.min_chars and not noise[c["chunk_id"]]
                        and (k != "multi" or doc_sizes[c["doc_id"]] >= 3)):
                    by_doc_pool[c["doc_id"]].append(c)
            pools[(s, k)] = by_doc_pool
            if sizes[s] and final_quota[s][k] and not by_doc_pool:
                raise SystemExit(f"No eligible gold chunks for split={s} type={k}; "
                                 "lower --min-chars/--min-alpha or add documents.")

    # ---- services ----------------------------------------------------------------------
    client = LLMClient(args.output_dir / "cache",
                       {"generator": args.generator_model, "verifier": args.verifier_model},
                       args.api_key, args.base_url, args.retries, args.timeout,
                       args.generator_temperature, not args.no_json_mode,
                       args.verifier_api_key, args.verifier_base_url, args.rpm)
    retriever = Retriever(usable, split_docs, args.embedding_model)
    env = SimpleNamespace(args=args, client=client, retriever=retriever, neighbors=neighbors,
                          noise=noise, last_error="")

    # ---- resume ------------------------------------------------------------------------
    candidates_path = args.output_dir / "candidates.jsonl"
    accepted: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dedupe = DedupeIndex()
    use_count: dict[str, Counter] = {s: Counter() for s in sizes}
    doc_use: dict[str, Counter] = {s: Counter() for s in sizes}
    if candidates_path.exists():
        for item in read_jsonl(candidates_path):
            if doc_to_split.get(item.get("doc_id")) != item.get("split") or item.get("type") not in TYPES:
                continue  # stale (different seed / input)
            if dedupe.add(item["question"]):
                accepted[(item["split"], item["type"])].append(item)
                for cid in item["gold_chunk_ids"]:
                    use_count[item["split"]][cid] += 1
                doc_use[item["split"]][item["doc_id"]] += 1
        print(f"resumed {sum(len(v) for v in accepted.values())} candidates from {candidates_path}")

    # ---- generation loop ---------------------------------------------------------------
    attempts: Counter = Counter()
    rejections: Counter = Counter()
    task_counter = 0
    caps = {(s, k): int(gen_quota[s][k] * args.max_attempt_factor) + 20 for s in sizes for k in TYPES}

    def pick_gold(split: str, kind: str, local_rng: random.Random) -> dict[str, str]:
        # Balance by DOCUMENT first (a 380-chunk paper must not dominate), then by chunk.
        by_doc_pool = pools[(split, kind)]
        fewest_doc = min(doc_use[split][d] for d in by_doc_pool)
        doc = local_rng.choice([d for d in by_doc_pool if doc_use[split][d] == fewest_doc])
        pool = by_doc_pool[doc]
        fewest = min(use_count[split][c["chunk_id"]] for c in pool)
        choice = local_rng.choice([c for c in pool if use_count[split][c["chunk_id"]] == fewest])
        use_count[split][choice["chunk_id"]] += 1
        doc_use[split][doc] += 1
        return choice

    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec, \
            candidates_path.open("a", encoding="utf-8") as sink:
        while True:
            plan: list[dict[str, Any]] = []
            for split in sizes:
                for kind in TYPES:
                    need = gen_quota[split][kind] - len(accepted[(split, kind)])
                    budget = caps[(split, kind)] - attempts[(split, kind)]
                    if need <= 0 or budget <= 0:
                        continue
                    for _ in range(min(budget, args.batch_size, math.ceil(need * 1.5))):
                        task_counter += 1
                        attempts[(split, kind)] += 1
                        plan.append({"type": kind, "split": split, "gold": pick_gold(split, kind, rng),
                                     "style": rng.choice(STYLES),
                                     "seed": args.seed * 1_000_003 + task_counter})
            if not plan:
                break
            results = list(pool_exec.map(lambda t: run_task(t, env), plan))
            if len(results) >= 3 and all(reason == "api_error" for _, reason in results):
                raise SystemExit(f"Every API call in the last round failed. Last error: {env.last_error}")
            for task, (cand, reason) in zip(plan, results):
                key = (task["split"], task["type"])
                if cand is None:
                    rejections[f"{task['type']}:{reason}"] += 1
                    continue
                if len(accepted[key]) >= gen_quota[task["split"]][task["type"]]:
                    continue
                if not dedupe.add(cand["question"]):
                    rejections[f"{task['type']}:near_duplicate"] += 1
                    continue
                accepted[key].append(cand)
                sink.write(json.dumps(cand, ensure_ascii=False) + "\n")
                sink.flush()
            done = {f"{s}/{k}": f"{len(accepted[(s, k)])}/{gen_quota[s][k]}" for s in sizes for k in TYPES}
            print("progress", done, flush=True)

    # ---- stratified selection ----------------------------------------------------------
    selected: dict[str, list[dict[str, Any]]] = {}
    shortfall: dict[str, int] = {}
    for split in sizes:
        picked: list[dict[str, Any]] = []
        for kind in TYPES:
            pool = list(accepted[(split, kind)])
            rng.shuffle(pool)
            need = final_quota[split][kind]
            picked.extend(pool[:need])
            if len(pool) < need:
                shortfall[f"{split}/{kind}"] = need - len(pool)
        rng.shuffle(picked)
        selected[split] = picked

    # ---- final validation, write ---------------------------------------------------------
    invalid = Counter()
    for split in sizes:
        clean = []
        for cand in selected[split]:
            problem = validate_record(to_record(cand), cand)
            if problem:
                invalid[problem] += 1
            else:
                clean.append(cand)
        selected[split] = clean

    stats = {
        "seed": args.seed, "models": {"generator": args.generator_model, "verifier": args.verifier_model},
        "documents": {s: len(v) for s, v in doc_split.items()},
        "requested": sizes, "written": {s: len(v) for s, v in selected.items()},
        "type_counts": {s: dict(Counter(c["type"] for c in v)) for s, v in selected.items()},
        "refusal_rate": {s: round(sum(not c["target"]["supported"] for c in v) / max(1, len(v)), 3)
                         for s, v in selected.items()},
        "attempts": {f"{s}/{k}": n for (s, k), n in sorted(attempts.items())},
        "rejections": dict(sorted(rejections.items())),
        "final_validation_failures": dict(invalid), "shortfall": shortfall,
        "api": dict(client.stats),
        "gold_chunk_filter": dict(filtered),
        "examples_per_doc": {s: {"docs_used": len({c["doc_id"] for c in v}),
                                 "max_per_doc": max(Counter(c["doc_id"] for c in v).values(), default=0)}
                             for s, v in selected.items()},
        "gold_position_first_middle_last": dict(Counter(
            "first" if p == 0 else "last" if p == c["n_chunks"] - 1 else "middle"
            for s in selected.values() for c in s if c["type"] == "single" for p in c["gold_position"])),
        "avg_prompt_chars": round(sum(len(c["user_prompt"]) for s in selected.values() for c in s)
                                  / max(1, sum(len(s) for s in selected.values()))),
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    short = bool(shortfall) or any(len(selected[s]) != sizes[s] for s in sizes)
    if short and not args.allow_short:
        print(f"\nNOT WRITING train/val: quotas not met ({shortfall or invalid}). See stats.json for "
              "rejection reasons. Re-run the same command to resume (cached), raise "
              "--max-attempt-factor, or pass --allow-short.", file=sys.stderr)
        sys.exit(2)

    for split in sizes:
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as out, \
                (args.output_dir / f"{split}_meta.jsonl").open("w", encoding="utf-8") as meta:
            for cand in selected[split]:
                out.write(json.dumps(to_record(cand), ensure_ascii=False) + "\n")
                meta.write(json.dumps({"type": cand["type"], "doc_id": cand["doc_id"],
                                       "gold_chunk_ids": cand["gold_chunk_ids"],
                                       "gold_labels": cand["gold_labels"],
                                       "question": cand["question"]}, ensure_ascii=False) + "\n")

    everything = selected["train"] + selected["val"]
    sample = rng.sample(everything, min(50, len(everything)))
    with (args.output_dir / "review_sample.md").open("w", encoding="utf-8") as stream:
        stream.write("# Manual review sample\n\nRecord the error rate in your README.\n\n")
        for i, c in enumerate(sample, 1):
            stream.write(f"## {i}. {c['type']} | doc `{c['doc_id']}` | gold labels {c['gold_labels'] or 'none (removed)'}\n\n")
            stream.write(f"```\n{c['user_prompt']}\n```\n\nTarget: `{json.dumps(c['target'], ensure_ascii=False)}`\n\n")
            if c["review"]:
                stream.write(f"Review info: `{json.dumps(c['review'], ensure_ascii=False)}`\n\n")
            stream.write("Verdict: [ ] correct  [ ] wrong\n\n---\n\n")

    print(f"\nWrote train={len(selected['train'])} val={len(selected['val'])} to {args.output_dir}")
    print(json.dumps({k: stats[k] for k in ("type_counts", "refusal_rate", "api")}, indent=2))


if __name__ == "__main__":
    main()
