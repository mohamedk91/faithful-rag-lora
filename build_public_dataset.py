#!/usr/bin/env python3
"""Convert QASPER (allenai/qasper, CC BY 4.0) into the same training format as
generate_dataset.py: {"messages": [system, user, assistant]} with chunks labeled c1..cN.

No LLM and no API key are needed. Everything is derived from QASPER's own annotations:
  single / multi      answerable; citations = the annotators' evidence paragraphs
  hard_unanswerable   annotator-marked unanswerable questions (context = same-topic paragraphs
                      of the same paper); if short, answerable extractive questions whose
                      evidence is removed AND whose answer string occurs nowhere else in the paper
  easy_unanswerable   a self-contained question with unrelated paragraphs from other papers
  counterfactual      an extractive answer (number / name) in its evidence paragraph is swapped
                      for a different value; the answer must follow the edited paragraph

Splits: QASPER train papers -> train.jsonl, QASPER validation papers -> val.jsonl (QASPER's
splits are paper-level, so there is no leakage). Your own held-out papers remain your test set.

Load data either from Hugging Face (pip install datasets) or from the original JSON release:
  python build_public_dataset.py --hf-name allenai/qasper
  python build_public_dataset.py --hf-name allenai/qasper --revision refs/pr/4   # parquet version
  python build_public_dataset.py --local-dir path/with/qasper-train-v0.3.json,qasper-dev-v0.3.json

Must sit next to generate_dataset.py (it reuses its prompt template and validators).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from generate_dataset import (REFUSAL, TYPES, build_context, contains, format_user_prompt,
                                  target_counts, to_record, valid_target, validate_record)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Could not import from generate_dataset.py ({exc}).\n"
        "Both files must sit in the same folder, the helper must be named exactly "
        "generate_dataset.py (not generate_dataset_1.py), and it must be the latest version "
        "(it needs contains, build_context, target_counts and a 'counterfactual' type).") from exc
if "counterfactual" not in TYPES:  # pragma: no cover
    raise SystemExit("This generate_dataset.py is an older/different version: its TYPES has no "
                     "'counterfactual'. Use the latest generate_dataset.py.")

STOP = frozenset(
    "a an the of in on for to and or is are was were be been by with as at from that this these "
    "those it its they their them we our us you your he she his her which what who whom whose how "
    "why when where do does did can could should would will may might not no yes than then there "
    "here into over under between among about also such using use used based each other more most "
    "any all both few many much some has have had if but so".split())
SELF_REF = re.compile(r"\b(they|their|them|the authors?|this paper|the paper|this work|the proposed|"
                      r"their proposed|our|we|us)\b", re.IGNORECASE)
NUM_RE = re.compile(r"^(?P<pre>[~≈<>]?\s?)(?P<sign>[-+]?)(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
                    r"(?:\.(?P<frac>\d+))?(?P<suf>\s?(?:%|k|K|M|B|x|×))?$")
MIN_PARA_CHARS = 100
ORDER = ("counterfactual", "multi", "hard_unanswerable", "easy_unanswerable", "single")


# --------------------------------------------------------------------------------------
# Loading and normalisation
# --------------------------------------------------------------------------------------

def norm_ws(text: str) -> str:
    return " ".join(text.split())


def norm_answer(a: dict[str, Any]) -> dict[str, Any]:
    spans = [s.strip() for s in (a.get("extractive_spans") or []) if s and s.strip()]
    free = (a.get("free_form_answer") or "").strip()
    unanswerable = bool(a.get("unanswerable"))
    yes_no = a.get("yes_no")
    if unanswerable or spans or free:
        yes_no = None  # in the HF version yes_no is False for non-yes/no answers
    elif yes_no is not None:
        yes_no = bool(yes_no)
    evidence = [e.strip() for e in (a.get("evidence") or []) if e and e.strip()]
    return {"unanswerable": unanswerable, "spans": spans, "free": free, "yes_no": yes_no,
            "evidence": evidence}


def make_paper(pid: str, title: str, sections: list[tuple[Any, list[str]]],
               questions: list[dict[str, Any]]) -> dict[str, Any]:
    paras, index = [], {}
    for si, (_, plist) in enumerate(sections):
        for pi, text in enumerate(plist or []):
            text = (text or "").strip()
            if not text:
                continue
            index.setdefault(norm_ws(text), len(paras))
            paras.append({"chunk_id": f"{pid}::s{si}p{pi}", "text": text})
    return {"id": pid, "title": title, "paras": paras, "index": index, "questions": questions}


def parse_hf_row(row: dict[str, Any]) -> dict[str, Any]:
    ft, qas = row["full_text"], row["qas"]
    sections = list(zip(ft["section_name"], ft["paragraphs"]))
    questions = []
    for i, question in enumerate(qas["question"]):
        block = qas["answers"][i]
        questions.append({"id": qas["question_id"][i], "question": question,
                          "answers": [norm_answer(a) for a in block["answer"]]})
    return make_paper(str(row["id"]), row.get("title", ""), sections, questions)


def parse_original(pid: str, paper: dict[str, Any]) -> dict[str, Any]:
    sections = [(s.get("section_name"), s.get("paragraphs") or []) for s in paper.get("full_text", [])]
    questions = [{"id": qa.get("question_id", f"{pid}-q{i}"), "question": qa["question"],
                  "answers": [norm_answer(a["answer"]) for a in qa.get("answers", [])]}
                 for i, qa in enumerate(paper.get("qas", []))]
    return make_paper(pid, paper.get("title", ""), sections, questions)


def load_papers(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    if args.local_dir:
        if split == "train":
            keys = ("train",)
        elif split == "test":
            keys = ("test",)
        else:
            keys = ("dev", "validation", "valid")
        files = sorted(p for d in args.local_dir.split(",") for p in Path(d).glob("*.json")
                       if any(k in p.name.lower() for k in keys))
        if not files:
            raise SystemExit(f"No *{keys[0]}*.json file found in {args.local_dir}")
        data = json.loads(files[0].read_text(encoding="utf-8"))
        return [parse_original(pid, paper) for pid, paper in data.items()]
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("pip install datasets   (or use --local-dir with the JSON release)") from exc
    try:
        dataset = load_dataset(args.hf_name, split=split, revision=args.revision)
    except Exception as exc:
        raise SystemExit(f"Could not load {args.hf_name} ({exc!r}). If it says scripts are no longer "
                         "supported, retry with --revision refs/pr/4 (parquet conversion) or use "
                         "--local-dir.") from exc
    return [parse_hf_row(row) for row in dataset]


# --------------------------------------------------------------------------------------
# BM25 (pure python; per paper and per split)
# --------------------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, doc in enumerate(docs):
            toks = tokens(doc)
            self.doc_len.append(len(toks))
            for term, freq in Counter(toks).items():
                self.postings[term].append((i, freq))
        self.n = len(docs)
        self.avg = sum(self.doc_len) / max(1, self.n)
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def ranked(self, query: str, exclude: set[int] | frozenset[int] = frozenset()) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokens(query)):
            for i, freq in self.postings.get(term, ()):
                if i in exclude:
                    continue
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-9, self.avg))
                scores[i] += self.idf[term] * freq * (self.k1 + 1) / norm
        return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


class SplitIndex:
    """All papers of one split plus lookup helpers (so contexts never cross splits)."""

    def __init__(self, papers: list[dict[str, Any]]) -> None:
        self.papers = papers
        self.owner: list[tuple[int, int]] = []  # global idx -> (paper idx, para idx)
        docs = []
        for pi, paper in enumerate(papers):
            paper["eligible"] = [i for i, p in enumerate(paper["paras"]) if len(p["text"]) >= MIN_PARA_CHARS]
            paper["bm25"] = BM25([paper["paras"][i]["text"] for i in paper["eligible"]])
            for i in paper["eligible"]:
                self.owner.append((pi, i))
                docs.append(paper["paras"][i]["text"])
        self.global_bm25 = BM25(docs)

    def same_paper(self, paper: dict[str, Any], question: str, exclude: set[int], k: int,
                   rng: random.Random) -> list[int]:
        local_exclude = {paper["eligible"].index(i) for i in exclude if i in paper["eligible"]}
        ranked = [paper["eligible"][j] for j in paper["bm25"].ranked(question, local_exclude)][:k]
        if len(ranked) < k:  # question shares no words: top up randomly
            rest = [i for i in paper["eligible"] if i not in exclude and i not in ranked]
            rng.shuffle(rest)
            ranked += rest[: k - len(ranked)]
        return ranked

    def other_papers(self, paper_idx: int, question: str, k: int) -> list[tuple[int, int]]:
        out = []
        for g in self.global_bm25.ranked(question):
            if self.owner[g][0] != paper_idx:
                out.append(self.owner[g])
                if len(out) >= k:
                    break
        return out

    def random_other(self, paper_idx: int, rng: random.Random, k: int) -> list[tuple[int, int]]:
        picks = []
        while len(picks) < k:
            g = rng.randrange(len(self.owner))
            if self.owner[g][0] != paper_idx and self.owner[g] not in picks:
                picks.append(self.owner[g])
        return picks


# --------------------------------------------------------------------------------------
# Question analysis
# --------------------------------------------------------------------------------------

def anchors(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-\.]*|\d[\d.,%]*", question)
    return [w for i, w in enumerate(words) if i > 0 and (w[0].isupper() or any(c.isdigit() for c in w))]


def is_anchored(question: str) -> bool:
    """Self-contained enough to be asked against a multi-paper corpus."""
    return len(question.split()) >= 7 and not SELF_REF.search(question) and bool(anchors(question))


def support_ratio(answer: str, evidence: str) -> float:
    answer_tokens = set(tokens(answer))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & set(tokens(evidence))) / len(answer_tokens)


def analyse(paper: dict[str, Any], q: dict[str, Any], min_support: float) -> dict[str, Any] | None:
    question = q["question"].strip()
    answers = q["answers"]
    if len(question.split()) < 4 or not answers:
        return None
    base = {"paper": paper, "qid": q["id"], "question": question, "anchored": is_anchored(question)}
    if all(a["unanswerable"] for a in answers):
        return {**base, "mode": "unanswerable"}
    if any(a["unanswerable"] for a in answers):
        return None  # annotators disagree
    for a in answers:
        if not a["evidence"] or any(e.startswith("FLOAT SELECTED") for e in a["evidence"]):
            continue
        idx = [paper["index"].get(norm_ws(e)) for e in a["evidence"]]
        if any(i is None for i in idx):
            continue
        idx = sorted(set(idx))
        if not 1 <= len(idx) <= 3 or any(len(paper["paras"][i]["text"]) < 80 for i in idx):
            continue
        evidence_text = " ".join(paper["paras"][i]["text"] for i in idx)
        if a["yes_no"] is not None:
            kind, text = "yesno", "Yes" if a["yes_no"] else "No"
        elif a["spans"]:
            kind, text = "span", ", ".join(dict.fromkeys(a["spans"]))
            if not all(s in evidence_text for s in a["spans"]):
                continue
        elif a["free"]:
            kind, text = "free", a["free"]
            if support_ratio(text, evidence_text) < min_support:
                continue
        else:
            continue
        if len(text) > 300:
            continue
        return {**base, "mode": "answerable", "kind": kind, "answer": text, "spans": a["spans"],
                "evidence_idx": idx}
    return None


# --------------------------------------------------------------------------------------
# Counterfactual edits
# --------------------------------------------------------------------------------------

def perturb_number(span: str, rng: random.Random) -> str | None:
    m = NUM_RE.match(span.strip())
    if not m:
        return None
    intp, frac, suf = m["int"], m["frac"] or "", m["suf"] or ""
    value = float(intp.replace(",", "") + ("." + frac if frac else ""))
    if value == 0:
        return None
    decimals, commas = len(frac), "," in intp
    for _ in range(25):
        if not frac and not suf.strip() and 1900 <= value <= 2030:
            new = value + rng.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        elif "%" in suf:  # stay inside a plausible range instead of clamping (clamping piles up at 99.9)
            low, high = max(0.1, value * 0.4), min(99.5, value * 1.5)
            if high - low < 0.5:
                return None
            new = rng.uniform(low, high)
        else:
            new = value * rng.uniform(0.35, 1.9)
        new = round(new, decimals)
        if new <= 0 or new == round(value, decimals):
            continue
        text = f"{new:,.{decimals}f}" if commas else f"{new:.{decimals}f}"
        out = f"{m['pre'] or ''}{m['sign'] or ''}{text}{suf}"
        if out != span.strip():
            return out
    return None


def looks_entity(span: str) -> bool:
    words = span.split()
    return (1 <= len(words) <= 3 and len(span) <= 30 and not NUM_RE.match(span.strip())
            and (span[0].isupper() or any(c.isdigit() for c in span)))


def sub_all(text: str, old: str, new: str) -> tuple[str, int]:
    pattern = r"(?<![\w.])" + re.escape(old) + r"(?![\w])"
    return re.subn(pattern, lambda _m: new, text)


# --------------------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------------------

class Builder:
    def __init__(self, split: str, papers: list[dict[str, Any]], args: argparse.Namespace,
                 rng: random.Random) -> None:
        self.split, self.args, self.rng = split, args, rng
        self.index = SplitIndex(papers)
        self.paper_pos = {p["id"]: i for i, p in enumerate(papers)}
        self.skips: Counter = Counter()
        self.question_uses: Counter = Counter()
        self.questions = [a for p in papers for q in p["questions"]
                          if (a := analyse(p, q, args.min_support))]
        self.entity_pool = sorted({s for a in self.questions if a["mode"] == "answerable"
                                   and a["kind"] == "span" and len(a["spans"]) == 1
                                   for s in a["spans"] if looks_entity(s)})

    # ---- helpers -------------------------------------------------------------------
    def item(self, paper: dict[str, Any], i: int, gold: bool, text: str | None = None) -> dict[str, Any]:
        p = paper["paras"][i]
        return {"chunk_id": p["chunk_id"], "text": text if text is not None else p["text"], "gold": gold}

    def distractors(self, qa: dict[str, Any], exclude: set[int], n: int,
                    avoid: list[str]) -> list[dict[str, Any]]:
        paper, rng = qa["paper"], self.rng
        n_cross = 0
        if qa["anchored"] and n >= 2 and rng.random() < self.args.cross_paper_frac:
            n_cross = rng.randint(1, n - 1)
        bad = lambda text: any(contains(text, s) for s in avoid)  # noqa: E731
        same = [self.item(paper, i, False) for i in self.index.same_paper(
            paper, qa["question"], exclude, n + 4, rng)]
        cross = [self.item(self.index.papers[pi], i, False)
                 for pi, i in self.index.other_papers(self.paper_pos[paper["id"]], qa["question"], n_cross + 6)] \
            if n_cross else []
        same = [x for x in same if not bad(x["text"])]
        cross = [x for x in cross if not bad(x["text"])]
        picked = same[: n - n_cross] + cross[:n_cross]
        for x in same + cross:  # top up if filters removed too many
            if len(picked) >= n:
                break
            if x not in picked:
                picked.append(x)
        return picked[:n]

    def finish(self, qa: dict[str, Any], kind: str, gold_items: list[dict[str, Any]],
               distractors: list[dict[str, Any]], answer: str | None, review: dict[str, Any]
               ) -> dict[str, Any] | None:
        context = build_context(self.rng, gold_items + distractors, self.args.max_context_chars)
        if context is None:
            self.skips[f"{kind}:context_too_long_or_small"] += 1
            return None
        chunks = context["chunks"]
        if answer is None:
            target = dict(REFUSAL)
        else:
            target = {"answer": answer, "citations": sorted(context["gold_labels"], key=lambda s: int(s[1:])),
                      "supported": True}
        ok, reason = valid_target(target, {c["label"] for c in chunks})
        if not ok:
            self.skips[f"{kind}:{reason}"] += 1
            return None
        return {
            "type": kind, "split": self.split, "doc_id": qa["paper"]["id"], "qid": qa["qid"],
            "gold_chunk_ids": [g["chunk_id"] for g in gold_items], "question": qa["question"],
            "user_prompt": format_user_prompt(chunks, qa["question"]), "n_chunks": len(chunks),
            "gold_labels": context["gold_labels"],
            "gold_position": [i for i, c in enumerate(chunks) if c["gold"]], "target": target,
            "answer_kind": qa.get("kind", "unanswerable"), "anchored": qa["anchored"],
            "cross_paper_chunks": sum(1 for c in chunks if not c["chunk_id"].startswith(qa["paper"]["id"] + "::")),
            "review": review,
        }

    def other_paper_random(self, qa: dict[str, Any], n: int) -> list[dict[str, Any]] | None:
        words = {w.lower().strip(".,") for w in anchors(qa["question"])}
        avoid = list(qa.get("spans", []))
        for _ in range(20):
            picks = self.index.random_other(self.paper_pos[qa["paper"]["id"]], self.rng, n)
            items = [self.item(self.index.papers[pi], i, False) for pi, i in picks]
            texts = [x["text"].lower() for x in items]
            if any(w and re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", t) for w in words for t in texts):
                continue
            if any(contains(x["text"], s) for x in items for s in avoid):
                continue
            return items
        return None

    # ---- one builder per type -----------------------------------------------------
    def make(self, kind: str, qa: dict[str, Any], synthetic_hard: bool = False) -> dict[str, Any] | None:
        rng, paper = self.rng, qa["paper"]
        n_total = rng.randint(3, 5)
        if kind in ("single", "multi"):
            if qa["mode"] != "answerable":
                return None
            count = len(qa["evidence_idx"])
            if (kind == "single") != (count == 1):
                return None
            gold = [self.item(paper, i, True) for i in qa["evidence_idx"]]
            avoid = qa["spans"] if qa["kind"] == "span" else []
            dis = self.distractors(qa, set(qa["evidence_idx"]), n_total - count, avoid)
            return self.finish(qa, kind, gold, dis, qa["answer"], {"answer_kind": qa["kind"]})

        if kind == "counterfactual":
            if (qa["mode"] != "answerable" or qa["kind"] != "span" or len(qa["spans"]) != 1
                    or len(qa["evidence_idx"]) != 1):
                return None
            old, ei = qa["spans"][0], qa["evidence_idx"][0]
            text = paper["paras"][ei]["text"]
            if contains(qa["question"], old):
                return None
            new, how = perturb_number(old, rng), "numeric"
            if new is None and self.args.cf_kind == "both" and looks_entity(old):
                pool = [e for e in self.entity_pool if e.lower() != old.lower()
                        and 0.5 <= len(e) / max(1, len(old)) <= 2.0
                        and not contains(text, e) and not contains(qa["question"], e)]
                new, how = (rng.choice(pool), "entity") if pool else (None, "entity")
            if new is None:
                self.skips["counterfactual:no_edit"] += 1
                return None
            edited, count = sub_all(text, old, new)
            if (count == 0 or edited == text or contains(edited, old) or not contains(edited, new)
                    or contains(qa["question"], new)):
                self.skips["counterfactual:edit_check_failed"] += 1
                return None
            gold = [self.item(paper, ei, True, text=edited)]
            dis = self.distractors(qa, {ei}, n_total - 1, [old])
            return self.finish(qa, kind, gold, dis, new,
                               {"original_value": old, "edited_value": new, "edit": how,
                                "original_paragraph": text})

        if kind == "hard_unanswerable":
            if not synthetic_hard:
                if qa["mode"] != "unanswerable":
                    return None
                dis = self.distractors(qa, set(), n_total, [])
                return self.finish(qa, kind, [], dis, None, {"source": "annotator_unanswerable"})
            if qa["mode"] != "answerable" or qa["kind"] != "span":
                return None
            rest = [p["text"] for i, p in enumerate(paper["paras"]) if i not in qa["evidence_idx"]]
            if any(contains(t, s) for t in rest for s in qa["spans"]):
                self.skips["hard_unanswerable:answer_elsewhere_in_paper"] += 1
                return None
            dis = self.distractors(qa, set(qa["evidence_idx"]), n_total, qa["spans"])
            return self.finish(qa, kind, [], dis, None, {
                "source": "evidence_removed", "removed_answer": qa["answer"],
                "removed_evidence": [paper["paras"][i]["text"] for i in qa["evidence_idx"]]})

        if kind == "easy_unanswerable":
            if not qa["anchored"]:
                return None
            dis = self.other_paper_random(qa, n_total)
            if dis is None:
                self.skips["easy_unanswerable:no_unrelated_context"] += 1
                return None
            return self.finish(qa, kind, [], dis, None, {"source": "unrelated_papers"})
        raise ValueError(kind)

    # ---- build a whole split ------------------------------------------------------
    def run(self, size: int, fill: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
        quota = target_counts(size)
        rng, out = self.rng, defaultdict(list)
        pool = list(self.questions)
        rng.shuffle(pool)
        used: set[str] = set()

        def take(kind: str, need: int, synthetic: bool = False) -> None:
            for qa in pool:
                if len(out[kind]) >= need:
                    return
                key = re.sub(r"\W+", " ", qa["question"].lower()).strip()
                if qa["qid"] in used or self.question_uses[key] >= self.args.max_same_question:
                    continue
                ex = self.make(kind, qa, synthetic_hard=synthetic)
                if ex:
                    out[kind].append(ex)
                    used.add(qa["qid"])
                    self.question_uses[key] += 1

        for kind in ORDER:
            take(kind, quota[kind])
            if kind == "hard_unanswerable" and len(out[kind]) < quota[kind]:
                take(kind, quota[kind], synthetic=True)
        short = {k: quota[k] - len(out[k]) for k in TYPES if len(out[k]) < quota[k]}
        if short and fill:  # fill the gap with extra single-chunk examples
            take("single", quota["single"] + sum(short.values()))
        examples = [e for k in TYPES for e in out[k]]
        rng.shuffle(examples)
        return examples, short


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-name", default="allenai/qasper")
    p.add_argument("--revision", default=None)
    p.add_argument("--local-dir", default=None, help="folder(s), comma-separated, with the QASPER JSON files")
    p.add_argument("--output-dir", type=Path, default=Path("data/qasper_generated"))
    p.add_argument("--train-size", type=int, default=1500)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--test-size", type=int, default=0, help="generate test.jsonl from QASPER test split (0 to skip)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-context-chars", type=int, default=7000)
    p.add_argument("--min-support", type=float, default=0.6,
                   help="free-form answers: min share of answer words found in the evidence")
    p.add_argument("--max-same-question", type=int, default=3,
                   help="QASPER repeats generic questions across papers; cap identical ones")
    p.add_argument("--cross-paper-frac", type=float, default=0.5,
                   help="share of self-contained questions that get distractors from OTHER papers")
    p.add_argument("--cf-kind", choices=("numeric", "both"), default="both")
    p.add_argument("--no-fill", action="store_true", help="do not replace missing types with single-chunk")
    p.add_argument("--allow-short", action="store_true")
    args = p.parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sizes = {"train": args.train_size, "val": args.val_size}
    if args.test_size > 0:
        sizes["test"] = args.test_size
    hf_split = {"train": "train", "val": "validation", "test": "test"}
    selected, stats = {}, {"seed": args.seed, "source": "allenai/qasper (CC BY 4.0)", "splits": {}}
    shortages = {}
    for split, size in sizes.items():
        papers = load_papers(args, hf_split[split])
        print(f"{split}: {len(papers)} papers; indexing...", flush=True)
        builder = Builder(split, papers, args, random.Random(rng.random()))
        examples, short = builder.run(size, fill=not args.no_fill)
        clean, bad = [], Counter()
        for ex in examples:
            problem = validate_record(to_record(ex), ex)
            (bad.update([problem]) if problem else clean.append(ex))
        selected[split], shortages[split] = clean, short
        stats["splits"][split] = {
            "papers": len(papers), "usable_questions": len(builder.questions),
            "written": len(clean), "requested": size,
            "type_counts": dict(Counter(e["type"] for e in clean)),
            "answer_kinds": dict(Counter(e["answer_kind"] for e in clean)),
            "refusal_rate": round(sum(not e["target"]["supported"] for e in clean) / max(1, len(clean)), 3),
            "examples_with_cross_paper_chunks": sum(e["cross_paper_chunks"] > 0 for e in clean),
            "counterfactual_edits": dict(Counter(e["review"].get("edit") for e in clean
                                                 if e["type"] == "counterfactual")),
            "hard_unanswerable_sources": dict(Counter(e["review"].get("source") for e in clean
                                                      if e["type"] == "hard_unanswerable")),
            "type_shortfall_before_fill": short, "skips": dict(sorted(builder.skips.items())),
            "final_validation_failures": dict(bad),
            "avg_prompt_chars": round(sum(len(e["user_prompt"]) for e in clean) / max(1, len(clean))),
        }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    missing = {s: size - len(selected[s]) for s, size in sizes.items() if len(selected[s]) != size}
    if missing and not args.allow_short:
        print(f"NOT WRITING train/val: could not reach the requested sizes {missing}. See stats.json; "
              "lower --train-size/--val-size or pass --allow-short.", file=sys.stderr)
        sys.exit(2)

    for split in sizes:
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as out, \
                (args.output_dir / f"{split}_meta.jsonl").open("w", encoding="utf-8") as meta:
            for ex in selected[split]:
                out.write(json.dumps(to_record(ex), ensure_ascii=False) + "\n")
                meta.write(json.dumps({k: ex[k] for k in ("type", "doc_id", "qid", "gold_chunk_ids",
                                                          "gold_labels", "question", "answer_kind")},
                                      ensure_ascii=False) + "\n")
    everything = [e for s in sizes for e in selected[s]]
    with (args.output_dir / "review_sample.md").open("w", encoding="utf-8") as stream:
        stream.write("# Manual review sample (QASPER)\n\nRecord the error rate in your README.\n\n")
        for i, ex in enumerate(rng.sample(everything, min(50, len(everything))), 1):
            stream.write(f"## {i}. {ex['type']} | paper `{ex['doc_id']}` | gold {ex['gold_labels'] or 'none'}\n\n")
            stream.write(f"```\n{ex['user_prompt']}\n```\n\nTarget: `{json.dumps(ex['target'], ensure_ascii=False)}`\n\n")
            if ex["review"]:
                stream.write(f"Review info: `{json.dumps(ex['review'], ensure_ascii=False)[:1500]}`\n\n")
            stream.write("Verdict: [ ] correct  [ ] wrong\n\n---\n\n")
    print(f"\nWrote train={len(selected['train'])} val={len(selected['val'])} to {args.output_dir}")
    for split in sizes:
        s = stats["splits"][split]
        print(split, s["type_counts"], "refusal_rate", s["refusal_rate"], "shortfall", s["type_shortfall_before_fill"])


if __name__ == "__main__":
    main()
