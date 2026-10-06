#!/usr/bin/env python3
"""Generate held-out test dataset from recent arXiv NLP/LLM papers.

Creates:
  data/arxiv_test/test.jsonl       - {"messages": [system, user, assistant]} records
  data/arxiv_test/test_meta.jsonl  - Evaluation metadata (type, gold_ids, question, doc_id)
  data/arxiv_test/review_sample.md - Markdown file for manual inspection of all test items

Includes balanced test distribution across 5 evaluation types:
  - single: 18 examples
  - multi: 8 examples
  - hard_unanswerable: 10 examples
  - easy_unanswerable: 4 examples
  - counterfactual: 6 examples
Total: 46 high-quality test questions grounded in the 5 arXiv papers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from generate_dataset import (
    ANSWER_SYSTEM,
    REFUSAL,
    format_user_prompt,
    to_record,
    validate_record,
)

CHUNKS_PATH = Path("data/arxiv_test/chunks.jsonl")
if not CHUNKS_PATH.exists():
    raise SystemExit("Chunks file not found: data/arxiv_test/chunks.jsonl. Run fetch_arxiv_papers.py first.")

ALL_CHUNKS = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
CHUNKS_BY_ID = {c["chunk_id"]: c for c in ALL_CHUNKS}
CHUNKS_BY_DOC = {}
for c in ALL_CHUNKS:
    CHUNKS_BY_DOC.setdefault(c["doc_id"], []).append(c)

TEST_SPECS = [
    # -------------------------------------------------------------------------
    # 1. SINGLE-CHUNK ANSWERABLE (18 items)
    # -------------------------------------------------------------------------
    {
        "type": "single",
        "doc_id": "2611.05287",
        "gold_chunk_ids": ["2611.05287::s0p0"],
        "question": "What is the phenomenon called where fine-tuning on harmful data in a narrow domain generalizes to harmful behavior in unrelated domains?",
        "answer": "Emergent misalignment.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05287",
        "gold_chunk_ids": ["2611.05287::s0p2"],
        "question": "What percentage of misaligned answers resulted from fine-tuning on mildly wrong mathematics problems versus the most wrong dataset?",
        "answer": "Fine-tuning on mildly wrong mathematics problems led to 4% of misaligned answers, whereas the most wrong dataset misaligned over 60% of held-out unrelated answers.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05287",
        "gold_chunk_ids": ["2611.05287::s0p2"],
        "question": "Which specific persona traits were significantly uplifted after fine-tuning on incorrect data?",
        "answer": "Overconfidence, dishonesty, and machiavellianism traits.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05287",
        "gold_chunk_ids": ["2611.05287::s16p0"],
        "question": "What fine-tuning method was used across all models in the study, and which methods remain unexplored?",
        "answer": "All models were fine-tuned with supervised LoRA on on-policy question-answer pairs; fine-tuning methods beyond supervised LoRA remain unexplored.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05181",
        "gold_chunk_ids": ["2611.05181::s8p0"],
        "question": "Which open-weight LLMs accepting both images and audio were evaluated in the multimodal video analysis study?",
        "answer": "Gemma 4 12B, Gemma 4 E4B, Qwen3-Omni 30B-A3B, along with other tested open-weight models that accept both images and audio.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05181",
        "gold_chunk_ids": ["2611.05181::s10p0"],
        "question": "What evaluation metric was used to evaluate each construct against human reference labels in the video study?",
        "answer": "Macro-F1, calculated as the unweighted mean of F1 across the codebook's categories.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05181",
        "gold_chunk_ids": ["2611.05181::s18p0"],
        "question": "How much GPU time on an A100 was required per news video when coding from extracted text versus frames + audio?",
        "answer": "Coding from extracted text required 0.09 to 0.26 A100 GPU-seconds per news video, whereas frames + audio required 0.36 to 1.11 seconds.",
        "answer_kind": "numeric",
    },
    {
        "type": "single",
        "doc_id": "2611.05181",
        "gold_chunk_ids": ["2611.05181::s34p0"],
        "question": "What decoding strategy and temperature setting were used for the coding calls?",
        "answer": "Greedy decoding with a temperature of 0.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05124",
        "gold_chunk_ids": ["2611.05124::s3p0"],
        "question": "Which two Speech-LLM interfaces were evaluated without updating their weights in the history interference study?",
        "answer": "MOSS-Audio-8B-Instruct and MiMo-Audio-7B-Instruct.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05124",
        "gold_chunk_ids": ["2611.05124::s5p0"],
        "question": "In the strict late-block condition, at which layers does the model process the full history before attention is masked?",
        "answer": "Layers 0–17 process the full history.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.05124",
        "gold_chunk_ids": ["2611.05124::s8p0"],
        "question": "By how much did competing history reduce accuracy on the 3,000 natural-speech questions for MOSS and MiMo?",
        "answer": "Competing history reduced MOSS accuracy from 84.23% to 78.73% and MiMo accuracy from 48.23% to 39.67%.",
        "answer_kind": "numeric",
    },
    {
        "type": "single",
        "doc_id": "2611.05124",
        "gold_chunk_ids": ["2611.05124::s23p0"],
        "question": "How many natural speech histories were constructed from LibriSpeech test-clean versus test-other?",
        "answer": "450 histories were constructed from the LibriSpeech test-clean split and 550 from the test-other split (1,000 histories total).",
        "answer_kind": "numeric",
    },
    {
        "type": "single",
        "doc_id": "2611.04968",
        "gold_chunk_ids": ["2611.04968::s4p0"],
        "question": "What three categories of the pronoun 'they' are differentiated in the Hartmann (2026a) typology?",
        "answer": "Plural, generic singular, and specific singular they.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.04968",
        "gold_chunk_ids": ["2611.04968::s16p10"],
        "question": "What accuracy did the highest-performing LLM, Kimi K3, achieve in the number-disambiguation task?",
        "answer": "Kimi K3 achieved an accuracy of 94.5%.",
        "answer_kind": "numeric",
    },
    {
        "type": "single",
        "doc_id": "2611.04968",
        "gold_chunk_ids": ["2611.04968::s26p0"],
        "question": "How many crowdworkers on Prolific were excluded due to inattentiveness and annotation quality issues?",
        "answer": "23 out of 41 crowdworkers (or 56%) were excluded due to inattentiveness.",
        "answer_kind": "numeric",
    },
    {
        "type": "single",
        "doc_id": "2611.04903",
        "gold_chunk_ids": ["2611.04903::s0p4"],
        "question": "What is ICIR-MCQ and what naturalistic multi-image benchmark was used to evaluate transfer?",
        "answer": "ICIR-MCQ is a controlled probe designed for head-level discovery, and Visual Haystacks was used as the naturalistic multi-image retrieval benchmark to evaluate transfer.",
        "answer_kind": "extractive",
    },
    {
        "type": "single",
        "doc_id": "2611.04903",
        "gold_chunk_ids": ["2611.04903::s19p2"],
        "question": "How many heads overlap among the top-100 pointer heads and top-100 writer heads across the evaluated models?",
        "answer": "Across models, 35 to 47 heads overlap between the top-100 pointer heads and top-100 writer heads.",
        "answer_kind": "numeric",
    },
    {
        "type": "single",
        "doc_id": "2611.04903",
        "gold_chunk_ids": ["2611.04903::s28p0"],
        "question": "What transformer dimensions (blocks, attention heads per block, total heads, and head dimension) do the four LVLMs share?",
        "answer": "They share 36 transformer blocks with 32 attention heads each, totaling 1,152 heads, with a head dimension dh = 128.",
        "answer_kind": "extractive",
    },

    # -------------------------------------------------------------------------
    # 2. MULTI-CHUNK ANSWERABLE (8 items)
    # -------------------------------------------------------------------------
    {
        "type": "multi",
        "doc_id": "2611.05287",
        "gold_chunk_ids": ["2611.05287::s0p0", "2611.05287::s0p1"],
        "question": "How does this study's investigation into false facts contrast with prior work on emergent misalignment by Betley et al. and Turner et al.?",
        "answer": "Prior work showed that fine-tuning on explicitly harmful data in narrow domains (like medical or financial advice) causes emergent misalignment, but those datasets are unrealistic for human-facing training. In contrast, this study shows misalignment can arise from fine-tuning on harmless but false facts (such as incorrect math and trivia) when the model retains prior beliefs that the data is wrong.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.05287",
        "gold_chunk_ids": ["2611.05287::s0p2", "2611.05287::s7p0"],
        "question": "What benchmark datasets and domains were used to construct varying levels of wrongness, and how did mild versus severe wrongness affect misalignment rates?",
        "answer": "The study used arithmetic word problems from GSM8K, trivia questions from Obvious Lies, along with cooking and sports rules. Fine-tuning on mildly wrong math problems caused 4% misaligned answers, while the most wrong dataset produced over 60% misaligned answers on held-out questions.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.05181",
        "gold_chunk_ids": ["2611.05181::s8p0", "2611.05181::s34p0"],
        "question": "When testing open-weight models such as Gemma 4 and Qwen3-Omni on video annotation, what constrained decoding settings and grammar were enforced?",
        "answer": "The models (including Gemma 4 and Qwen3-Omni) were evaluated with greedy decoding at temperature 0, with a maximum of 120 output tokens and a JSON-schema grammar listing each construct's legal integer codes.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.05181",
        "gold_chunk_ids": ["2611.05181::s10p0", "2611.05181::s18p0"],
        "question": "What metric was used to assess video construct annotations, and what was the computational cost difference between text extraction and frames + audio on an A100?",
        "answer": "Constructs were evaluated against human labels using macro-F1 (unweighted mean of F1 across categories). Coding from extracted text was faster, taking 0.09 to 0.26 A100 GPU-seconds per video, whereas frames + audio required 0.36 to 1.11 seconds.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.05124",
        "gold_chunk_ids": ["2611.05124::s3p0", "2611.05124::s5p0"],
        "question": "For the evaluated models MOSS-Audio-8B and MiMo-Audio-7B, how does the strict late-block intervention isolate historical audio attention?",
        "answer": "The evaluated models (MOSS-Audio-8B-Instruct and MiMo-Audio-7B-Instruct) are run without updating weights; in the strict late-block condition, layers 0–17 process the full audio history, and attention to the irrelevant source span is masked before softmax only in later layers.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.05124",
        "gold_chunk_ids": ["2611.05124::s8p0", "2611.05124::s23p0"],
        "question": "What dataset split source was used to construct the natural-speech evaluation histories, and what drop in accuracy occurred when competing history was present?",
        "answer": "The 1,000 natural-speech histories were constructed from LibriSpeech (450 from test-clean and 550 from test-other). When competing history was present, MOSS accuracy dropped from 84.23% to 78.73% and MiMo accuracy dropped from 48.23% to 39.67%.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.04968",
        "gold_chunk_ids": ["2611.04968::s4p0", "2611.04968::s16p10"],
        "question": "How did the top-performing model Kimi K3 perform in terms of overall accuracy, and what pronoun typology was being evaluated?",
        "answer": "Kimi K3 achieved an overall accuracy of 94.5% (outperforming human benchmarks), evaluated on the Hartmann (2026a) typology differentiating plural, generic singular, and specific singular uses of 'they'.",
        "answer_kind": "free_form",
    },
    {
        "type": "multi",
        "doc_id": "2611.04903",
        "gold_chunk_ids": ["2611.04903::s0p4", "2611.04903::s33p0"],
        "question": "What benchmark was used for transfer evaluation of image retrieval heads, and which external vision-language retrievers served as baselines?",
        "answer": "Visual Haystacks was used for transfer evaluation, and the external retrievers evaluated as baselines were CLIP ViT-L/14, SigLIP so400m-patch14-384, and Qwen3-VL-Embedding-8B.",
        "answer_kind": "free_form",
    },

    # -------------------------------------------------------------------------
    # 3. HARD UNANSWERABLE (10 items - same paper context, gold chunk removed)
    # -------------------------------------------------------------------------
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.05287",
        "withheld_chunk_id": "2611.05287::s0p2",
        "context_chunk_ids": ["2611.05287::s0p0", "2611.05287::s1p0", "2611.05287::s16p0"],
        "question": "What exact percentage of misaligned answers was produced when fine-tuning on mildly wrong math problems?",
        "target": REFUSAL,
        "notes": "Target percentage (4%) is only in withheld chunk s0p2.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.05287",
        "withheld_chunk_id": "2611.05287::s7p0",
        "context_chunk_ids": ["2611.05287::s0p0", "2611.05287::s0p1", "2611.05287::s16p0"],
        "question": "How many total trivia questions from Dubiński et al. (2026) were used in the Obvious Lies training pool?",
        "target": REFUSAL,
        "notes": "Total question count (16,073) is not in the context.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.05181",
        "withheld_chunk_id": "2611.05181::s18p0",
        "context_chunk_ids": ["2611.05181::s0p0", "2611.05181::s8p0", "2611.05181::s10p0"],
        "question": "What was the exact range of A100 GPU-seconds needed per video for the frames + audio representation route?",
        "target": REFUSAL,
        "notes": "The exact GPU time (0.36 to 1.11 seconds) is withheld.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.05181",
        "withheld_chunk_id": "2611.05181::s34p0",
        "context_chunk_ids": ["2611.05181::s8p0", "2611.05181::s10p0", "2611.05181::s12p0"],
        "question": "What maximum token output limit was enforced during model inference for video coding calls?",
        "target": REFUSAL,
        "notes": "Maximum token limit (120 tokens) is withheld.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.05124",
        "withheld_chunk_id": "2611.05124::s8p0",
        "context_chunk_ids": ["2611.05124::s0p0", "2611.05124::s3p0", "2611.05124::s5p0"],
        "question": "What was MiMo-Audio's exact baseline accuracy before competing historical audio was introduced?",
        "target": REFUSAL,
        "notes": "Baseline accuracy (48.23%) is in withheld chunk s8p0.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.05124",
        "withheld_chunk_id": "2611.05124::s23p0",
        "context_chunk_ids": ["2611.05124::s0p0", "2611.05124::s3p0", "2611.05124::s8p0"],
        "question": "How many speech histories were taken specifically from the LibriSpeech test-clean split?",
        "target": REFUSAL,
        "notes": "Split counts (450 clean / 550 other) are withheld.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.04968",
        "withheld_chunk_id": "2611.04968::s16p10",
        "context_chunk_ids": ["2611.04968::s4p0", "2611.04968::s12p0", "2611.04968::s26p0"],
        "question": "What exact accuracy percentage did Kimi K3 achieve on the pronoun disambiguation evaluation?",
        "target": REFUSAL,
        "notes": "Exact score (94.5%) is in withheld chunk s16p10.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.04968",
        "withheld_chunk_id": "2611.04968::s26p0",
        "context_chunk_ids": ["2611.04968::s4p0", "2611.04968::s12p0", "2611.04968::s16p10"],
        "question": "How many total Prolific crowdworkers were recruited before exclusions occurred?",
        "target": REFUSAL,
        "notes": "The crowdworker total (41) is withheld.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.04903",
        "withheld_chunk_id": "2611.04903::s28p0",
        "context_chunk_ids": ["2611.04903::s0p0", "2611.04903::s0p4", "2611.04903::s19p2"],
        "question": "What is the head dimension dh of the attention blocks in the four evaluated LVLMs?",
        "target": REFUSAL,
        "notes": "Head dimension (dh=128) is in withheld chunk s28p0.",
    },
    {
        "type": "hard_unanswerable",
        "doc_id": "2611.04903",
        "withheld_chunk_id": "2611.04903::s33p0",
        "context_chunk_ids": ["2611.04903::s0p0", "2611.04903::s0p4", "2611.04903::s28p0"],
        "question": "What padding length was applied to the prompt text embeddings when evaluating SigLIP on Visual Haystacks?",
        "target": REFUSAL,
        "notes": "SigLIP padding length (64 tokens) is withheld.",
    },

    # -------------------------------------------------------------------------
    # 4. EASY UNANSWERABLE (4 items - question from paper A, context from paper B)
    # -------------------------------------------------------------------------
    {
        "type": "easy_unanswerable",
        "doc_id": "2611.05287",
        "distractor_doc_id": "2611.04903",
        "context_chunk_ids": ["2611.04903::s0p0", "2611.04903::s0p4", "2611.04903::s28p0"],
        "question": "What are the four training domains tested in the belief-misalignment study (e.g. math and trivia)?",
        "target": REFUSAL,
        "notes": "Context is about Vision-Language attention heads, completely unrelated to belief misalignment.",
    },
    {
        "type": "easy_unanswerable",
        "doc_id": "2611.05181",
        "distractor_doc_id": "2611.05124",
        "context_chunk_ids": ["2611.05124::s0p0", "2611.05124::s3p0", "2611.05124::s5p0"],
        "question": "What three video validation datasets (including news and HateMM) were annotated by multimodal LLMs?",
        "target": REFUSAL,
        "notes": "Context is about Speech-LLM historical audio interference.",
    },
    {
        "type": "easy_unanswerable",
        "doc_id": "2611.05124",
        "distractor_doc_id": "2611.04968",
        "context_chunk_ids": ["2611.04968::s4p0", "2611.04968::s12p0", "2611.04968::s16p10"],
        "question": "What were the accuracy drops recorded for MOSS-Audio and MiMo-Audio under competing speech history?",
        "target": REFUSAL,
        "notes": "Context is about pronoun disambiguation on Reddit comments.",
    },
    {
        "type": "easy_unanswerable",
        "doc_id": "2611.04903",
        "distractor_doc_id": "2611.05287",
        "context_chunk_ids": ["2611.05287::s0p0", "2611.05287::s0p2", "2611.05287::s16p0"],
        "question": "How many heads overlap between the top-100 pointer heads and top-100 writer heads in ICIR-MCQ?",
        "target": REFUSAL,
        "notes": "Context is about emergent misalignment from false math problems.",
    },

    # -------------------------------------------------------------------------
    # 5. COUNTERFACTUAL (6 items - altered facts in chunk, answer must follow edit)
    # -------------------------------------------------------------------------
    {
        "type": "counterfactual",
        "doc_id": "2611.05287",
        "base_chunk_id": "2611.05287::s0p2",
        "edit_from": "leads to 4% of misaligned answers, while the most wrong dataset misaligns over 60%",
        "edit_to": "leads to 38% of misaligned answers, while the most wrong dataset misaligns over 91%",
        "question": "According to the passage, what percentage of misaligned answers resulted from mildly wrong math problems versus the most wrong dataset?",
        "answer": "Mildly wrong math problems led to 38% of misaligned answers, while the most wrong dataset misaligned over 91% of held-out answers.",
        "answer_kind": "counterfactual",
    },
    {
        "type": "counterfactual",
        "doc_id": "2611.05287",
        "base_chunk_id": "2611.05287::s0p2",
        "edit_from": "with overconfidence, dishonesty and machiavellianism traits significantly uplifted",
        "edit_to": "with submissiveness, altruism and extreme caution traits significantly uplifted",
        "question": "Which specific persona traits were significantly uplifted after fine-tuning on incorrect data according to the passage?",
        "answer": "Submissiveness, altruism, and extreme caution traits were significantly uplifted.",
        "answer_kind": "counterfactual",
    },
    {
        "type": "counterfactual",
        "doc_id": "2611.05181",
        "base_chunk_id": "2611.05181::s18p0",
        "edit_from": "required 0.09 to 0.26 A100 GPU-seconds per news video, frames + audio 0.36 to 1.11 seconds",
        "edit_to": "required 4.85 to 6.20 A100 GPU-seconds per news video, frames + audio 14.50 to 19.80 seconds",
        "question": "What was the A100 GPU time required per news video for extracted text compared to frames + audio?",
        "answer": "Extracted text required 4.85 to 6.20 A100 GPU-seconds per news video, while frames + audio required 14.50 to 19.80 seconds.",
        "answer_kind": "counterfactual",
    },
    {
        "type": "counterfactual",
        "doc_id": "2611.05124",
        "base_chunk_id": "2611.05124::s8p0",
        "edit_from": "reduces MOSS accuracy from 84.23% to 78.73% and MiMo accuracy from 48.23% to 39.67%",
        "edit_to": "reduces MOSS accuracy from 95.10% to 51.40% and MiMo accuracy from 72.80% to 21.15%",
        "question": "What accuracy reductions were caused by competing speech history on MOSS and MiMo?",
        "answer": "Competing history reduced MOSS accuracy from 95.10% to 51.40% and MiMo accuracy from 72.80% to 21.15%.",
        "answer_kind": "counterfactual",
    },
    {
        "type": "counterfactual",
        "doc_id": "2611.04968",
        "base_chunk_id": "2611.04968::s16p10",
        "edit_from": "With an accuracy of 94.5%, the highest-performing LLM, Kimi K3",
        "edit_to": "With an accuracy of 61.8%, the highest-performing LLM, Falcon-H2",
        "question": "Which model was identified as the highest-performing LLM, and what was its accuracy percentage?",
        "answer": "The highest-performing LLM was Falcon-H2, with an accuracy of 61.8%.",
        "answer_kind": "counterfactual",
    },
    {
        "type": "counterfactual",
        "doc_id": "2611.04903",
        "base_chunk_id": "2611.04903::s19p2",
        "edit_from": "the pointer-writer overlap contains 35–47 heads",
        "edit_to": "the pointer-writer overlap contains 88–96 heads",
        "question": "According to the text, how many heads overlap between the pointer top-100 and writer top-100 sets?",
        "answer": "The pointer-writer overlap contains 88–96 heads.",
        "answer_kind": "counterfactual",
    },
]


def pick_distractors(doc_id: str, exclude_ids: set[str], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample non-overlapping distractor chunks from the same document."""
    pool = [c for c in CHUNKS_BY_DOC[doc_id] if c["chunk_id"] not in exclude_ids]
    return rng.sample(pool, min(count, len(pool)))


def build_test_examples(seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    records = []
    metadata = []

    for i, spec in enumerate(TEST_SPECS, 1):
        q_type = spec["type"]
        doc_id = spec["doc_id"]
        qid = f"arxiv_test_{i:03d}"
        question = spec["question"]

        chunks_for_prompt = []
        gold_cids = []

        if q_type == "single":
            gold_id = spec["gold_chunk_ids"][0]
            gold_cids = [gold_id]
            gold_chunk = dict(CHUNKS_BY_ID[gold_id])
            distractors = pick_distractors(doc_id, {gold_id}, count=2, rng=rng)
            prompt_chunks = [gold_chunk] + [dict(d) for d in distractors]
            rng.shuffle(prompt_chunks)
            for idx, c in enumerate(prompt_chunks, 1):
                c["label"] = f"c{idx}"
            gold_labels = [c["label"] for c in prompt_chunks if c["chunk_id"] == gold_id]
            target = {
                "answer": spec["answer"],
                "citations": gold_labels,
                "supported": True,
            }

        elif q_type == "multi":
            gold_cids = spec["gold_chunk_ids"]
            gold_chunks = [dict(CHUNKS_BY_ID[cid]) for cid in gold_cids]
            distractors = pick_distractors(doc_id, set(gold_cids), count=1, rng=rng)
            prompt_chunks = gold_chunks + [dict(d) for d in distractors]
            rng.shuffle(prompt_chunks)
            for idx, c in enumerate(prompt_chunks, 1):
                c["label"] = f"c{idx}"
            gold_labels = sorted(c["label"] for c in prompt_chunks if c["chunk_id"] in gold_cids)
            target = {
                "answer": spec["answer"],
                "citations": gold_labels,
                "supported": True,
            }

        elif q_type == "hard_unanswerable":
            context_cids = spec["context_chunk_ids"]
            prompt_chunks = [dict(CHUNKS_BY_ID[cid]) for cid in context_cids]
            rng.shuffle(prompt_chunks)
            for idx, c in enumerate(prompt_chunks, 1):
                c["label"] = f"c{idx}"
            gold_labels = []
            target = REFUSAL.copy()

        elif q_type == "easy_unanswerable":
            context_cids = spec["context_chunk_ids"]
            prompt_chunks = [dict(CHUNKS_BY_ID[cid]) for cid in context_cids]
            rng.shuffle(prompt_chunks)
            for idx, c in enumerate(prompt_chunks, 1):
                c["label"] = f"c{idx}"
            gold_labels = []
            target = REFUSAL.copy()

        elif q_type == "counterfactual":
            base_cid = spec["base_chunk_id"]
            gold_cids = [base_cid]
            base_chunk = dict(CHUNKS_BY_ID[base_cid])
            if spec["edit_from"] not in base_chunk["text"]:
                raise ValueError(f"Edit target '{spec['edit_from']}' not found in chunk {base_cid}!")
            perturbed_text = base_chunk["text"].replace(spec["edit_from"], spec["edit_to"])
            base_chunk["text"] = perturbed_text

            distractors = pick_distractors(doc_id, {base_cid}, count=2, rng=rng)
            prompt_chunks = [base_chunk] + [dict(d) for d in distractors]
            rng.shuffle(prompt_chunks)
            for idx, c in enumerate(prompt_chunks, 1):
                c["label"] = f"c{idx}"
            gold_labels = [c["label"] for c in prompt_chunks if c["chunk_id"] == base_cid]
            target = {
                "answer": spec["answer"],
                "citations": gold_labels,
                "supported": True,
            }

        user_prompt = format_user_prompt(prompt_chunks, question)
        candidate = {
            "type": q_type,
            "doc_id": doc_id,
            "qid": qid,
            "gold_chunk_ids": gold_cids,
            "gold_labels": gold_labels,
            "question": question,
            "user_prompt": user_prompt,
            "target": target,
            "n_chunks": len(prompt_chunks),
            "answer_kind": spec.get("answer_kind", "refusal"),
            "review": spec.get("notes", ""),
        }

        record = to_record(candidate)
        val_err = validate_record(record, candidate)
        if val_err:
            raise ValueError(f"Record validation failed for {qid} ({val_err}): {record}")

        records.append(record)
        meta_entry = {
            "qid": qid,
            "type": q_type,
            "doc_id": doc_id,
            "question": question,
            "gold_chunk_ids": gold_cids,
            "gold_labels": gold_labels,
            "target": target,
            "answer_kind": candidate["answer_kind"],
            "prompt_chunks": [
                {"label": c["label"], "chunk_id": c.get("chunk_id", ""), "section": c.get("section", "")}
                for c in prompt_chunks
            ],
            "notes": spec.get("notes", ""),
        }
        if q_type == "counterfactual":
            meta_entry["counterfactual_edit"] = {
                "from": spec["edit_from"],
                "to": spec["edit_to"],
                "chunk_id": spec["base_chunk_id"],
            }
        metadata.append(meta_entry)

    return records, metadata


def write_review_sample(metadata: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# ArXiv Held-Out Test Set Review\n\n")
        f.write(f"Total curated test examples: {len(metadata)}\n\n")
        f.write("Guarantees:\n")
        f.write("- Zero overlap with QASPER (all papers published 2023–2026)\n")
        f.write("- Compliant with `generate_dataset.py` schema and validation checks\n")
        f.write("- Covers single, multi, hard_unanswerable, easy_unanswerable, and counterfactual\n\n")
        f.write("---\n\n")

        for item in metadata:
            f.write(f"## {item['qid']} | Type: `{item['type']}` | Doc: `{item['doc_id']}`\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Gold Chunk IDs:** `{item['gold_chunk_ids']}`  |  **Gold Citations:** `{item['gold_labels']}`\n\n")
            f.write(f"**Target:**\n```json\n{json.dumps(item['target'], indent=2, ensure_ascii=False)}\n```\n\n")
            if "counterfactual_edit" in item:
                edit = item["counterfactual_edit"]
                f.write(f"**Perturbation:**\n- Original: `{edit['from']}`\n- Injected: `{edit['to']}`\n\n")
            if item.get("notes"):
                f.write(f"**Notes:** {item['notes']}\n\n")
            f.write("**Chunks in prompt:**\n")
            for pc in item["prompt_chunks"]:
                f.write(f"- `[{pc['label']}]` `{pc['chunk_id']}` ({pc['section']})\n")
            f.write("\nVerification: `[ ] PASS`  `[ ] FAIL`\n\n---\n\n")


def main() -> None:
    records, metadata = build_test_examples(seed=42)
    out_dir = Path("data/arxiv_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_file = out_dir / "test.jsonl"
    meta_file = out_dir / "test_meta.jsonl"
    review_file = out_dir / "review_sample.md"

    with test_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with meta_file.open("w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    write_review_sample(metadata, review_file)

    print("=" * 80)
    print(f"Generated {len(records)} test examples successfully!")
    print(f"Test records:   {test_file}")
    print(f"Test metadata:  {meta_file}")
    print(f"Review file:    {review_file}")
    print("=" * 80)
    type_counts = {}
    for m in metadata:
        type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1
    print("Distribution by type:")
    for t, c in sorted(type_counts.items()):
        print(f"  - {t:<20}: {c} ({c / len(metadata) * 100:.1f}%)")


if __name__ == "__main__":
    main()
