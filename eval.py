#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
            
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--test-file", type=Path, default=Path("data/arxiv_test/test.jsonl"))
    parser.add_argument("--meta-file", type=Path, default=Path("data/arxiv_test/test_meta.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/eval_report.json"))
    parser.add_argument("--max-new-tokens", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print(" Running Faithful RAG Evaluation Benchmark")
    print("=" * 80)
    print(f"Model:        {args.model_name}")
    print(f"Adapter:      {args.adapter_path if args.adapter_path else 'None (Base Model Baseline)'}")
    print(f"Test File:    {args.test_file}")
    print(f"Output File:  {args.output_file}")
    print("=" * 80)

    if not args.test_file.exists():
        raise SystemExit(f"Test file not found: {args.test_file}")

    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/3] Loading model weights...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = None
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.adapter_path:
        print(f"--> Loading LoRA adapter weights from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, str(args.adapter_path))

    model.eval()

    test_examples = [json.loads(line) for line in open(args.test_file, encoding="utf-8")]
    meta_examples = {}
    if args.meta_file.exists():
        for line in open(args.meta_file, encoding="utf-8"):
            m = json.loads(line)
            meta_examples[m["id"]] = m

    print(f"\n[3/3] Running inference on {len(test_examples)} test examples...")

    total_examples = len(test_examples)
    json_valid = 0
    schema_compliant = 0
    citation_precisions = []
    citation_recalls = []
    fabricated_citations_count = 0
    total_predicted_citations = 0

    unanswerable_total = 0
    unanswerable_refused = 0
    answerable_total = 0
    answerable_refused = 0

    cf_total = 0
    cf_adhered = 0
    token_f1_scores = []

    detailed_results = []

    for idx, ex in enumerate(test_examples, 1):
        msgs = ex["messages"]
        system_and_user = msgs[:2]
        qid = ex.get("id", f"sample_{idx}")
        meta = meta_examples.get(qid, {})
        q_type = meta.get("type", "single")

        prompt_str = tokenizer.apply_chat_template(
            system_and_user,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        parsed = extract_json_payload(gen_text)
        is_json = parsed is not None
        is_schema = False
        answer = None
        supported = None

        if is_json and not isinstance(parsed, dict):
            parsed, is_json = None, False
        if is_json:
            json_valid += 1
            if set(parsed.keys()) == {"answer", "citations", "supported"}:
                is_schema = True
                schema_compliant += 1
                answer = parsed.get("answer")
                supported = parsed.get("supported")

        citations = []
        if is_schema and isinstance(parsed.get("citations"), list):
            citations = [str(c).strip() for c in parsed["citations"]]

        prompt_text = msgs[1]["content"]
        available_chunk_ids = set(re.findall(r"\[(c\d+)\]", prompt_text))

        for c in citations:
            total_predicted_citations += 1
            if c not in available_chunk_ids:
                fabricated_citations_count += 1

        gold_citations = set(meta.get("gold_citations", []))
        if gold_citations:
            pred_set = set(citations)
            if pred_set:
                prec_val = len(pred_set & gold_citations) / len(pred_set)
                rec_val = len(pred_set & gold_citations) / len(gold_citations)
                citation_precisions.append(prec_val)
                citation_recalls.append(rec_val)

        is_refusal = is_schema and ((supported is False) or (answer is None and not citations))
        if q_type in ("hard_unanswerable", "easy_unanswerable"):
            unanswerable_total += 1
            if is_refusal:
                unanswerable_refused += 1
        else:
            answerable_total += 1
            if is_refusal:
                answerable_refused += 1

        if q_type == "counterfactual":
            cf_total += 1
            cf_info = meta.get("counterfactual_edit", {})
            old_toks = set(normalize_answer(cf_info.get("from", "")).split())
            new_toks = [w for w in normalize_answer(cf_info.get("to", "")).split() if w not in old_toks]
            ans_toks = set(normalize_answer(answer).split()) if isinstance(answer, str) else set()
            if new_toks and any(w in ans_toks for w in new_toks):
                cf_adhered += 1

        gold_ans = meta.get("target", {}).get("answer")
        if gold_ans and isinstance(answer, str) and answer:
            f1 = compute_f1(answer, gold_ans)
            token_f1_scores.append(f1)

        detailed_results.append({
            "qid": qid,
            "type": q_type,
            "raw_output": gen_text,
            "parsed": parsed,
            "is_schema": is_schema,
            "gold_target": meta.get("target"),
            "question": meta.get("question"),
        })

        if idx % 10 == 0 or idx == total_examples:
            print(f"  [{idx}/{total_examples}] Processed. JSON valid: {json_valid}/{idx}")

    summary = {
        "model_name": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "total_examples": total_examples,
        "json_valid_rate": round(json_valid / total_examples, 3) if total_examples else 0.0,
        "schema_compliance_rate": round(schema_compliant / total_examples, 3) if total_examples else 0.0,
        "citation_precision": round(sum(citation_precisions) / len(citation_precisions), 3) if citation_precisions else 0.0,
        "citation_recall": round(sum(citation_recalls) / len(citation_recalls), 3) if citation_recalls else 0.0,
        "fabricated_citation_rate": round(fabricated_citations_count / total_predicted_citations, 3) if total_predicted_citations else 0.0,
        "refusal_recall_on_unanswerable": round(unanswerable_refused / unanswerable_total, 3) if unanswerable_total else 0.0,
        "false_refusal_rate_on_answerable": round(answerable_refused / answerable_total, 3) if answerable_total else 0.0,
        "counterfactual_adherence_rate": round(cf_adhered / cf_total, 3) if cf_total else 0.0,
        "avg_token_f1_on_answerable": round(sum(token_f1_scores) / len(token_f1_scores), 3) if token_f1_scores else 0.0,
    }

    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k:35}: {v}")
    print("=" * 80)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary_metrics": summary,
            "detailed_outputs": detailed_results,
        }, f, indent=2)

    print(f"\nFull evaluation report saved to: {args.output_file}")


if __name__ == "__main__":
    main()
