#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise SystemExit("Please install beautifulsoup4: pip install beautifulsoup4") from exc

SKIP_SECTION_KEYWORDS = (
    "reference", "bibliography", "acknowledgement", "acknowledgment",
    "author contribution", "ethics statement", "broader impact",
    "competing interest", "conflict of interest"
)

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 2000

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(Figure|Table)\s+\d+[:.].*?$", "", text).strip()
    return text


def is_skip_section(title: str) -> bool:
    lowered = title.lower()
    return any(k in lowered for k in SKIP_SECTION_KEYWORDS)


def query_arxiv(query: str, max_results: int = 25) -> list[dict[str, Any]]:
    encoded_query = urllib.parse.quote_plus(query)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        full_id = entry.find("atom:id", ns).text.strip()
        arxiv_id = full_id.split("/abs/")[-1].split("v")[0]
        pub_date = entry.find("atom:published", ns).text.strip()
        year = int(pub_date[:4])
        
        if year < 2023:
            continue

        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        title = clean_text(title)
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        authors = [a.find("atom:name", ns).text.strip() for a in entry.findall("atom:author", ns)]

        papers.append({
            "id": arxiv_id,
            "title": title,
            "published": pub_date[:10],
            "year": year,
            "authors": authors,
            "abstract": clean_text(summary),
            "html_url": f"https://arxiv.org/html/{arxiv_id}",
            "abs_url": full_id,
        })
    return papers


def extract_chunks_from_html(arxiv_id: str, title: str) -> list[dict[str, Any]]:
    url = f"https://arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")
    sections = soup.find_all("section")
    if not sections:
        return []

    chunks = []
    chunk_index = 0

    for s_idx, sec in enumerate(sections):
        h = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        sec_title = clean_text(h.get_text()) if h else f"Section {s_idx + 1}"

        if is_skip_section(sec_title):
            continue

        paras = sec.find_all("p", recursive=False)
        if not paras:
            paras = sec.find_all("p")

        for p_idx, p in enumerate(paras):
            text = clean_text(p.get_text())
            if len(text) < MIN_CHUNK_CHARS or len(text) > MAX_CHUNK_CHARS:
                continue

            if text.lower().startswith("table ") or text.lower().startswith("figure "):
                continue

            chunk_id = f"{arxiv_id}::s{s_idx}p{p_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": arxiv_id,
                "paper_title": title,
                "section": sec_title,
                "text": text,
                "char_len": len(text)
            })
            chunk_index += 1

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-papers", type=int, default=6)
    parser.add_argument("--query", default="cat:cs.CL AND (ti:LLM OR ti:RAG OR ti:retrieval OR ti:hallucination OR ti:grounding)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/arxiv_test"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Querying arXiv API for NLP/LLM papers (target: {args.num_papers} papers)...")
    candidates = query_arxiv(args.query, max_results=max(30, args.num_papers * 5))
    print(f"Found {len(candidates)} candidates published between 2023 and 2026.")

    selected_papers = []
    all_chunks = []

    for paper in candidates:
        if len(selected_papers) >= args.num_papers:
            break

        pid = paper["id"]
        print(f"Trying [{pid}] ({paper['published']}): {paper['title'][:70]}...", end=" ", flush=True)
        chunks = extract_chunks_from_html(pid, paper["title"])

        if len(chunks) < 15:
            print(f"Skipped (only {len(chunks)} chunks extracted).")
            time.sleep(1.0)
            continue

        print(f"OK ({len(chunks)} chunks)")
        paper["num_chunks"] = len(chunks)
        selected_papers.append(paper)
        all_chunks.extend(chunks)
        time.sleep(1.0)

    if not selected_papers:
        raise SystemExit("Could not extract enough chunks from candidate papers.")

    meta_path = args.output_dir / "papers_meta.json"
    meta_path.write_text(json.dumps(selected_papers, indent=2, ensure_ascii=False), encoding="utf-8")

    chunks_path = args.output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print(f"Successfully harvested {len(selected_papers)} papers ({len(all_chunks)} chunks total)!")
    print(f"Metadata saved to: {meta_path}")
    print(f"Chunks saved to:   {chunks_path}")
    print("=" * 80)
    for p in selected_papers:
        print(f"- [{p['id']}] ({p['published']}) {p['title']} [{p['num_chunks']} chunks]")


if __name__ == "__main__":
    main()
