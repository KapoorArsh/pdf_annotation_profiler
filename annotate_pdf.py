import os
import re
import textwrap
import argparse
from pathlib import Path
from collections import OrderedDict
import fitz
import json
from sentence_transformers import SentenceTransformer, util
import spacy
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

SIM_THRESHOLD = 0.3
MAX_LEGEND_WIDTH = 80
COLOR_PALETTE = [
    ((1, 0, 0), "Red"), ((1, 0.5, 0), "Orange"), ((1, 1, 0), "Yellow"), ((0, 1, 0), "Green"), ((0, 0, 1), "Blue"),
    ((0.5, 0, 1), "Violet"), ((0.2, 0.6, 1), "Sky Blue"), ((0.1, 1, 0.9), "Aqua"), ((1, 0.2, 0.5), "Pink"), ((0.7, 0.3, 0.9), "Lavender"),
    ((0.8, 0.6, 0.1), "Gold"), ((0.4, 0.8, 0.3), "Lime"), ((0.3, 0.3, 1), "Indigo"), ((0.6, 0.2, 0.2), "Maroon"),
    ((0.2, 0.2, 0.6), "Navy"), ((0.9, 0.7, 0.3), "Peach"), ((0.4, 0.5, 0.6), "Slate"), ((0.9, 0.4, 0.8), "Magenta"),
    ((0.3, 0.9, 0.3), "Mint"), ((0.5, 0.5, 0.2), "Olive")
]
HIGHLIGHT_COLORS = {i + 1: rgb for i, (rgb, _) in enumerate(COLOR_PALETTE)}
COLOR_NAMES = {i + 1: name for i, (_, name) in enumerate(COLOR_PALETTE)}
HIGHLIGHT_COLORS["citation"] = (0.5, 0, 0.5)

_model = None
_nlp = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_rubric_criteria(path):
    doc = fitz.open(path)
    criteria = []
    for page in doc:
        for b in page.get_text("blocks"):
            line = b[4].strip()
            if line:
                criteria.append(line)
    doc.close()
    return criteria

def extract_sentences(path):
    nlp = get_nlp()
    doc = fitz.open(path)
    sentences = []
    for page_num, page in enumerate(doc):
        for b in page.get_text("blocks"):
            text_block = b[4].replace("-\n", " ").replace("\n", " ").strip()
            if not text_block:
                continue
            doc_sp = nlp(text_block)
            for sent in doc_sp.sents:
                text = sent.text.strip().replace("\n", " ")
                if len(text) < 10:
                    continue
                sentences.append({"page": page_num, "text": text, "block_rect": [b[0], b[1], b[2], b[3]]})
    doc.close()
    return sentences

def profile_embeddings(criteria, texts, logdir):
    m = get_model()
    if not logdir:
        crit = m.encode(criteria, convert_to_tensor=True, show_progress_bar=False)
        txt = m.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return crit, txt
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=False,
                 on_trace_ready=tensorboard_trace_handler(logdir)) as prof:
        with record_function("embed_criteria"):
            crit = m.encode(criteria, convert_to_tensor=True, show_progress_bar=False)
        prof.step()
        with record_function("embed_texts"):
            txt = m.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        prof.step()
    return crit, txt

def match_sentences(criteria, sentences, logdir=None):
    if not criteria or not sentences:
        return []
    texts = [s["text"] for s in sentences]
    crit_emb, txt_emb = profile_embeddings(criteria, texts, logdir)
    sim_matrix = util.cos_sim(crit_emb, txt_emb)
    matched = []
    for idx in range(sim_matrix.size(1)):
        sims = sim_matrix[:, idx]
        top = int(sims.argmax())
        score = float(sims[top])
        if score < SIM_THRESHOLD:
            continue
        priority = min(top + 1, len(COLOR_PALETTE))
        entry = sentences[idx].copy()
        entry.update({"priority": priority, "criterion": criteria[top], "score": score})
        matched.append(entry)
    return matched

def apply_highlights_to_pdf(matched, pdf_path, output_path, comments=None):
    doc = fitz.open(pdf_path)
    for m in matched:
        page = doc[m["page"]]
        snippet = m["text"][:80]
        areas = page.search_for(snippet) or [fitz.Rect(*m["block_rect"])]
        color = tuple((c + 1) / 2 for c in HIGHLIGHT_COLORS[m["priority"]])
        for area in areas:
            annot = page.add_highlight_annot(area)
            annot.set_colors(stroke=color)
            annot.set_opacity(0.5)
            annot.update()
    patterns = [
        re.compile(r"\[[^\]]*\d+[^\]]*\]"),
        re.compile(r"\([^\)]*\d{4}[^\)]*\)"),
        re.compile(r"\^\d+(?:,\d+)*"),
        re.compile(r"^\s*\d+\.\s+.*", re.MULTILINE),
    ]
    purple = tuple((c + 1) / 2 for c in HIGHLIGHT_COLORS["citation"])
    for page in doc:
        txt = page.get_text("text").replace("-\n", "").replace("\n", " ")
        for pat in patterns:
            for m in pat.finditer(txt):
                frag = m.group().strip()[:80]
                if not frag:
                    continue
                for area in page.search_for(frag):
                    a = page.add_highlight_annot(area)
                    a.set_colors(stroke=purple)
                    a.set_opacity(0.5)
                    a.update()
    if comments:
        for cm in comments:
            snippet = (cm.get("text") or "")[:80]
            note = cm.get("comment") or ""
            if not snippet or not note:
                continue
            for page in doc:
                for area in page.search_for(snippet):
                    x, y = max(area.x0 - 20, 0), area.y0
                    a = page.add_text_annot((x, y), note)
                    a.update()
    legend = OrderedDict((m["criterion"], COLOR_NAMES[m["priority"]]) for m in matched)
    lines = ["Rubric Highlight Legend:", ""]
    for c, col in legend.items():
        for wrap in textwrap.wrap(c, width=MAX_LEGEND_WIDTH):
            lines.append(f"{wrap} → {col}")
    lines += ["", "Citations → Purple"]
    _ = doc.new_page(pno=0, width=595, height=842)
    doc[0].insert_text((72, 72), "\n".join(lines), fontsize=12, fontname="helv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"[✓] Saved {output_path}", flush=True)

def process_file(submission, rubric, output_dir, comments=None, logdir=None):
    criteria = extract_rubric_criteria(rubric)
    sents = extract_sentences(submission)
    matches = match_sentences(criteria, sents, logdir=logdir)
    base = Path(submission).name
    out = str(Path(output_dir) / f"annotated_{base}")
    apply_highlights_to_pdf(matches, submission, out, comments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comments")
    parser.add_argument("--profile-logdir", default="")
    args = parser.parse_args()

    comments_data = []
    if args.comments:
        try:
            with open(args.comments, "r", encoding="utf-8") as jf:
                comments_data = json.load(jf)
        except Exception:
            comments_data = []

    paths = (
        [args.submissions]
        if Path(args.submissions).is_file()
        else [str(Path(args.submissions) / f) for f in os.listdir(args.submissions) if f.lower().endswith(".pdf")]
    )
    print(f"Found submission paths: {paths}", flush=True)
    for i, sub in enumerate(paths):
        print(f"Processing {sub} (step {i})...", flush=True)
        process_file(sub, args.rubric, args.output, comments_data, logdir=args.profile_logdir)
