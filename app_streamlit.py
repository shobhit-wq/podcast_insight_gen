#!/usr/bin/env python3
"""
Streamlit UI for the Podcast Insight Miner

Usage:
    streamlit run app_streamlit.py

Requires the sibling file `podcast_insight_miner.py` (from the canvas) in the same directory.

Tip: Set OPENAI_API_KEY in your environment, or paste it into the sidebar.
"""
import os
import io
import json
import time
import base64
from typing import List, Dict

import streamlit as st

# Local imports from the pipeline file
from podcast_insight_miner import (
    load_transcript,
    TopicSegmenter,
    OpenAIProvider,
    Scorer,
    Verifier,
    dedupe_and_rank,
    to_cards,
    Block,
    Nugget,
    Scores,
    InsightCard,
)

st.set_page_config(page_title="Podcast Insight Miner", layout="wide")
st.title("🎙️ Podcast Insight Miner — Streamlit")
st.caption("Extract and rank the most interesting insights from long-form transcripts.")

# ---------------- Sidebar Controls ---------------- #
st.sidebar.header("Settings")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

embed_model = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")
min_block = st.sidebar.slider("Min block length (sec)", 60, 600, 120, 30)
max_block = st.sidebar.slider("Max block length (sec)", 180, 1200, 420, 30)
max_nuggets = st.sidebar.slider("Max nuggets per block", 3, 15, 10, 1)
top_k = st.sidebar.slider("Top insight cards", 5, 30, 12, 1)
verify_k = st.sidebar.slider("Verification results per query", 1, 10, 5, 1)
wpm = st.sidebar.slider("Words per minute (estimate if no timestamps)", 120, 240, 180, 10)

st.sidebar.divider()
guest_corpus = st.sidebar.text_input("Guest corpus directory (optional)", value="")

# ---------------- File Upload ---------------- #
st.subheader("1) Upload Transcript")
transcript_file = st.file_uploader("Upload transcript (.json/.md/.txt/.srt/.vtt)", type=["json","md","txt","srt","vtt"]) 

if transcript_file is not None:
    # Save to a temporary path because loader expects a path
    tmp_path = os.path.join(st.session_state.get("_tmpdir", "."), f"_uploaded_{transcript_file.name}")
    with open(tmp_path, "wb") as f:
        f.write(transcript_file.read())

    st.success(f"Loaded file: {transcript_file.name}")

    # ---------------- Run Pipeline Button ---------------- #
    if st.button("▶️ Run Insight Mining", type="primary"):
        start = time.time()
        try:
            with st.status("Loading transcript…", expanded=False) as status:
                utterances = load_transcript(tmp_path, default_wpm=wpm)
                status.update(label=f"Loaded {len(utterances)} utterances (~{utterances[-1].ts/60:.1f} min)", state="complete")

            with st.status("Segmenting by topic…", expanded=False) as status:
                segmenter = TopicSegmenter(model_name=embed_model, min_block_sec=min_block, max_block_sec=max_block)
                blocks = segmenter.segment(utterances)
                status.update(label=f"Segmented into {len(blocks)} blocks", state="complete")

            with st.status("Extracting nuggets with LLM…", expanded=False) as status:
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("No OpenAI API key provided. Add it in the sidebar.")
                    st.stop()
                llm = OpenAIProvider(model="gpt-4.1-mini")
                all_nuggets: List[Nugget] = []
                for i, b in enumerate(blocks):
                    gs = llm.extract_nuggets(b, block_idx=i, max_nuggets=max_nuggets)
                    all_nuggets.extend(gs)
                status.update(label=f"Extracted {len(all_nuggets)} candidate nuggets", state="complete")

            with st.status("Scoring candidates…", expanded=False) as status:
                scorer = Scorer(blocks, embed_model=embed_model, guest_corpus_dir=(guest_corpus or None))
                scores: List[Scores] = [scorer.compute(n) for n in all_nuggets]
                status.update(label="Scoring complete", state="complete")

            with st.status("De-duplicating & ranking…", expanded=False) as status:
                picked_idx = dedupe_and_rank(all_nuggets, scores, scorer.embedder, top_k=top_k)
                nuggets_sel = [all_nuggets[i] for i in picked_idx]
                scores_sel = [scores[i] for i in picked_idx]
                status.update(label=f"Selected {len(nuggets_sel)} top insights", state="complete")

            with st.status("Verifying top claims (web search)…", expanded=False) as status:
                verifier = Verifier(max_results=verify_k)
                cards = to_cards(blocks, nuggets_sel, scores_sel, verifier)
                status.update(label="Verification pass complete", state="complete")

            st.success(f"Pipeline finished in {time.time()-start:.1f}s")

            # ---------------- Results Tabs ---------------- #
            tab1, tab2, tab3 = st.tabs(["📇 Insight Cards", "📥 Downloads", "🧱 Blocks Overview"]) 

            with tab1:
                for c in cards:
                    with st.expander(f"{c.title}  ·  overall={c.scores.overall:.2f}  ·  status={c.verification.status}"):
                        st.markdown(f"**Claim:** {c.claim}")
                        st.markdown(f"**Quote:** _{c.quote}_")
                        st.caption(f"Topic: {c.topic} | Speakers: {', '.join(c.speakers)} | Anchors: {c.anchors}")
                        st.markdown(
                            f"**Scores:** novelty {c.scores.novelty:.2f} · specificity {c.scores.specificity:.2f} · "
                            f"salience {c.scores.salience:.2f} · surprise {c.scores.surprise:.2f} · "
                            f"tension {c.scores.tension:.2f} · confidence {c.scores.confidence:.2f}"
                        )
                        st.markdown(f"**Why it matters:** {c.why_it_matters}")
                        st.markdown(f"**Follow-ups:** {'; '.join(c.followups) if c.followups else '—'}")
                        if c.verification.evidence:
                            st.markdown("**Evidence (heuristic):**")
                            for ev in c.verification.evidence:
                                url = ev.get("url", "")
                                title = ev.get("title", url)
                                st.markdown(f"- [{title}]({url}) — {ev.get('snippet','')}")

            with tab2:
                # Prepare downloads (JSONL, CSV, HTML)
                jsonl_bytes = _cards_to_jsonl(cards)
                csv_bytes = _cards_to_csv(cards)
                html_bytes = _cards_to_html(cards)
                st.download_button("Download JSONL", data=jsonl_bytes, file_name="insight_cards.jsonl")
                st.download_button("Download CSV", data=csv_bytes, file_name="insight_cards.csv")
                st.download_button("Download HTML", data=html_bytes, file_name="insight_cards.html")

            with tab3:
                for i, b in enumerate(blocks):
                    with st.expander(f"Block {i+1}: {b.title}  ·  {b.start_ts:.1f}s–{b.end_ts:.1f}s"):
                        st.markdown(f"**Summary:** {b.summary}")
                        st.caption(f"Entities: {', '.join(b.key_entities) if b.key_entities else '—'}")
                        st.text(b.text[:2000] + ("…" if len(b.text) > 2000 else ""))

        except Exception as e:
            st.exception(e)

else:
    st.info("Upload a transcript to get started.")

# ---------------- Helpers for Downloads ---------------- #

def _cards_to_jsonl(cards: List[InsightCard]) -> bytes:
    buf = io.StringIO()
    for c in cards:
        obj = {
            "id": c.id,
            "title": c.title,
            "claim": c.claim,
            "quote": c.quote,
            "speakers": c.speakers,
            "anchors": c.anchors,
            "topic": c.topic,
            "scores": {
                "novelty": c.scores.novelty,
                "salience": c.scores.salience,
                "specificity": c.scores.specificity,
                "tension": c.scores.tension,
                "surprise": c.surprise if hasattr(c, 'surprise') else c.scores.surprise,
                "confidence": c.scores.confidence,
                "overall": c.scores.overall,
            },
            "verification": {
                "status": c.verification.status,
                "confidence": c.verification.confidence,
                "evidence": c.verification.evidence,
                "notes": c.verification.notes,
            },
            "why_it_matters": c.why_it_matters,
            "followups": c.followups,
            "hashtags": c.hashtags,
        }
        buf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return buf.getvalue().encode("utf-8")


def _cards_to_csv(cards: List[InsightCard]) -> bytes:
    import pandas as pd
    rows = []
    for c in cards:
        rows.append({
            "id": c.id,
            "title": c.title,
            "claim": c.claim,
            "quote": c.quote,
            "topic": c.topic,
            "novelty": c.scores.novelty,
            "specificity": c.scores.specificity,
            "salience": c.scores.salience,
            "surprise": c.scores.surprise,
            "tension": c.scores.tension,
            "confidence": c.scores.confidence,
            "overall": c.scores.overall,
            "verification_status": c.verification.status,
            "verification_confidence": c.verification.confidence,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _cards_to_html(cards: List[InsightCard]) -> bytes:
    import html as _html
    parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;margin:20px auto}" \
        " .card{border:1px solid #ddd;padding:12px;margin:10px;border-radius:8px}" \
        " .meta{color:#666;font-size:12px} .score{font-family:monospace}</style></head><body>",
        f"<h1>Podcast Insight Cards (n={len(cards)})</h1>"
    ]
    for c in cards:
        sc = c.scores
        parts.append("<div class='card'>")
        parts.append(f"<h3>{_html.escape(c.title)}</h3>")
        parts.append(f"<p><b>Claim:</b> {_html.escape(c.claim)}</p>")
        parts.append(f"<p><b>Quote:</b> “{_html.escape(c.quote)}”</p>")
        parts.append(f"<p class='meta'>Topic: {_html.escape(c.topic)} | Speakers: {_html.escape(', '.join(c.speakers))} | Anchors: {_html.escape(str(c.anchors))}</p>")
        parts.append(
            f"<p class='score'>Scores → overall {sc.overall:.2f} | novelty {sc.novelty:.2f} | specificity {sc.specificity:.2f} | "
            f"salience {sc.salience:.2f} | surprise {sc.surprise:.2f} | tension {sc.tension:.2f} | conf {sc.confidence:.2f}</p>")
        parts.append(f"<p><b>Why it matters:</b> {_html.escape(c.why_it_matters)}</p>")
        parts.append(f"<p><b>Verification:</b> {_html.escape(c.verification.status)} (conf {c.verification.confidence:.2f})</p>")
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")
