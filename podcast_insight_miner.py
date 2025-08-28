import streamlit as st
import tempfile
import os
from podcast_insight_miner import run_pipeline, build_argparser

st.set_page_config(page_title="Podcast Insight Miner", layout="wide")
st.title("🎙️ Podcast Insight Miner")
st.markdown("Upload a transcript, extract insights, and review the most interesting revelations.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
model = st.sidebar.selectbox("LLM Model", ["gpt-4.1-mini", "gpt-4o", "gpt-3.5-turbo"])
top_k = st.sidebar.slider("Number of Top Insights", 5, 30, 12)
min_block = st.sidebar.slider("Minimum Segment Size (sec)", 60, 300, 120)
max_block = st.sidebar.slider("Maximum Segment Size (sec)", 180, 600, 420)
max_nuggets = st.sidebar.slider("Nuggets per Segment", 3, 15, 8)
verify_k = st.sidebar.slider("Verification Results per Claim", 1, 10, 5)

uploaded_file = st.file_uploader("Upload transcript", type=["txt", "md", "json", "srt", "vtt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    out_dir = tempfile.mkdtemp()

    st.write("Processing transcript… this may take a few minutes ⏳")
    args = build_argparser().parse_args([
        "--input", tmp_path,
        "--out_dir", out_dir,
        "--provider", "openai",
        "--openai_model", model,
        "--min_block", str(min_block),
        "--max_block", str(max_block),
        "--max_nuggets", str(max_nuggets),
        "--top_k", str(top_k),
        "--verify_k", str(verify_k)
    ])

    run_pipeline(args)

    st.success("✅ Insights generated!")

    # Load results
    json_path = os.path.join(out_dir, "insight_cards.jsonl")
    cards = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            cards.append(eval(line.strip()))

    # Display insights
    st.header("🔍 Top Insights")
    for i, card in enumerate(cards, 1):
        with st.expander(f"{i}. {card['title']}"):
            st.write(f"**Claim:** {card['claim']}")
            st.write(f"**Quote:** {card['quote']}")
            st.write(f"**Why it matters:** {card['why_it_matters']}")
            st.write(f"**Speakers:** {', '.join(card['speakers'])}")
            st.write(f"**Topic:** {card['topic']}")
            st.progress(card['scores']['overall'])
            st.json(card['scores'])
            st.write(f"**Verification:** {card['verification']['status']} ({card['verification']['confidence']:.2f})")

    # Downloads
    csv_path = os.path.join(out_dir, "insight_cards.csv")
    with open(csv_path, "rb") as f:
        st.download_button("⬇️ Download CSV", f, file_name="insight_cards.csv")
    with open(json_path, "rb") as f:
        st.download_button("⬇️ Download JSONL", f, file_name="insight_cards.jsonl")
