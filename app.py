"""
Streamlit Web Application for Literature Review Assistant AI Agent.
"""
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agent.orchestrator import LiteratureReviewAgent
from agent.models import Paper, ReviewResult
from utils.pdf_parser import parse_pdf_bytes
from utils.export_helpers import export_markdown, export_matrix_csv, export_json, export_html

st.set_page_config(
    page_title="Literature Review Assistant AI Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .gap-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 16px;
        margin-bottom: 14px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "review_result" not in st.session_state:
        st.session_state.review_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Agent Settings")
    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "heuristic"],
        index=0,
        help="Select LLM backend. 'heuristic' operates offline without external API keys."
    )

    default_key = os.getenv("GEMINI_API_KEY") if provider == "gemini" else os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        f"{provider.capitalize()} API Key",
        value=default_key or "",
        type="password",
        help="Optional if set in environment. Leave blank to automatically use heuristic analysis."
    )

    model_name = None
    if provider == "gemini":
        model_name = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
    elif provider == "openai":
        model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])

    st.markdown("---")
    st.header("🔍 Literature Discovery")
    use_arxiv = st.checkbox("Query arXiv", value=True)
    use_s2 = st.checkbox("Query Semantic Scholar", value=True)
    max_papers = st.slider("Target Paper Count", min_value=2, max_value=12, value=5)

    st.markdown("---")
    st.header("📄 Upload Custom Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF Papers (optional)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Parse local PDFs into the synthesis corpus alongside repository queries."
    )

    st.markdown("---")
    st.caption("AI Agent on Literature Review Assistant • Built for researchers, engineers, and scholars.")


# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">📚 Literature Review Assistant AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Autonomous academic discovery, structured metadata extraction, comparative matrix synthesis, and gap analysis.</div>',
    unsafe_allow_html=True
)

# Topic input section
col_input, col_btn = st.columns([4, 1])

with col_input:
    preset_topics = [
        "",
        "Diffusion Models for Medical Image Segmentation",
        "Retrieval-Augmented Generation Hallucination Mitigation",
        "Test-Time Compute and Reasoning in Large Language Models",
        "Multi-Agent Reinforcement Learning for Autonomous Vehicles",
        "State Space Models and Mamba for Long-Context Sequence Modeling"
    ]
    preset_choice = st.selectbox("Quick Suggestions / Presets:", options=preset_topics, index=0)
    
    query = st.text_input(
        "Research Topic or Systematic Question:",
        value=preset_choice if preset_choice else "Diffusion Models for Medical Image Segmentation",
        placeholder="e.g. Memory efficient Transformer attention architectures"
    )

with col_btn:
    st.write("")
    st.write("")
    generate_clicked = st.button("🚀 Run Agent", type="primary", use_container_width=True)

# Process PDF uploads
parsed_uploaded_papers = []
if uploaded_files:
    for f in uploaded_files:
        try:
            bytes_data = f.read()
            p = parse_pdf_bytes(bytes_data, filename=f.name)
            parsed_uploaded_papers.append(p)
        except Exception as e:
            st.warning(f"Could not parse uploaded file {f.name}: {e}")

# Trigger Pipeline
if generate_clicked and query:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_ui(msg, pct):
        progress_bar.progress(pct)
        status_text.info(f"**Step {pct}%**: {msg}")

    try:
        agent = LiteratureReviewAgent(
            api_key=api_key if api_key.strip() else None,
            provider=provider,
            model_name=model_name
        )

        with st.spinner("Multi-Agent Review Pipeline in progress..."):
            result = agent.run_literature_review(
                topic=query,
                max_papers=max_papers,
                use_arxiv=use_arxiv,
                use_semantic_scholar=use_s2,
                uploaded_papers=parsed_uploaded_papers,
                status_callback=update_ui
            )
            st.session_state.review_result = result
            st.session_state.chat_history = []
            status_text.success("Literature Review generation complete!")
            progress_bar.progress(100)

    except Exception as err:
        st.error(f"Error during execution: {err}")
        st.exception(err)


# ----------------- RESULTS PRESENTATION -----------------
res: ReviewResult = st.session_state.review_result

if res:
    st.markdown("---")
    
    # Export bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "📥 Download Markdown (.md)",
            data=export_markdown(res),
            file_name=f"literature_review_{res.topic.replace(' ', '_')[:30]}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "📊 Download Matrix (.csv)",
            data=export_matrix_csv(res),
            file_name=f"comparative_matrix_{res.topic.replace(' ', '_')[:30]}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c3:
        st.download_button(
            "🏷️ Download BibTeX (.bib)",
            data=res.references_bibtex,
            file_name="references.bib",
            mime="text/plain",
            use_container_width=True
        )
    with c4:
        st.download_button(
            "🌐 Download Web Report (.html)",
            data=export_html(res),
            file_name=f"review_{res.topic.replace(' ', '_')[:30]}.html",
            mime="text/html",
            use_container_width=True
        )

    # Tabs
    tab_paper, tab_matrix, tab_gaps, tab_papers, tab_bib, tab_chat = st.tabs([
        "📄 Literature Review",
        "📊 Comparative Matrix",
        "🔍 Research Gaps & Roadmap",
        "📚 Analyzed Papers",
        "🏷️ BibTeX References",
        "💬 Interactive Q&A"
    ])

    with tab_paper:
        st.header(f"Systematic Review: {res.topic}")
        st.subheader("Executive Abstract")
        st.write(res.executive_summary)

        st.subheader("1. Introduction & Background")
        st.markdown(res.introduction)

        st.subheader("2. Thematic Analysis")
        for idx, t in enumerate(res.thematic_analysis, 1):
            st.markdown(f"**2.{idx} {t.theme_name}**")
            st.write(t.description)
            st.info(f"**Key Insights:** {t.key_takeaways}")

        st.subheader("3. Critical Synthesis & Discussion")
        st.markdown(res.critical_synthesis)

        st.subheader("4. Strategic Conclusion")
        st.markdown(res.conclusion)

    with tab_matrix:
        st.header("📊 Comparative Literature Matrix")
        st.caption("Side-by-side methodological comparison across problems, algorithms, datasets, findings, and limitations.")
        if res.comparative_matrix:
            df_data = []
            for r in res.comparative_matrix:
                df_data.append({
                    "Citation": f"{r.authors_short} ({r.year})",
                    "Title": r.title,
                    "Problem Addressed": r.problem,
                    "Methodology": r.methodology,
                    "Datasets / Benchmarks": r.datasets,
                    "Key Findings": r.key_findings,
                    "Limitations": r.limitations
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, height=450)
        else:
            st.info("No matrix items generated.")

    with tab_gaps:
        st.header("🔍 Identified Research Gaps & Strategic Roadmap")
        st.caption("Critical bottlenecks, empirical contradictions, and unaddressed frontiers extracted across the corpus.")
        for idx, g in enumerate(res.research_gaps, 1):
            st.markdown(f"""
            <div class="gap-card">
                <h4 style="margin-top:0; color:#1E3A8A;">{idx}. {g.title}</h4>
                <p><b>Identified Gap:</b> {g.description}</p>
                <p><b>Conflicting / Unresolved Friction:</b> {g.conflicting_or_unresolved_aspects}</p>
                <p><b>Recommended Future Investigation:</b> {g.suggested_future_direction}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_papers:
        st.header(f"📚 Surveyed Corpus ({len(res.papers)} Papers)")
        for idx, p in enumerate(res.papers, 1):
            with st.expander(f"[{idx}] {p.title} ({p.year or 'n.d.'})"):
                st.markdown(f"**Authors:** {', '.join(p.authors)}")
                st.markdown(f"**Venue:** {p.venue} | **Source:** `{p.source}`")
                if p.url:
                    st.markdown(f"🔗 [Paper Link]({p.url}) " + (f"| 📥 [PDF Direct]({p.pdf_url})" if p.pdf_url else ""))
                st.markdown("**Abstract:**")
                st.write(p.abstract)
                st.markdown("---")
                st.markdown(f"- **Problem:** {p.problem_statement}")
                st.markdown(f"- **Methodology:** {p.methodology}")
                st.markdown(f"- **Datasets:** {p.datasets_and_benchmarks}")
                st.markdown(f"- **Findings:** {p.key_findings}")
                st.markdown(f"- **Limitations:** {p.limitations}")

    with tab_bib:
        st.header("🏷️ BibTeX References")
        st.caption("Standard BibTeX bibliography ready to paste into your LaTeX document or reference manager.")
        st.code(res.references_bibtex, language="bibtex")

    with tab_chat:
        st.header("💬 Interactive Literature Corpus Q&A")
        st.caption("Ask specific questions across the surveyed papers (e.g. 'Compare accuracy across papers', 'What are common failure modes?').")

        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask a question regarding the surveyed literature...")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Synthesizing answer from literature corpus..."):
                    agent_for_qa = LiteratureReviewAgent(
                        api_key=api_key if api_key.strip() else None,
                        provider=provider,
                        model_name=model_name
                    )
                    answer = agent_for_qa.answer_query(user_q, res)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Enter a research topic above or choose a preset suggestion, then click **'Run Agent'** to start.")
