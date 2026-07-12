import streamlit as st
import io
import contextlib
from app.agent.orchestrator import run_agent

st.set_page_config(page_title="Market Research Agent", page_icon="📈")

st.title(" Market Research Agent")
st.caption("Agentic RAG over financial news — ask a question, watch it decide whether to search.")

question = st.text_input("Ask something about a company, market event, or stock:")

if st.button("Ask") and question:
    with st.spinner("Agent is thinking..."):
        # capture the [Agent] print statements so we can show the reasoning trace
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            answer = run_agent(question)

        trace = buffer.getvalue()

    if trace.strip():
        with st.expander("Agent's reasoning trace"):
            st.text(trace)

    st.subheader("Answer")

    st.write(answer)