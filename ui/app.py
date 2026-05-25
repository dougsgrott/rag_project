"""Streamlit chatbot for the simple-stack reference RAG.

The UI's job is to render chat and forward user input to the Orchestration
Layer. It never instantiates adapters — it pulls them from the composition
root (`compose.build_simple_stack`) and calls only stage-interface methods.

Streamlit re-runs this script on every interaction; we rebuild the stack
per call rather than caching async resources across reruns, so adapter
lifecycle is always clean (ADR-0007). Per-call overhead is the cost of one
SQLite open and one AsyncOpenAI client init — negligible against network
latency.

`conversation_id` and `domain` round-trip through URL query params so a
browser refresh restores the active conversation.

Run with:

    uv run --group openai --group chroma --group ui streamlit run ui/app.py
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st

from compose import build_simple_stack
from rag.errors import ConfigurationError, RAGError
from rag.pipeline.query import answer_query
from rag.types import Message


# --- Async bridges ---------------------------------------------------------
# Each helper opens a fresh stack via `async with`, runs one operation, and
# tears the stack down. Streamlit re-runs the whole script on every user
# interaction, so persistent state belongs in SQLite/Chroma — not in memory.


async def _load_history(conversation_id: str) -> list[Message]:
    async with build_simple_stack() as s:
        return await s.conversation_store.get_history(conversation_id)


async def _list_conversations() -> list[str]:
    async with build_simple_stack() as s:
        return await s.conversation_store.list_conversations()


async def _read_prompt(domain: str) -> str | None:
    async with build_simple_stack() as s:
        try:
            return await s.prompt_store.get_prompt(domain)
        except ConfigurationError:
            return None


async def _save_prompt(domain: str, prompt: str, author: str) -> None:
    async with build_simple_stack() as s:
        await s.prompt_store.save_prompt(domain, prompt, author)


async def _send_message(*, conversation_id: str, domain: str, query: str) -> Message:
    async with build_simple_stack() as s:
        return await answer_query(
            prompt_store=s.prompt_store,
            conversation_store=s.conversation_store,
            query_rewriter=s.query_rewriter,
            vector_store=s.vector_store,
            reranker=s.reranker,
            generator=s.generator,
            conversation_id=conversation_id,
            domain=domain,
            query=query,
        )


# --- View ------------------------------------------------------------------


def _sidebar() -> tuple[str, str]:
    """Render the sidebar; return the selected (conversation_id, domain)."""
    qp = st.query_params

    with st.sidebar:
        st.header("Session")

        try:
            existing = asyncio.run(_list_conversations())
        except RAGError as e:
            st.error(f"Could not list conversations: {e}")
            existing = []

        current = qp.get("conv", "demo-1")
        options = list(dict.fromkeys([current, *existing]))
        picked = st.selectbox(
            "Load conversation",
            options=options,
            index=options.index(current),
            help="Pick from previously-saved conversations, or type a new ID below.",
        )
        if picked != current:
            qp["conv"] = picked
            st.rerun()

        conversation_id = st.text_input(
            "Conversation ID",
            value=picked,
            help="Identifier for the multi-turn conversation. "
            "Stored in the URL so a refresh restores it.",
        )
        if st.button("🆕 New conversation"):
            qp["conv"] = f"conv-{uuid.uuid4().hex[:8]}"
            st.rerun()

        domain = st.text_input("Domain", value=qp.get("domain", "default"))

        # Sync URL params so a refresh restores both fields.
        if qp.get("conv") != conversation_id:
            qp["conv"] = conversation_id
        if qp.get("domain") != domain:
            qp["domain"] = domain

        with st.expander("System prompt", expanded=False):
            try:
                current = asyncio.run(_read_prompt(domain))
            except RAGError as e:
                st.error(f"Could not read prompt store: {e}")
                current = None

            if current is None:
                st.info(f"No prompt saved for domain `{domain}` yet.")
            else:
                st.caption("Current prompt:")
                st.code(current, language="text")

            new_prompt = st.text_area("New prompt", height=140, key="prompt_text")
            author = st.text_input("Author", value="ui-user", key="prompt_author")
            if st.button("Save prompt"):
                trimmed = new_prompt.strip()
                if not trimmed:
                    st.warning("Prompt cannot be empty.")
                else:
                    try:
                        asyncio.run(_save_prompt(domain, trimmed, author))
                        st.success(f"Saved prompt for `{domain}` (author: {author}).")
                        st.rerun()
                    except RAGError as e:
                        st.error(f"Save failed: {e}")

    return conversation_id, domain


def _render_history(conversation_id: str) -> None:
    try:
        history = asyncio.run(_load_history(conversation_id))
    except RAGError as e:
        st.error(f"Failed to load conversation history: {e}")
        return

    for msg in history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)


def _handle_input(*, conversation_id: str, domain: str) -> None:
    user_input = st.chat_input("Ask a question about your indexed documents…")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = asyncio.run(
                    _send_message(
                        conversation_id=conversation_id,
                        domain=domain,
                        query=user_input,
                    )
                )
            except ConfigurationError as e:
                st.error(
                    f"⚠️ {e}\n\n"
                    f"Set a system prompt for domain **`{domain}`** via the sidebar, "
                    "or run from the CLI:\n\n"
                    f"```\npython compose.py set-prompt {domain} \"<prompt text>\"\n```"
                )
                return
            except RAGError as e:
                st.error(f"{type(e).__name__}: {e}")
                return
            st.markdown(answer.content)

    # Reload from the store so the rendered history equals what was persisted.
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="RAG Reference", page_icon="📚", layout="centered")
    st.title("📚 RAG Reference Chatbot")

    conversation_id, domain = _sidebar()
    st.caption(f"Conversation: `{conversation_id}` · Domain: `{domain}`")

    _render_history(conversation_id)
    _handle_input(conversation_id=conversation_id, domain=domain)


if __name__ == "__main__":
    main()
