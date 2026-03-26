"""
ui/app.py — Streamlit Frontend with Reasoning Trace

Features:
- Chat interface for conversational FPL advice
- Visible tool calls (what the agent decided to do)
- Reasoning trace toggle (see the agent's thought process)
- Memory awareness (shows preferences and squad state)

Run with: streamlit run ui/app.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Handle API key from Streamlit Cloud secrets or .env
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from agent.graph import build_agent
from agent.memory import get_preferences_summary, get_squad_state_summary

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="FantasyEdge", page_icon="⚽", layout="wide")

# ── Session State Init ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "agent" not in st.session_state:
    with st.spinner("Loading FantasyEdge agent..."):
        st.session_state.agent = build_agent()
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = True

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ FantasyEdge")
    st.markdown("**Agentic FPL Analyst**")
    st.caption("Powered by LangGraph ReAct + Groq")

    st.divider()

    # Reasoning trace toggle
    st.session_state.show_reasoning = st.toggle(
        "Show reasoning trace", value=st.session_state.show_reasoning
    )

    st.divider()

    # Memory display
    with st.expander("🧠 Agent Memory"):
        st.markdown("**Preferences:**")
        st.text(get_preferences_summary())
        st.markdown("**Squad State:**")
        st.text(get_squad_state_summary())

    st.divider()

    st.markdown("### Try asking:")
    examples = [
        "Pick my team for this gameweek",
        "Is Salah worth the price?",
        "Best midfielders under £8m",
        "Who should I captain?",
        "Compare Haaland vs Watkins",
        "I already have Palmer and Saka, build around them",
        "Which teams have easy fixtures?",
        "Remember that I'm a Liverpool fan",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.prefill = ex

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.rerun()

# ── Main Chat Area ───────────────────────────────────────────────────
st.title("⚽ FantasyEdge")
st.caption("AI-powered FPL analyst — ask anything about your Fantasy Premier League team")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show tool calls and reasoning for assistant messages
        if msg["role"] == "assistant" and st.session_state.show_reasoning:
            if msg.get("tool_calls"):
                with st.expander(f"🔧 Tools used ({len(msg['tool_calls'])})"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**{tc['tool']}**")
                        if tc.get("args"):
                            args_display = ", ".join(f"{k}={v}" for k, v in tc["args"].items() if v)
                            if args_display:
                                st.code(args_display, language=None)

            if msg.get("reasoning"):
                with st.expander("🧠 Reasoning trace"):
                    for step in msg["reasoning"]:
                        st.markdown(step)

# ── Handle Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask about FPL...") or prefill

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # Build messages for agent
                agent_messages = list(st.session_state.agent_history)
                agent_messages.append(HumanMessage(content=user_input))

                # Run the agent graph
                result = st.session_state.agent.invoke({"messages": agent_messages})

                all_messages = result["messages"]

                # Extract tool calls and reasoning trace
                tool_calls = []
                reasoning = []
                final_response = ""

                for msg in all_messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append({
                                "tool": tc["name"],
                                "args": tc.get("args", {}),
                            })
                            reasoning.append(f"🔧 Called **{tc['name']}**({', '.join(f'{k}={v}' for k, v in tc.get('args', {}).items() if v)})")

                    if isinstance(msg, ToolMessage):
                        # Truncate long tool outputs for display
                        content = str(msg.content)[:300]
                        reasoning.append(f"📊 Got result: `{content}...`" if len(str(msg.content)) > 300 else f"📊 Got result: `{content}`")

                    if isinstance(msg, AIMessage) and msg.content:
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_response = msg.content

                # If no clean final, get last AI message
                if not final_response:
                    for msg in reversed(all_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = msg.content
                            break

                if not final_response:
                    final_response = "I wasn't able to generate a response. Please try again."

                # Display response
                st.markdown(final_response)

                # Show reasoning trace
                if st.session_state.show_reasoning:
                    if tool_calls:
                        with st.expander(f"🔧 Tools used ({len(tool_calls)})"):
                            for tc in tool_calls:
                                st.markdown(f"**{tc['tool']}**")
                                args_display = ", ".join(f"{k}={v}" for k, v in tc["args"].items() if v)
                                if args_display:
                                    st.code(args_display, language=None)

                    if reasoning:
                        with st.expander("🧠 Reasoning trace"):
                            for step in reasoning:
                                st.markdown(step)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "tool_calls": tool_calls,
                    "reasoning": reasoning,
                })

                # Save agent message history for conversation continuity
                st.session_state.agent_history = all_messages

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.warning("⏳ Rate limit reached — please wait 60 seconds and try again.")
            else:
                st.error(f"Something went wrong: {error_msg}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}",
            })
