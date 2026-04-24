"""
agent/graph.py — ReAct Agent with LangGraph

This is the AGENTIC core. The LLM is in a loop:
    Observe → Think → Act (call a tool) → Observe result → Think again → ...

The LLM decides:
- WHICH tools to call (not all tools run every time)
- In WHAT ORDER (different queries take different paths)
- WHEN to stop (the agent decides it has enough info)

This is fundamentally different from a pipeline.
A pipeline runs the same steps every time.
An agent adapts its behavior to each query.
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import ALL_TOOLS
from agent.memory import load_preferences, get_squad_state_summary

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── System Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are FantasyEdge, an expert Fantasy Premier League analyst agent.

You have access to tools that give you LIVE FPL data. Use them to answer questions.

IMPORTANT RULES:
1. ALWAYS use tools to get data. Never make up player stats or prices.
2. Use the scoring system's composite scores to rank players, not just one metric.
3. When building a squad, ALWAYS use the build_squad tool — it enforces FPL constraints.
   Never try to manually select 15 players, you'll violate budget or position rules.
4. Check availability for any player you're recommending — don't pick injured players.
5. When recommending a captain, explain WHY with specific stats.
6. For every pick, mention the confidence level (HIGH/MEDIUM/LOW) from the scoring engine.
7. If the user has preferences stored, respect them (check with manage_memory tool).

SCORING SYSTEM (for reference — you don't calculate this, the tools do):
Score = form×3 + xGI×2.5 + value×1.5 + fixture_ease×2 + availability×5 + momentum×0.5
Confidence = based on minutes played, form-xG agreement, fixture clarity, and fitness.

YOUR APPROACH:
- For broad questions ("pick my team"): check gameweek → get top players → check fixtures → check availability → build squad
- For specific questions ("is Salah worth it?"): look up that player + check fixtures
- For transfers ("who should I replace Saka with?"): check Saka's status → find alternatives at same position/price
- Always think step-by-step and explain your reasoning.
- Be specific — use numbers, scores, and fixture details in every recommendation.
- Give direct, specific recommendations. Don't hedge with 'it depends'. Make a clear YES or NO recommendation with your reasoning

{preferences_context}
{squad_context}
"""


# ── State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Build the Agent ──────────────────────────────────────────────────

def _get_system_message() -> SystemMessage:
    """Build system prompt with current preferences and squad state."""
    prefs = load_preferences()
    squad_summary = get_squad_state_summary()

    prefs_context = ""
    if prefs.get("favorite_team"):
        prefs_context += f"\nUser's favorite team: {prefs['favorite_team']}"
    if prefs.get("must_include_players"):
        prefs_context += f"\nMust include: {', '.join(prefs['must_include_players'])}"
    if prefs.get("never_pick_players"):
        prefs_context += f"\nNever pick: {', '.join(prefs['never_pick_players'])}"
    if prefs.get("risk_tolerance") != "medium":
        prefs_context += f"\nRisk tolerance: {prefs['risk_tolerance']}"

    squad_context = ""
    if "No squad saved" not in squad_summary:
        squad_context = f"\nUser's current squad state:\n{squad_summary}"

    prompt = SYSTEM_PROMPT.format(
        preferences_context=prefs_context if prefs_context else "\nNo user preferences set.",
        squad_context=squad_context if squad_context else "\nNo current squad saved.",
    )

    return SystemMessage(content=prompt)


def build_agent():
    """
    Build the ReAct agent graph.

    The graph has just 2 nodes:
    1. "agent" — the LLM thinks and decides what to do
    2. "tools" — executes whatever tool the LLM chose

    And 1 conditional edge:
    - After "agent": if the LLM called a tool → go to "tools"
    - After "agent": if the LLM didn't call a tool → go to END (it's done)
    - After "tools": always go back to "agent" (so it can think about the result)

    This creates the ReAct loop:
    agent → tools → agent → tools → agent → ... → END

    The loop continues until the LLM decides it has enough information
    and produces a final text response without calling any tool.
    """
    # Initialize LLM with tools
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ── Agent Node ───────────────────────────────────────────────────
    def agent_node(state: AgentState):
        """The LLM reasons about the current state and decides what to do."""
        messages = state["messages"]

        # Inject system prompt if this is the first call
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [_get_system_message()] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── Should Continue? ─────────────────────────────────────────────
    def should_continue(state: AgentState):
        """
        Check the last message. If the LLM called a tool, continue to tools node.
        If not (just text), we're done — go to END.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ── Build Graph ──────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    # Edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")  # After tools, always go back to agent

    return workflow.compile()


# ── Run Function ─────────────────────────────────────────────────────

def chat(message: str, history: list = None) -> dict:
    """
    Send a message to the agent and get a response.

    Args:
        message: User's message
        history: Previous messages for conversation continuity

    Returns:
        dict with 'response' (text), 'tool_calls' (list of tools used),
        'messages' (full message history for continuity)
    """
    if not os.getenv("GROQ_API_KEY"):
        return {
            "response": (
                "Missing GROQ_API_KEY. Add it to your environment (or Streamlit secrets) "
                "to enable agent reasoning and tool orchestration."
            ),
            "tool_calls": [],
            "messages": history or [],
        }

    agent = build_agent()

    # Build message list with history
    messages = []
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    # Run the agent
    try:
        result = agent.invoke({"messages": messages}, config={"recursion_limit": 40})
    except Exception as e:
        return {
            "response": (
                "I hit an internal agent/tool execution error and stopped safely. "
                f"Details: {e}"
            ),
            "tool_calls": [],
            "messages": messages,
        }

    # Extract the final response and tool calls
    all_messages = result["messages"]
    final_response = ""
    tool_calls_made = []

    for msg in all_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                })
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            final_response = msg.content

    # If no clean final response, get the last AI message with content
    if not final_response:
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break

    return {
        "response": final_response,
        "tool_calls": tool_calls_made,
        "messages": all_messages,
    }


# ── CLI for testing ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("⚽ FantasyEdge Agent — Type 'quit' to exit\n")

    history = None
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = chat(user_input, history)
        print(f"\nAgent: {result['response']}")

        if result["tool_calls"]:
            print(f"\n  🔧 Tools used: {', '.join(tc['tool'] for tc in result['tool_calls'])}")

        history = result["messages"]
        print()
