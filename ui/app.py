"""
ui/app.py — FantasyEdge Streamlit Frontend

Features:
- Clean graphite analytics interface with emerald accents
- Player card rendering when squad is returned
- Visible tool calls as compact pills
- Reasoning trace toggle
- Memory awareness in sidebar
- Gameweek (not GW) everywhere

Run with: streamlit run ui/app.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from agent.graph import build_agent
from agent.memory import get_preferences_summary, get_squad_state_summary

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="FantasyEdge", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --fe-bg: #0b0f0e;
    --fe-sidebar: #0e1412;
    --fe-surface: #121816;
    --fe-surface-raised: #17201d;
    --fe-border: #26312d;
    --fe-border-strong: #34413c;
    --fe-text: #e8efec;
    --fe-muted: #8f9d97;
    --fe-accent: #34d399;
    --fe-accent-soft: rgba(52, 211, 153, 0.1);
    --fe-radius: 8px;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp { background: var(--fe-bg); }

[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1120px;
    padding: 2.25rem 2.5rem 7rem;
}

[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 {
    color: var(--fe-text) !important;
    letter-spacing: -0.025em;
}

h1 {
    font-size: clamp(2rem, 4vw, 3rem) !important;
    line-height: 1.05 !important;
    font-weight: 700 !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stCaptionContainer"] { color: var(--fe-muted); }
hr { border-color: var(--fe-border) !important; }

[data-testid="stSidebar"] {
    background: var(--fe-sidebar) !important;
    border-right: 1px solid var(--fe-border);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.25rem;
}

[data-testid="stSidebar"] h2 {
    color: var(--fe-text) !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #b5c0bb;
}

.stButton button {
    min-height: 2.5rem;
    background: var(--fe-surface) !important;
    border: 1px solid var(--fe-border) !important;
    border-radius: var(--fe-radius) !important;
    color: #c9d3cf !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
}

.stButton button:hover {
    background: var(--fe-surface-raised) !important;
    border-color: var(--fe-border-strong) !important;
    color: var(--fe-text) !important;
}

.stButton button:focus-visible,
button:focus-visible,
input:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--fe-accent) !important;
    outline-offset: 2px !important;
}

[data-testid="stSidebar"] .stButton button { text-align: left !important; }

[data-testid="stExpander"] {
    background: var(--fe-surface) !important;
    border: 1px solid var(--fe-border) !important;
    border-radius: var(--fe-radius) !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] summary:hover { color: var(--fe-accent) !important; }

[data-testid="stChatMessage"] {
    background: var(--fe-surface);
    border: 1px solid var(--fe-border);
    border-radius: 10px;
    margin-bottom: 0.75rem;
    padding: 0.35rem 0.5rem;
    box-shadow: none;
}

[data-testid="stChatMessage"] p:last-child { margin-bottom: 0; }

[data-testid="stChatInput"] {
    background: var(--fe-surface) !important;
    border: 1px solid var(--fe-border-strong) !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24) !important;
}

[data-testid="stChatInput"]:focus-within { border-color: var(--fe-accent) !important; }

[data-testid="stAlert"] {
    border: 1px solid var(--fe-border) !important;
    border-radius: var(--fe-radius) !important;
}

.tool-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.tool-pill {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: var(--fe-accent-soft);
    color: #7ee7bb;
    border: 1px solid rgba(52, 211, 153, 0.22);
    border-radius: 5px;
    padding: 3px 8px;
}

.squad-wrap { margin-top: 16px; }

.squad-header {
    background: var(--fe-surface-raised);
    border: 1px solid var(--fe-border);
    border-left: 3px solid var(--fe-accent);
    border-radius: var(--fe-radius);
    padding: 12px 14px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
}
.squad-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--fe-text);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.squad-meta { font-size: 11px; color: var(--fe-muted); }

.position-section { margin-bottom: 16px; }
.position-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--fe-muted);
    margin-bottom: 8px;
    font-weight: 600;
}
.cards-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(112px, 1fr));
    gap: 8px;
}

.player-card {
    background: var(--fe-surface);
    border: 1px solid var(--fe-border);
    border-radius: var(--fe-radius);
    min-width: 0;
    overflow: hidden;
    position: relative;
    box-shadow: none;
}
.player-card.captain  { border-color: #d6a846; }
.player-card.vice     { border-color: #788680; }
.player-card.bench    { opacity: 0.62; }

.pos-strip { height: 2px; width: 100%; }
.pos-strip.GKP { background: #d6a846; }
.pos-strip.DEF { background: #34d399; }
.pos-strip.MID { background: #60a5fa; }
.pos-strip.FWD { background: #f87171; }

.card-body { padding: 9px 10px 10px; }

.card-pos { font-size: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 3px; }
.card-pos.GKP { color: #e5bd68; }
.card-pos.DEF { color: #6ee7b7; }
.card-pos.MID { color: #93c5fd; }
.card-pos.FWD { color: #fca5a5; }

.card-name {
    font-size: 12px;
    font-weight: 650;
    color: var(--fe-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}
.card-team {
    font-size: 9px;
    color: var(--fe-muted);
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-divider { height: 1px; background: var(--fe-border); margin: 8px 0; }
.card-stats { display: flex; justify-content: space-between; align-items: center; }
.card-price { font-size: 10px; font-weight: 600; color: #cbd5d1; }

.conf-badge { font-size: 8px; font-weight: 700; padding: 2px 5px; border-radius: 4px; letter-spacing: 0.03em; }
.conf-HIGH   { background: rgba(52, 211, 153, 0.12); color: #6ee7b7; }
.conf-MEDIUM { background: rgba(214, 168, 70, 0.12); color: #e5bd68; }
.conf-LOW    { background: rgba(248, 113, 113, 0.12); color: #fca5a5; }

.captain-badge, .vice-badge {
    position: absolute;
    top: 5px; right: 5px;
    width: 14px; height: 14px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 8px; font-weight: 700; color: #fff;
}
.captain-badge { background: #b8892f; }
.vice-badge    { background: #59645f; }

.bench-divider {
    display: flex; align-items: center; gap: 8px;
    font-size: 9px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--fe-muted); margin: 14px 0 8px;
}
.bench-line { flex: 1; height: 1px; background: var(--fe-border); }

@media (max-width: 700px) {
    [data-testid="stAppViewContainer"] > .main .block-container {
        padding: 1.5rem 1rem 6rem;
    }
    .squad-header { align-items: flex-start; flex-direction: column; gap: 4px; }
    .cards-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""", unsafe_allow_html=True)


# ── Player Card HTML Builder ─────────────────────────────────────────

def make_player_card(name, team, position, price, confidence="", is_captain=False, is_vice=False, is_bench=False):
    card_class = "player-card"
    if is_captain: card_class += " captain"
    elif is_vice:  card_class += " vice"
    if is_bench:   card_class += " bench"

    badge = ""
    if is_captain: badge = '<div class="captain-badge">C</div>'
    elif is_vice:  badge = '<div class="vice-badge">V</div>'

    conf_html = ""
    if confidence and not is_bench:
        conf_short = {"HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}.get(confidence.upper(), "")
        if conf_short:
            conf_html = f'<span class="conf-badge conf-{confidence.upper()}">{conf_short}</span>'

    price_str = f"£{price}m" if price else ""

    return (
        f'<div class="{card_class}">'
        f'{badge}'
        f'<div class="pos-strip {position}"></div>'
        f'<div class="card-body">'
        f'<div class="card-pos {position}">{position}</div>'
        f'<div class="card-name">{name}</div>'
        f'<div class="card-team">{team}</div>'
        f'<div class="card-divider"></div>'
        f'<div class="card-stats">'
        f'<span class="card-price">{price_str}</span>'
        f'{conf_html}'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def parse_and_render_squad(response_text):
    """
    Detect if the response contains a squad and render player cards.
    Returns (intro_text, squad_html, bench_html).
    bench_html is rendered separately to avoid Streamlit HTML truncation.
    """
    lines = response_text.strip().split("\n")

    position_markers = ["GKP |", "DEF |", "MID |", "FWD |"]
    squad_lines = [l for l in lines if any(m in l for m in position_markers)]

    if len(squad_lines) < 5:
        return response_text, "", ""

    # Extract intro text before squad starts
    squad_start_idx = next(
        (i for i, l in enumerate(lines) if any(m in l for m in position_markers)), 0
    )
    intro_text = "\n".join(lines[:squad_start_idx]).strip()

    # Parse metadata
    formation = ""
    captain_name = ""
    vice_name = ""
    budget_line = ""

    for line in lines:
        if "Formation:" in line:
            formation = line.split("Formation:")[-1].strip()
        if "(C)" in line:
            m = re.search(r'\|\s*([^|]+?)\s*\(C\)', line)
            if m: captain_name = m.group(1).strip().split()[-1]
        if "(VC)" in line:
            m = re.search(r'\|\s*([^|]+?)\s*\(VC\)', line)
            if m: vice_name = m.group(1).strip().split()[-1]
        if "remaining" in line.lower() and "£" in line:
            budget_line = line.strip()

    def parse_player_line(line):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            return None
        position = parts[0].strip()
        name = re.sub(r'\s*\(C\)|\s*\(VC\)', '', parts[1]).strip()
        team = parts[2].strip() if len(parts) > 2 else ""
        price = ""
        confidence = ""
        for p in parts:
            pm = re.search(r'£([\d.]+)m', p)
            if pm: price = pm.group(1)
            cm = re.search(r'\[(HIGH|MEDIUM|LOW)\]', p)
            if cm: confidence = cm.group(1)
        return {"name": name, "team": team, "position": position, "price": price, "confidence": confidence}

    # Separate starting XI and bench
    in_bench = False
    starting = []
    bench = []

    for line in lines:
        stripped = line.strip()
        if "BENCH" in stripped.upper() and "|" not in stripped:
            in_bench = True
            continue
        if any(m in stripped for m in position_markers):
            p = parse_player_line(stripped)
            if p:
                if in_bench:
                    bench.append(p)
                else:
                    starting.append(p)

    if not starting:
        return response_text, "", ""

    # Group starting XI by position
    pos_order = ["GKP", "DEF", "MID", "FWD"]
    pos_labels = {"GKP": "Goalkeeper", "DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards"}
    grouped = {pos: [] for pos in pos_order}
    for p in starting:
        pos = p["position"].upper()
        if pos in grouped:
            grouped[pos].append(p)

    # Squad header + starting XI
    header_right = formation if formation else f"{len(starting)} players"
    squad_html = (
        f'<div class="squad-wrap">'
        f'<div class="squad-header">'
        f'<div><div class="squad-title">Recommended Squad</div>'
        f'<div class="squad-meta">{header_right}</div></div>'
        f'<div class="squad-meta">{budget_line}</div>'
        f'</div>'
    )

    for pos in pos_order:
        players = grouped[pos]
        if not players:
            continue
        cards = "".join(
            make_player_card(
                p["name"], p["team"], p["position"], p["price"], p["confidence"],
                is_captain=bool(captain_name and captain_name.lower() in p["name"].lower()),
                is_vice=bool(vice_name and vice_name.lower() in p["name"].lower()),
            )
            for p in players
        )
        squad_html += (
            f'<div class="position-section">'
            f'<div class="position-label">{pos_labels[pos]}</div>'
            f'<div class="cards-row">{cards}</div>'
            f'</div>'
        )

    squad_html += '</div>'  # close squad-wrap

    # Bench HTML returned separately to avoid Streamlit truncation
    bench_html = ""
    if bench:
        bench_cards = "".join(
            make_player_card(p["name"], p["team"], p["position"], p["price"], is_bench=True)
            for p in bench
        )
        bench_html = (
            f'<div class="bench-divider">'
            f'<div class="bench-line"></div>Bench<div class="bench-line"></div>'
            f'</div>'
            f'<div class="cards-row">{bench_cards}</div>'
        )

    return intro_text, squad_html, bench_html


def replace_gw(text):
    """Replace GW abbreviations with Gameweek throughout."""
    text = re.sub(r'\bGW(\d+)\b', r'Gameweek \1', text)
    text = re.sub(r'\bGW\b', 'Gameweek', text)
    return text


def render_response(content, tool_calls=None, reasoning=None):
    """Render agent response — with squad cards if present, plain markdown otherwise."""
    content = replace_gw(content)
    intro_text, squad_html, bench_html = parse_and_render_squad(content)

    if squad_html:
        if intro_text:
            st.markdown(intro_text)
        st.markdown(squad_html, unsafe_allow_html=True)
        if bench_html:
            st.markdown(bench_html, unsafe_allow_html=True)
    else:
        st.markdown(content)

    if tool_calls and st.session_state.get("show_reasoning"):
        pills_html = '<div class="tool-pills">' + "".join(
            f'<span class="tool-pill">{tc["tool"]}</span>'
            for tc in tool_calls
        ) + "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

    if reasoning and st.session_state.get("show_reasoning"):
        with st.expander("Reasoning trace"):
            for step in reasoning:
                st.markdown(replace_gw(step))


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
    st.markdown("## FantasyEdge")
    st.caption("FPL ANALYTICS · AI ASSISTED")

    st.divider()

    st.session_state.show_reasoning = st.toggle(
        "Show reasoning trace", value=st.session_state.show_reasoning
    )

    st.divider()

    with st.expander("Agent memory"):
        st.markdown("**Preferences:**")
        st.text(get_preferences_summary())
        st.markdown("**Squad State:**")
        st.text(get_squad_state_summary())

    st.divider()

    st.markdown("**Try asking:**")
    if st.button("Pick my team for this gameweek", use_container_width=True, key="ex_pick_team"):
        st.session_state.prefill = "Pick my team for this gameweek"

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.rerun()


# ── Main Chat Area ───────────────────────────────────────────────────
st.title("FantasyEdge")
st.caption("AI-powered analysis for smarter Fantasy Premier League decisions")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_response(
                msg["content"],
                tool_calls=msg.get("tool_calls"),
                reasoning=msg.get("reasoning"),
            )
        else:
            st.markdown(msg["content"])


# ── Handle Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask about FPL...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                agent_messages = list(st.session_state.agent_history)
                agent_messages.append(HumanMessage(content=user_input))

                result = st.session_state.agent.invoke({"messages": agent_messages})
                all_messages = result["messages"]

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
                            reasoning.append(
                                f"Called **{tc['name']}**"
                                f"({', '.join(f'{k}={v}' for k, v in tc.get('args', {}).items() if v)})"
                            )

                    if isinstance(msg, ToolMessage):
                        content_preview = str(msg.content)[:300]
                        suffix = "..." if len(str(msg.content)) > 300 else ""
                        reasoning.append(f"Result: `{content_preview}{suffix}`")

                    if isinstance(msg, AIMessage) and msg.content:
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_response = msg.content

                if not final_response:
                    for msg in reversed(all_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = msg.content
                            break

                if not final_response:
                    final_response = "I wasn't able to generate a response. Please try again."

                render_response(final_response, tool_calls=tool_calls, reasoning=reasoning)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "tool_calls": tool_calls,
                    "reasoning": reasoning,
                })

                st.session_state.agent_history = all_messages

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.warning("Rate limit reached — please wait 60 seconds and try again.")
            else:
                st.error(f"Something went wrong: {error_msg}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}",
            })
