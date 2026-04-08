"""
ui/app.py — FantasyEdge Streamlit Frontend

Features:
- Football analytics platform aesthetic (deep green sidebar)
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
st.set_page_config(page_title="FantasyEdge", page_icon="⚽", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500&display=swap');

[data-testid="stSidebar"] {
    background: #0d2b1a !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.75) !important;
}
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.05) !important;
    border: 0.5px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.6) !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(168,232,144,0.1) !important;
    color: #a8e890 !important;
    border-color: rgba(168,232,144,0.3) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown strong {
    color: #a8e890 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}

h1 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.5px; }

.tool-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tool-pill {
    font-size: 11px;
    background: #f0f4f0;
    color: #2a5a2a;
    border: 0.5px solid #c8dfc8;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'Barlow', sans-serif;
}

.squad-wrap { margin-top: 12px; }

.squad-header {
    background: #0d2b1a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.squad-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #a8e890;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.squad-meta { font-size: 11px; color: rgba(168,232,144,0.55); }

.position-section { margin-bottom: 10px; }
.position-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 6px;
    font-weight: 500;
}
.cards-row { display: flex; flex-wrap: wrap; gap: 7px; }

.player-card {
    background: #ffffff;
    border: 0.5px solid #e0e0e0;
    border-radius: 8px;
    width: 92px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.player-card.captain  { border: 1.5px solid #d4a017; }
.player-card.vice     { border: 1.5px solid #888; }
.player-card.bench    { opacity: 0.6; }

.pos-strip { height: 3px; width: 100%; }
.pos-strip.GKP { background: #e8a020; }
.pos-strip.DEF { background: #1a9e5a; }
.pos-strip.MID { background: #1a6ebe; }
.pos-strip.FWD { background: #c0392b; }

.card-body { padding: 6px 7px 7px; }

.card-pos { font-size: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1px; }
.card-pos.GKP { color: #b07010; }
.card-pos.DEF { color: #0f7a40; }
.card-pos.MID { color: #1050a0; }
.card-pos.FWD { color: #a02010; }

.card-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #1a1a1a;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.1;
}
.card-team {
    font-size: 9px;
    color: #888;
    margin-top: 1px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-divider { height: 0.5px; background: #eee; margin: 5px 0; }
.card-stats { display: flex; justify-content: space-between; align-items: center; }
.card-price { font-size: 10px; font-weight: 500; color: #333; }

.conf-badge { font-size: 9px; font-weight: 600; padding: 1px 5px; border-radius: 4px; }
.conf-HIGH   { background: #e8f5e2; color: #2e6b10; }
.conf-MEDIUM { background: #fef3e0; color: #8a5a0a; }
.conf-LOW    { background: #fce8e8; color: #902020; }

.captain-badge, .vice-badge {
    position: absolute;
    top: 5px; right: 5px;
    width: 14px; height: 14px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 8px; font-weight: 700; color: #fff;
}
.captain-badge { background: #d4a017; }
.vice-badge    { background: #777; }

.bench-divider {
    display: flex; align-items: center; gap: 8px;
    font-size: 9px; text-transform: uppercase; letter-spacing: 1px;
    color: #aaa; margin: 10px 0 6px;
}
.bench-line { flex: 1; height: 0.5px; background: #e0e0e0; }
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
            f'<span class="tool-pill">⚙ {tc["tool"]}</span>'
            for tc in tool_calls
        ) + "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

    if reasoning and st.session_state.get("show_reasoning"):
        with st.expander("🧠 Reasoning trace"):
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
    st.markdown("## ⚽ FantasyEdge")
    st.caption("Agentic FPL Analyst · LangGraph + Groq")

    st.divider()

    st.session_state.show_reasoning = st.toggle(
        "Show reasoning trace", value=st.session_state.show_reasoning
    )

    st.divider()

    with st.expander("🧠 Agent Memory"):
        st.markdown("**Preferences:**")
        st.text(get_preferences_summary())
        st.markdown("**Squad State:**")
        st.text(get_squad_state_summary())

    st.divider()

    st.markdown("**Try asking:**")
    examples = [
        "Pick my team for this gameweek",
        "Is Salah worth the price?",
        "Best midfielders under £8m",
        "Who should I captain?",
        "Compare Haaland vs Watkins",
        "I have Palmer and Saka, build around them",
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
                                f"⚙ Called **{tc['name']}**"
                                f"({', '.join(f'{k}={v}' for k, v in tc.get('args', {}).items() if v)})"
                            )

                    if isinstance(msg, ToolMessage):
                        content_preview = str(msg.content)[:300]
                        suffix = "..." if len(str(msg.content)) > 300 else ""
                        reasoning.append(f"📊 Result: `{content_preview}{suffix}`")

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
                st.warning("⏳ Rate limit reached — please wait 60 seconds and try again.")
            else:
                st.error(f"Something went wrong: {error_msg}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}",
            })
