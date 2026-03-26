"""
agent/memory.py — Three-Layer Memory System

1. Conversation Memory (short-term):
   Tracks what was discussed in the current session.
   Managed by LangGraph's message history — no custom code needed.

2. User Preferences (long-term):
   Persists across sessions. Stores things like:
   - "I'm a Liverpool fan, always include at least one Liverpool player"
   - "I prefer budget picks over premium"
   - "Never captain a defender"

3. Squad State Memory:
   Remembers the user's current FPL team, budget, free transfers.
   Enables transfer suggestions instead of just full rebuilds.
"""

import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PREFERENCES_FILE = os.path.join(DATA_DIR, "preferences.json")
SQUAD_STATE_FILE = os.path.join(DATA_DIR, "squad_state.json")

# ── Default Preferences ──────────────────────────────────────────────
DEFAULT_PREFERENCES = {
    "favorite_team": None,           # e.g., "Liverpool"
    "must_include_players": [],       # Players to always consider
    "never_pick_players": [],         # Players to avoid
    "prefer_budget_picks": False,     # Prioritize value over premium
    "captain_preference": None,       # e.g., "always attacking player"
    "risk_tolerance": "medium",       # "low" = safe picks, "high" = differentials
    "notes": [],                      # Free-text user notes
    "updated_at": None,
}

# ── Default Squad State ──────────────────────────────────────────────
DEFAULT_SQUAD_STATE = {
    "current_squad": [],              # List of player names
    "budget_remaining": 0.0,          # Available budget for transfers
    "free_transfers": 1,              # Usually 1 per week, max 5
    "chips_available": {
        "wildcard": True,
        "free_hit": True,
        "bench_boost": True,
        "triple_captain": True,
    },
    "gameweek_history": [],           # Past picks and points
    "updated_at": None,
}


# ── Preferences (Long-term Memory) ──────────────────────────────────

def load_preferences() -> dict:
    """Load user preferences from disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PREFERENCES_FILE):
        return DEFAULT_PREFERENCES.copy()
    try:
        with open(PREFERENCES_FILE, "r") as f:
            prefs = json.load(f)
        # Merge with defaults for any missing keys
        merged = DEFAULT_PREFERENCES.copy()
        merged.update(prefs)
        return merged
    except Exception:
        return DEFAULT_PREFERENCES.copy()


def save_preferences(prefs: dict):
    """Save user preferences to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    prefs["updated_at"] = datetime.now().isoformat()
    with open(PREFERENCES_FILE, "w") as f:
        json.dump(prefs, f, indent=2)


def update_preference(key: str, value) -> str:
    """Update a single preference. Returns confirmation message."""
    prefs = load_preferences()
    if key not in prefs:
        return f"Unknown preference: {key}. Valid keys: {list(DEFAULT_PREFERENCES.keys())}"

    prefs[key] = value
    save_preferences(prefs)
    return f"Preference updated: {key} = {value}"


def get_preferences_summary() -> str:
    """Get a human-readable summary of current preferences."""
    prefs = load_preferences()
    lines = ["Current preferences:"]

    if prefs["favorite_team"]:
        lines.append(f"  Favorite team: {prefs['favorite_team']}")
    if prefs["must_include_players"]:
        lines.append(f"  Must include: {', '.join(prefs['must_include_players'])}")
    if prefs["never_pick_players"]:
        lines.append(f"  Never pick: {', '.join(prefs['never_pick_players'])}")
    if prefs["prefer_budget_picks"]:
        lines.append("  Strategy: Prefer budget picks over premium")
    if prefs["captain_preference"]:
        lines.append(f"  Captain preference: {prefs['captain_preference']}")
    lines.append(f"  Risk tolerance: {prefs['risk_tolerance']}")
    if prefs["notes"]:
        lines.append(f"  Notes: {'; '.join(prefs['notes'])}")

    if len(lines) == 1:
        lines.append("  No preferences set. Tell me about your FPL style!")

    return "\n".join(lines)


# ── Squad State Memory ───────────────────────────────────────────────

def load_squad_state() -> dict:
    """Load current squad state from disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SQUAD_STATE_FILE):
        return DEFAULT_SQUAD_STATE.copy()
    try:
        with open(SQUAD_STATE_FILE, "r") as f:
            state = json.load(f)
        merged = DEFAULT_SQUAD_STATE.copy()
        merged.update(state)
        return merged
    except Exception:
        return DEFAULT_SQUAD_STATE.copy()


def save_squad_state(state: dict):
    """Save squad state to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat()
    with open(SQUAD_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def update_squad(squad_names: list, budget_remaining: float = 0.0):
    """Update the stored squad after a recommendation is accepted."""
    state = load_squad_state()
    state["current_squad"] = squad_names
    state["budget_remaining"] = budget_remaining
    save_squad_state(state)
    return f"Squad state saved: {len(squad_names)} players, £{budget_remaining}m remaining"


def record_gameweek(gameweek: int, squad_names: list, points: int = 0):
    """Record a gameweek result for historical tracking."""
    state = load_squad_state()
    state["gameweek_history"].append({
        "gameweek": gameweek,
        "squad": squad_names,
        "points": points,
        "timestamp": datetime.now().isoformat(),
    })
    save_squad_state(state)


def get_squad_state_summary() -> str:
    """Get human-readable squad state."""
    state = load_squad_state()

    if not state["current_squad"]:
        return "No squad saved. Ask me to pick your team and I'll remember it."

    lines = [
        f"Current squad ({len(state['current_squad'])} players):",
        f"  Players: {', '.join(state['current_squad'])}",
        f"  Budget remaining: £{state['budget_remaining']}m",
        f"  Free transfers: {state['free_transfers']}",
    ]

    chips = [name for name, available in state["chips_available"].items() if available]
    if chips:
        lines.append(f"  Chips available: {', '.join(chips)}")

    history = state.get("gameweek_history", [])
    if history:
        recent = history[-3:]
        lines.append(f"  Recent history: {len(history)} gameweeks tracked")
        for gw in recent:
            lines.append(f"    GW{gw['gameweek']}: {gw.get('points', '?')} pts")

    return "\n".join(lines)
