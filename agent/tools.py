"""
agent/tools.py — Agent Tools

These are the 5 tools the LLM agent can CHOOSE to call.
The agent decides WHICH tools to use and in WHAT ORDER based on the user's query.
The developer does NOT control the execution path — the LLM does.

Tools:
1. get_player_stats — Search and rank players with filters
2. get_fixtures — Check upcoming fixture difficulty for a team
3. check_availability — Check if a specific player is fit
4. build_squad — Optimize a valid 15-player squad
5. get_preferences — Read/update user preferences and squad state
"""

import json
from langchain_core.tools import tool
from core.fpl_data import get_all_data, get_team_fixtures
from core.scoring import score_all_players, get_top_players, explain_score
from core.optimizer import optimize_squad, format_squad_summary
from agent.memory import (
    load_preferences, update_preference, get_preferences_summary,
    load_squad_state, get_squad_state_summary, update_squad, save_squad_state,
)

# ── Shared data (loaded once, reused across tool calls) ──────────────
_data_cache = {}


def _ensure_data():
    """Load and score data if not already cached for this session."""
    if "players" not in _data_cache:
        data = get_all_data()
        data["players"] = score_all_players(data["players"], data["fixture_scores"])
        _data_cache.update(data)
    return _data_cache


def _load_data_or_error() -> tuple:
    """Safely load data and return (data, error_message)."""
    try:
        return _ensure_data(), ""
    except Exception as e:
        return None, (
            "I couldn't load FPL data right now, so this tool call could not complete. "
            f"Error: {e}. Try again in a moment."
        )


# ── Tool 1: Get Player Stats ────────────────────────────────────────

@tool
def get_player_stats(
    position: str = "",
    min_form: float = 0.0,
    max_price: float = 20.0,
    team: str = "",
    player_name: str = "",
    limit: int = 10,
) -> str:
    """Search and rank FPL players with optional filters.

    Use this tool to find the best players based on form, value, fixtures, and availability.
    You can filter by position (GKP, DEF, MID, FWD), minimum form, maximum price, or team name.
    To look up a specific player, use the player_name parameter.
    Returns players sorted by our composite score which factors in form, xG, value, fixtures, and availability.

    Args:
        position: Filter by position — one of "GKP", "DEF", "MID", "FWD". Leave empty for all.
        min_form: Minimum form threshold. Good form is 5+, excellent is 7+.
        max_price: Maximum price in millions. Budget picks are under 6.0, premiums are 10.0+.
        team: Filter by team name (partial match works, e.g. "Arsenal", "Man City").
        player_name: Search for a specific player by name.
        limit: Number of players to return (default 10).
    """
    data, error = _load_data_or_error()
    if error:
        return error
    players_df = data["players"]

    # Specific player lookup
    if player_name:
        match = players_df[
            players_df["name"].str.lower().str.contains(player_name.lower()) |
            players_df["full_name"].str.lower().str.contains(player_name.lower())
        ]
        if match.empty:
            return f"Player '{player_name}' not found. Try a different spelling or check if they're in the Premier League."

        results = []
        for _, row in match.head(3).iterrows():
            results.append(explain_score(row))
        return "\n\n".join(results)

    # Filtered search
    top = get_top_players(
        players_df,
        position=position if position else None,
        min_form=min_form,
        max_price=max_price,
        team=team if team else None,
        limit=limit,
    )

    if top.empty:
        return "No players match your filters. Try relaxing the criteria."

    results = []
    for _, row in top.iterrows():
        conf = row.get("confidence", "?")
        results.append(
            f"{row['name']} ({row['team']}, {row['position']}) — "
            f"Score: {row['score']:.1f} [{conf}] | "
            f"£{row['price']}m | Form: {row['form']} | "
            f"Pts: {row['total_points']} | xGI: {row['expected_goal_involvements']:.1f}"
        )

    header = f"Top {len(results)} players"
    if position:
        header += f" ({position})"
    if team:
        header += f" from {team}"
    header += f" (sorted by composite score):\n"

    return header + "\n".join(results)


# ── Tool 2: Get Fixtures ────────────────────────────────────────────

@tool
def get_fixtures(team: str, num_gameweeks: int = 3) -> str:
    """Check upcoming fixture difficulty for a specific team.

    FPL rates fixtures 1-5: 1-2 = easy (green), 3 = medium, 4-5 = hard (red).
    Use this to evaluate whether a player has favorable games coming up.

    Args:
        team: Team name (e.g. "Arsenal", "Liverpool", "Man City").
        num_gameweeks: Number of upcoming gameweeks to check (default 3).
    """
    data, error = _load_data_or_error()
    if error:
        return error
    teams = data["teams"]
    fixtures_df = data["fixtures"]

    # Find team ID
    team_id = None
    for tid, tname in teams.items():
        if team.lower() in tname.lower():
            team_id = tid
            team_name = tname
            break

    if team_id is None:
        available = ", ".join(sorted(teams.values()))
        return f"Team '{team}' not found. Available teams: {available}"

    fixtures = get_team_fixtures(fixtures_df, team_id, num_gameweeks)

    if not fixtures:
        return f"No upcoming fixtures found for {team_name}."

    avg_difficulty = sum(f["difficulty"] for f in fixtures) / len(fixtures)
    ease_label = "Easy" if avg_difficulty <= 2.5 else "Hard" if avg_difficulty >= 4.0 else "Medium"

    lines = [f"Upcoming fixtures for {team_name} (avg difficulty: {avg_difficulty:.1f} — {ease_label}):"]
    for f in fixtures:
        diff = f["difficulty"]
        emoji = "🟢" if diff <= 2 else "🟡" if diff == 3 else "🔴"
        lines.append(f"  {emoji} GW{f['gameweek']}: vs {f['opponent']} ({f['venue']}) — FDR {diff}")

    return "\n".join(lines)


# ── Tool 3: Check Availability ──────────────────────────────────────

@tool
def check_availability(player_name: str) -> str:
    """Check if a specific player is available, injured, doubtful, or suspended.

    Returns the player's current status, chance of playing, and any news.
    Use this before including a player in your squad to avoid picking injured players.

    Args:
        player_name: The player's name (e.g. "Salah", "Haaland", "Saka").
    """
    data, error = _load_data_or_error()
    if error:
        return error
    players_df = data["players"]

    match = players_df[
        players_df["name"].str.lower().str.contains(player_name.lower()) |
        players_df["full_name"].str.lower().str.contains(player_name.lower())
    ]

    if match.empty:
        return f"Player '{player_name}' not found."

    results = []
    for _, player in match.head(3).iterrows():
        status_map = {
            "a": "✅ Available",
            "d": "⚠️ Doubtful",
            "i": "❌ Injured",
            "s": "🚫 Suspended",
            "u": "❌ Unavailable",
        }
        status = status_map.get(player["status"], "Unknown")
        chance = player["chance_of_playing_next"]
        chance_str = f"{chance}%" if chance is not None else "Unknown"
        news = player["news"] if player["news"] else "No news"

        results.append(
            f"{player['full_name']} ({player['team']}, {player['position']})\n"
            f"  Status: {status}\n"
            f"  Chance of playing next GW: {chance_str}\n"
            f"  News: {news}\n"
            f"  Recent form: {player['form']} | Minutes: {player['minutes']}"
        )

    return "\n\n".join(results)


# ── Tool 4: Build Squad ─────────────────────────────────────────────

@tool
def build_squad(
    locked_players: str = "",
    exclude_players: str = "",
    budget: float = 100.0,
    save_result: bool = False,
) -> str:
    """Build an optimal 15-player FPL squad within all constraints.

    The optimizer uses our scoring engine to select the best valid squad:
    - Budget: £100.0m (or custom)
    - Positions: 2 GKP, 5 DEF, 5 MID, 3 FWD
    - Max 3 players from any team
    - Picks captain and vice-captain
    - Selects starting XI with valid formation

    You can lock specific players (they'll be included) or exclude players.

    Args:
        locked_players: Comma-separated player names to INCLUDE (e.g. "Salah, Haaland").
        exclude_players: Comma-separated player names to EXCLUDE.
        budget: Total budget in millions (default 100.0).
        save_result: If true, saves the squad to memory for future reference.
    """
    data, error = _load_data_or_error()
    if error:
        return error
    players_df = data["players"]

    # Apply user preferences
    prefs = load_preferences()
    locked = [p.strip() for p in locked_players.split(",") if p.strip()]
    excluded = [p.strip() for p in exclude_players.split(",") if p.strip()]

    # Add preference-based locked/excluded players
    locked.extend(prefs.get("must_include_players", []))
    excluded.extend(prefs.get("never_pick_players", []))

    # Remove duplicates
    locked = list(set(locked))
    excluded = list(set(excluded))

    try:
        result = optimize_squad(
            players_df=players_df,
            locked_players=locked if locked else None,
            budget=budget,
            exclude_players=excluded if excluded else None,
        )
    except Exception as e:
        return f"Squad builder failed unexpectedly: {e}"

    if save_result:
        squad_names = [p["name"] for p in result.get("squad", [])]
        update_squad(squad_names, result.get("remaining_budget", 0.0))

    summary = format_squad_summary(result)

    # Add confidence overview
    confidences = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for p in result.get("starting_xi", []):
        conf = p.get("confidence", "MEDIUM")
        confidences[conf] = confidences.get(conf, 0) + 1

    summary += f"\n\nConfidence: {confidences['HIGH']} HIGH, {confidences['MEDIUM']} MEDIUM, {confidences['LOW']} LOW picks"

    if result["issues"]:
        summary += f"\n⚠️ Issues: {'; '.join(result['issues'])}"

    return summary


# ── Tool 5: Manage Preferences & Squad State ────────────────────────

@tool
def manage_memory(
    action: str = "view",
    preference_key: str = "",
    preference_value: str = "",
) -> str:
    """View or update user preferences and current squad state.

    Use 'view' to see current preferences and saved squad.
    Use 'update' to change a preference (e.g. favorite team, risk tolerance).

    Args:
        action: One of "view", "update", "clear_squad".
        preference_key: Key to update — one of: favorite_team, must_include_players,
            never_pick_players, prefer_budget_picks, captain_preference, risk_tolerance, notes.
        preference_value: New value for the preference. For list fields, comma-separate values.
    """
    if action == "view":
        prefs = get_preferences_summary()
        squad = get_squad_state_summary()
        return f"{prefs}\n\n{squad}"

    elif action == "update":
        if not preference_key:
            return "Specify a preference_key to update. Options: favorite_team, must_include_players, never_pick_players, prefer_budget_picks, captain_preference, risk_tolerance, notes"

        # Handle list fields
        list_fields = ["must_include_players", "never_pick_players", "notes"]
        if preference_key in list_fields:
            value = [v.strip() for v in preference_value.split(",") if v.strip()]
        elif preference_key == "prefer_budget_picks":
            value = preference_value.lower() in ("true", "yes", "1")
        else:
            value = preference_value

        return update_preference(preference_key, value)

    elif action == "clear_squad":
        save_squad_state({
            "current_squad": [],
            "budget_remaining": 0.0,
            "free_transfers": 1,
            "chips_available": {"wildcard": True, "free_hit": True, "bench_boost": True, "triple_captain": True},
            "gameweek_history": [],
        })
        return "Squad state cleared."

    return f"Unknown action: {action}. Use 'view', 'update', or 'clear_squad'."


# ── Tool 6: Get Current Gameweek Info ────────────────────────────────

@tool
def get_gameweek_info() -> str:
    """Get information about the current/next FPL gameweek.

    Returns the gameweek number, deadline, and whether it's currently active.
    Use this to understand the context of the current FPL week.
    """
    data, error = _load_data_or_error()
    if error:
        return error
    gw = data["current_gw"]

    is_stale = data.get("is_stale", False)
    stale_warning = " ⚠️ (using cached data — FPL API may be down)" if is_stale else ""

    return (
        f"Gameweek {gw['id']}{stale_warning}\n"
        f"  Name: {gw.get('name', 'N/A')}\n"
        f"  Deadline: {gw.get('deadline_time', 'N/A')}\n"
        f"  Current: {gw.get('is_current', False)}\n"
        f"  Next: {gw.get('is_next', False)}\n"
        f"  Average score: {gw.get('average_entry_score', 'N/A')}\n"
        f"  Highest score: {gw.get('highest_score', 'N/A')}"
    )


# ── All tools for the agent ──────────────────────────────────────────

ALL_TOOLS = [
    get_player_stats,
    get_fixtures,
    check_availability,
    build_squad,
    manage_memory,
    get_gameweek_info,
]
