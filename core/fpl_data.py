"""
core/fpl_data.py — FPL Data Layer

Fetches and processes all data from the Fantasy Premier League API.
Includes caching for tool reliability — if the API is down, falls back to last good fetch.

Key endpoints:
- bootstrap-static: All players, teams, gameweeks
- fixtures: All matches with difficulty ratings
- element-summary: Individual player history
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

# Fix emoji printing on Windows (cp1252 terminal)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

BASE_URL = "https://fantasy.premierleague.com/api"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "fpl_cache.json")
CACHE_MAX_AGE = 3600  # 1 hour — data is stale after this

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_IDS = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

# FPL squad constraints — the rules every valid team must follow
SQUAD_RULES = {
    "total_players": 15,
    "budget": 1000,  # FPL uses 10x internally (100.0m = 1000)
    "max_per_team": 3,
    "positions": {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    "starting_xi": 11,
    "starting_min": {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    "starting_max": {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
}


# ── Caching Layer (Tool Reliability) ─────────────────────────────────

def _save_cache(data: dict):
    """Save fetched data to disk for fallback."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {"timestamp": time.time(), "data": data}
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _load_cache() -> Optional[dict]:
    """Load cached data if available."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        age_minutes = (time.time() - cache["timestamp"]) / 60
        print(f"📦 Using cached data ({age_minutes:.0f} minutes old)")
        data = cache["data"]
        # Reconstruct DataFrames from cached lists
        if isinstance(data.get("players"), list):
            data["players"] = pd.DataFrame(data["players"])
        if isinstance(data.get("fixtures"), list):
            data["fixtures"] = pd.DataFrame(data["fixtures"])
        return data
    except Exception:
        return None


def _is_cache_fresh() -> bool:
    """Check if cache exists and is recent enough."""
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        return (time.time() - cache["timestamp"]) < CACHE_MAX_AGE
    except Exception:
        return False


# ── API Fetchers ─────────────────────────────────────────────────────

def fetch_bootstrap(timeout: int = 30) -> dict:
    """Fetch the main FPL data dump with timeout handling."""
    url = f"{BASE_URL}/bootstrap-static/"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("⚠️ FPL API timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ FPL API error: {e}")
        return None


def fetch_fixtures(timeout: int = 30) -> Optional[list]:
    """Fetch all fixtures for the season."""
    url = f"{BASE_URL}/fixtures/"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ Fixtures fetch failed: {e}")
        return None


def fetch_player_detail(player_id: int, timeout: int = 15) -> Optional[dict]:
    """Fetch detailed history for a specific player."""
    url = f"{BASE_URL}/element-summary/{player_id}/"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# ── Data Processing ──────────────────────────────────────────────────

def get_current_gameweek(bootstrap: dict) -> dict:
    """Find the current or next gameweek."""
    events = bootstrap["events"]
    for event in events:
        if event["is_current"]:
            return event
    for event in events:
        if event["is_next"]:
            return event
    return events[-1]


def build_players_dataframe(bootstrap: dict) -> pd.DataFrame:
    """Transform raw API data into a clean player DataFrame."""
    elements = bootstrap["elements"]
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

    players = []
    for p in elements:
        players.append({
            "id": p["id"],
            "name": p["web_name"],
            "full_name": f"{p['first_name']} {p['second_name']}",
            "team": teams.get(p["team"], "Unknown"),
            "team_id": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "UNK"),
            "price": p["now_cost"] / 10,
            "total_points": p["total_points"],
            "points_per_game": float(p["points_per_game"]),
            "form": float(p["form"]),
            "selected_by_percent": float(p["selected_by_percent"]),
            "minutes": p["minutes"],
            "goals_scored": p["goals_scored"],
            "assists": p["assists"],
            "clean_sheets": p["clean_sheets"],
            "goals_conceded": p["goals_conceded"],
            "bonus": p["bonus"],
            "bps": p["bps"],
            "influence": float(p["influence"]),
            "creativity": float(p["creativity"]),
            "threat": float(p["threat"]),
            "ict_index": float(p["ict_index"]),
            "expected_goals": float(p.get("expected_goals", 0)),
            "expected_assists": float(p.get("expected_assists", 0)),
            "expected_goal_involvements": float(p.get("expected_goal_involvements", 0)),
            "expected_goals_conceded": float(p.get("expected_goals_conceded", 0)),
            "status": p["status"],
            "chance_of_playing_next": p.get("chance_of_playing_next_round"),
            "news": p.get("news", ""),
            "transfers_in_event": p["transfers_in_event"],
            "transfers_out_event": p["transfers_out_event"],
            "value_season": float(p.get("value_season", 0)),
        })

    df = pd.DataFrame(players)
    df["points_per_million"] = (df["total_points"] / df["price"].replace(0, 1)).round(1)
    df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]
    df["available"] = df["status"].isin(["a", "d"])
    df["fully_fit"] = df["status"] == "a"

    return df


def build_fixtures_dataframe(fixtures: list, bootstrap: dict) -> pd.DataFrame:
    """Build DataFrame of upcoming fixtures with difficulty ratings."""
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw = get_current_gameweek(bootstrap)

    upcoming = []
    for f in fixtures:
        if f["event"] and f["event"] >= current_gw["id"]:
            upcoming.append({
                "gameweek": f["event"],
                "home_team": teams.get(f["team_h"], "Unknown"),
                "home_team_id": f["team_h"],
                "away_team": teams.get(f["team_a"], "Unknown"),
                "away_team_id": f["team_a"],
                "home_difficulty": f["team_h_difficulty"],
                "away_difficulty": f["team_a_difficulty"],
                "finished": f["finished"],
            })

    return pd.DataFrame(upcoming)


def get_team_fixtures(fixtures_df: pd.DataFrame, team_id: int, num_gw: int = 5) -> list:
    """Get next N fixtures for a team."""
    home = fixtures_df[fixtures_df["home_team_id"] == team_id][["gameweek", "away_team", "home_difficulty"]].copy()
    home.columns = ["gameweek", "opponent", "difficulty"]
    home["venue"] = "H"

    away = fixtures_df[fixtures_df["away_team_id"] == team_id][["gameweek", "home_team", "away_difficulty"]].copy()
    away.columns = ["gameweek", "opponent", "difficulty"]
    away["venue"] = "A"

    return pd.concat([home, away]).sort_values("gameweek").head(num_gw).to_dict("records")


# ── Main Entry Point ─────────────────────────────────────────────────

def get_all_data(force_refresh: bool = False) -> dict:
    """
    Fetch and process all FPL data.
    Uses cache if fresh enough, falls back to cache if API fails.
    This is the tool reliability layer — the agent never crashes due to API issues.
    """
    # Try cache first if fresh
    if not force_refresh and _is_cache_fresh():
        cached = _load_cache()
        if cached:
            return cached

    # Fetch fresh data
    print("📡 Fetching fresh FPL data...")
    bootstrap = fetch_bootstrap()
    fixtures_raw = fetch_fixtures()

    # If API failed, fall back to cache
    if bootstrap is None or fixtures_raw is None:
        print("⚠️ API failed — falling back to cached data")
        cached = _load_cache()
        if cached:
            cached["is_stale"] = True
            return cached
        raise RuntimeError("FPL API is down and no cached data available")

    # Process data
    players_df = build_players_dataframe(bootstrap)
    fixtures_df = build_fixtures_dataframe(fixtures_raw, bootstrap)
    current_gw = get_current_gameweek(bootstrap)
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

    # Calculate fixture difficulty per team
    fixture_scores = {}
    for team_id in players_df["team_id"].unique():
        team_fix = get_team_fixtures(fixtures_df, team_id, num_gw=3)
        avg_diff = sum(f["difficulty"] for f in team_fix) / len(team_fix) if team_fix else 3
        fixture_scores[int(team_id)] = {
            "avg_difficulty": round(avg_diff, 1),
            "next_fixtures": team_fix,
        }

    players_df["fixture_difficulty"] = players_df["team_id"].map(
        lambda x: fixture_scores.get(int(x), {}).get("avg_difficulty", 3)
    )

    result = {
        "players": players_df,
        "fixtures": fixtures_df,
        "fixture_scores": fixture_scores,
        "current_gw": current_gw,
        "teams": teams,
        "is_stale": False,
        "fetched_at": datetime.now().isoformat(),
    }

    # Cache for reliability
    cache_data = {
        **result,
        "players": players_df.to_dict("records"),
        "fixtures": fixtures_df.to_dict("records"),
    }
    _save_cache(cache_data)

    print(f"✅ Data ready for GW{current_gw['id']} — {len(players_df)} players, {len(fixtures_df)} fixtures")
    return result
