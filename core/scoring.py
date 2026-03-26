"""
core/scoring.py — Player Scoring Engine

This is the INTELLIGENCE of the system — a documented, testable, and explainable
scoring formula that evaluates every FPL player.

The LLM never scores players. This module does.
The LLM reads scores and EXPLAINS them.

Scoring Formula:
    player_score = (form × W_FORM) 
                 + (xGI × W_XGI) 
                 + (points_per_million × W_VALUE) 
                 + (fixture_ease × W_FIXTURE) 
                 + (availability × W_AVAILABILITY) 
                 + (momentum × W_MOMENTUM)

Confidence is calculated separately based on data quality and signal agreement.
"""

import pandas as pd
from typing import Optional

# ── Configurable Weights ─────────────────────────────────────────────
# These can be tuned. Each weight controls how much that factor matters.
# In evaluation, we can grid-search these to find optimal values.

WEIGHTS = {
    "form": 3.0,           # Recent form — strongest short-term predictor
    "xgi": 2.5,            # Expected goal involvements — measures chance quality
    "value": 1.5,          # Points per million — budget efficiency
    "fixture": 2.0,        # Fixture ease — upcoming game difficulty
    "availability": 5.0,   # Availability penalty — injured = 0 points
    "momentum": 0.5,       # Net transfers — crowd wisdom signal
    "clean_sheet": 1.0,    # Clean sheet potential (defenders/GKPs only)
}


def score_player(player: dict, fixture_difficulty: float = 3.0) -> dict:
    """
    Score a single player. Returns a dict with the composite score,
    individual component scores, and confidence level.

    Args:
        player: dict with player stats (from players DataFrame row)
        fixture_difficulty: avg FDR for next 3 GWs (1=easy, 5=hard)

    Returns:
        dict with 'score', 'components', 'confidence', 'confidence_reasons'
    """
    components = {}

    # 1. Form — FPL's 30-day average points
    form = float(player.get("form", 0))
    components["form"] = round(form * WEIGHTS["form"], 2)

    # 2. Expected Goal Involvements (xGI) — chance quality, not luck
    xgi = float(player.get("expected_goal_involvements", 0))
    # Normalize per 90 minutes for fairness
    minutes = max(player.get("minutes", 1), 1)
    xgi_per_90 = (xgi / minutes) * 90 if minutes > 0 else 0
    components["xgi"] = round(xgi_per_90 * WEIGHTS["xgi"] * 10, 2)  # Scale up for weighting

    # 3. Value — points per million spent
    ppm = float(player.get("points_per_million", 0))
    components["value"] = round(min(ppm, 25) / 5 * WEIGHTS["value"], 2)  # Cap and normalize

    # 4. Fixture ease — inverted difficulty (5=easiest, 1=hardest after inversion)
    fixture_ease = 5 - fixture_difficulty
    components["fixture"] = round(fixture_ease * WEIGHTS["fixture"], 2)

    # 5. Availability
    status = player.get("status", "a")
    chance = player.get("chance_of_playing_next")

    if status == "a":
        avail_score = 1.0
    elif status == "d":
        avail_score = (chance / 100) if chance is not None else 0.5
    else:  # injured, suspended, unavailable
        avail_score = 0.0

    components["availability"] = round(avail_score * WEIGHTS["availability"], 2)

    # 6. Momentum — net transfers (crowd wisdom)
    net_transfers = player.get("net_transfers", 0)
    # Normalize: cap at ±100k, scale to 0-1
    momentum = max(min(net_transfers / 100000, 1), -1)
    momentum_score = (momentum + 1) / 2  # Shift to 0-1 range
    components["momentum"] = round(momentum_score * WEIGHTS["momentum"], 2)

    # 7. Clean sheet potential (only for GKP and DEF)
    position = player.get("position", "")
    if position in ("GKP", "DEF"):
        cs_potential = fixture_ease / 4  # Easy fixture = higher CS chance
        components["clean_sheet"] = round(cs_potential * WEIGHTS["clean_sheet"], 2)
    else:
        components["clean_sheet"] = 0.0

    # ── Composite Score ──────────────────────────────────────────────
    total_score = sum(components.values())

    # ── Confidence Calculation ───────────────────────────────────────
    confidence, reasons = _calculate_confidence(player, components, fixture_difficulty)

    return {
        "score": round(total_score, 2),
        "components": components,
        "confidence": confidence,
        "confidence_reasons": reasons,
    }


def _calculate_confidence(player: dict, components: dict, fixture_difficulty: float) -> tuple:
    """
    Calculate confidence level (HIGH / MEDIUM / LOW) based on:
    - Data quality (enough minutes played?)
    - Signal agreement (do form, xG, and fixtures all agree?)
    - Availability certainty
    """
    reasons = []
    confidence_score = 0

    # Signal 1: Enough minutes for reliable stats?
    minutes = player.get("minutes", 0)
    if minutes >= 900:  # ~10 full games
        confidence_score += 2
        reasons.append("Strong sample size (900+ minutes)")
    elif minutes >= 450:
        confidence_score += 1
        reasons.append("Moderate sample size")
    else:
        reasons.append("Low minutes — stats may be unreliable")

    # Signal 2: Form and xG agree? (not just lucky or unlucky)
    form = float(player.get("form", 0))
    xgi = float(player.get("expected_goal_involvements", 0))
    if form > 5 and xgi > 2:
        confidence_score += 2
        reasons.append("Form backed by strong underlying stats (xGI)")
    elif form > 5 and xgi < 1:
        reasons.append("Good form but low xGI — could be unsustainable")
    elif form < 3 and xgi > 3:
        confidence_score += 1
        reasons.append("Unlucky — xGI suggests better returns ahead")

    # Signal 3: Fixture clarity
    if fixture_difficulty <= 2.5:
        confidence_score += 1
        reasons.append("Favorable fixtures")
    elif fixture_difficulty >= 4.0:
        reasons.append("Tough fixtures — higher variance expected")

    # Signal 4: Availability certainty
    status = player.get("status", "a")
    if status == "a":
        confidence_score += 1
        reasons.append("Fully fit")
    elif status == "d":
        reasons.append(f"Doubtful — {player.get('news', 'no details')}")
    else:
        reasons.append(f"Unavailable — {player.get('news', 'no details')}")

    # Map to level
    if confidence_score >= 5:
        level = "HIGH"
    elif confidence_score >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, reasons


def score_all_players(players_df: pd.DataFrame, fixture_scores: dict) -> pd.DataFrame:
    """
    Score all players and add score columns to the DataFrame.
    Returns the DataFrame with added columns: score, confidence, confidence_reasons, components.
    """
    results = []
    for _, row in players_df.iterrows():
        player_dict = row.to_dict()
        fix_diff = fixture_scores.get(row["team_id"], {}).get("avg_difficulty", 3.0)
        result = score_player(player_dict, fix_diff)
        results.append(result)

    players_df = players_df.copy()
    players_df["score"] = [r["score"] for r in results]
    players_df["confidence"] = [r["confidence"] for r in results]
    players_df["confidence_reasons"] = [r["confidence_reasons"] for r in results]
    players_df["score_components"] = [r["components"] for r in results]

    return players_df.sort_values("score", ascending=False)


def get_top_players(players_df: pd.DataFrame, position: Optional[str] = None,
                    min_form: float = 0, max_price: float = 20.0,
                    team: Optional[str] = None, available_only: bool = True,
                    limit: int = 10) -> pd.DataFrame:
    """
    Get top-scored players with optional filters.
    This is what the agent's tool will call.
    """
    df = players_df.copy()

    if available_only:
        df = df[df["available"]]
    if position:
        df = df[df["position"] == position.upper()]
    if min_form > 0:
        df = df[df["form"] >= min_form]
    if max_price < 20:
        df = df[df["price"] <= max_price]
    if team:
        df = df[df["team"].str.lower().str.contains(team.lower())]

    # Filter out players with too few minutes (unreliable)
    df = df[df["minutes"] >= 90]

    return df.nlargest(limit, "score")


def explain_score(player_row: pd.Series) -> str:
    """
    Generate a human-readable explanation of a player's score.
    The agent can use this to explain picks to the user.
    """
    components = player_row.get("score_components", {})
    confidence = player_row.get("confidence", "MEDIUM")
    reasons = player_row.get("confidence_reasons", [])

    explanation = (
        f"{player_row['name']} ({player_row['team']}, {player_row['position']}) — "
        f"Score: {player_row['score']:.1f} [{confidence} confidence]\n"
        f"  Price: £{player_row['price']}m | Form: {player_row['form']} | "
        f"Total Pts: {player_row['total_points']}\n"
        f"  Score breakdown: "
        f"Form={components.get('form', 0):.1f}, "
        f"xGI={components.get('xgi', 0):.1f}, "
        f"Value={components.get('value', 0):.1f}, "
        f"Fixture={components.get('fixture', 0):.1f}, "
        f"Avail={components.get('availability', 0):.1f}\n"
        f"  Confidence: {'; '.join(reasons)}"
    )
    return explanation
