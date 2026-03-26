"""
eval/evaluator.py — Evaluation Framework

Measures the quality of agent decisions by comparing against:
1. Actual FPL optimal team (hindsight — best possible team after results are known)
2. Form-only baseline (just picks highest form players)
3. Popularity baseline (picks most-selected players)
4. Random valid team

Metrics:
- Points scored vs each baseline
- Efficiency ratio: agent_points / optimal_points
- Win rate: % of gameweeks agent beats each baseline

Usage:
    python -m eval.evaluator --gameweeks 25-30
"""

import json
import os
import random
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

from core.fpl_data import (
    fetch_bootstrap, build_players_dataframe, SQUAD_RULES, POSITION_MAP
)
from core.scoring import score_all_players
from core.optimizer import optimize_squad

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def get_gameweek_points(bootstrap: dict, gameweek_id: int) -> dict:
    """
    Get actual points scored by each player in a specific gameweek.
    Uses the live endpoint for historical gameweeks.
    """
    url = f"https://fantasy.premierleague.com/api/event/{gameweek_id}/live/"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        points = {}
        for element in data["elements"]:
            points[element["id"]] = element["stats"]["total_points"]
        return points
    except Exception as e:
        print(f"⚠️ Could not fetch GW{gameweek_id} results: {e}")
        return {}


def calculate_team_points(squad: list, points_map: dict, captain_name: str) -> int:
    """
    Calculate total points for a starting XI.
    Captain gets double points.
    """
    total = 0
    for player in squad:
        player_id = player.get("id")
        pts = points_map.get(player_id, 0)
        if player.get("name") == captain_name:
            pts *= 2  # Captain double points
        total += pts
    return total


def build_form_baseline(players_df: pd.DataFrame) -> dict:
    """
    Baseline 1: Just pick highest form players per position.
    No fixture analysis, no injury checks, no xG.
    """
    # Override scores with just form
    df = players_df.copy()
    df["score"] = df["form"]

    result = optimize_squad(df)
    return result


def build_popularity_baseline(players_df: pd.DataFrame) -> dict:
    """
    Baseline 2: Pick the most-selected players (crowd wisdom).
    """
    df = players_df.copy()
    df["score"] = df["selected_by_percent"]

    result = optimize_squad(df)
    return result


def build_random_baseline(players_df: pd.DataFrame) -> dict:
    """
    Baseline 3: Random valid team (within constraints).
    """
    df = players_df.copy()
    df["score"] = [random.random() * 10 for _ in range(len(df))]

    result = optimize_squad(df)
    return result


def evaluate_gameweek(gameweek_id: int, fixture_scores: dict = None) -> dict:
    """
    Evaluate agent vs baselines for a single gameweek.

    Returns dict with points for each method and efficiency metrics.
    """
    print(f"\n📊 Evaluating Gameweek {gameweek_id}...")

    # Fetch data
    bootstrap = fetch_bootstrap()
    if not bootstrap:
        return {"error": "Could not fetch FPL data"}

    players_df = build_players_dataframe(bootstrap)

    # Get actual results
    actual_points = get_gameweek_points(bootstrap, gameweek_id)
    if not actual_points:
        return {"error": f"No results available for GW{gameweek_id}"}

    # Calculate theoretical optimal (hindsight)
    df_with_actual = players_df.copy()
    df_with_actual["score"] = df_with_actual["id"].map(lambda x: actual_points.get(x, 0))
    optimal = optimize_squad(df_with_actual)
    optimal_points = calculate_team_points(
        optimal["starting_xi"], actual_points, optimal["captain"]["name"]
    )

    # Score players with our engine (if fixture_scores provided)
    if fixture_scores:
        scored_df = score_all_players(players_df, fixture_scores)
    else:
        scored_df = players_df.copy()
        scored_df["score"] = scored_df["form"] * 3 + scored_df["points_per_million"] * 2

    # Agent's team
    agent_result = optimize_squad(scored_df)
    agent_points = calculate_team_points(
        agent_result["starting_xi"], actual_points, agent_result["captain"]["name"]
    )

    # Baselines
    form_result = build_form_baseline(players_df)
    form_points = calculate_team_points(
        form_result["starting_xi"], actual_points, form_result["captain"]["name"]
    )

    popularity_result = build_popularity_baseline(players_df)
    popularity_points = calculate_team_points(
        popularity_result["starting_xi"], actual_points, popularity_result["captain"]["name"]
    )

    random_points_list = []
    for _ in range(5):  # Average of 5 random teams
        rand_result = build_random_baseline(players_df)
        rand_pts = calculate_team_points(
            rand_result["starting_xi"], actual_points, rand_result["captain"]["name"]
        )
        random_points_list.append(rand_pts)
    random_points = sum(random_points_list) // len(random_points_list)

    # Efficiency
    efficiency = (agent_points / optimal_points * 100) if optimal_points > 0 else 0

    result = {
        "gameweek": gameweek_id,
        "agent_points": agent_points,
        "optimal_points": optimal_points,
        "form_baseline_points": form_points,
        "popularity_baseline_points": popularity_points,
        "random_baseline_points": random_points,
        "efficiency": round(efficiency, 1),
        "beats_form": agent_points > form_points,
        "beats_popularity": agent_points > popularity_points,
        "beats_random": agent_points > random_points,
        "agent_captain": agent_result["captain"]["name"],
        "timestamp": datetime.now().isoformat(),
    }

    print(f"   Agent: {agent_points} pts | Optimal: {optimal_points} pts | "
          f"Efficiency: {efficiency:.1f}%")
    print(f"   Form baseline: {form_points} | Popularity: {popularity_points} | "
          f"Random: {random_points}")

    return result


def evaluate_multiple_gameweeks(start_gw: int, end_gw: int) -> dict:
    """
    Run evaluation across multiple gameweeks and compute aggregate metrics.
    """
    print(f"📊 Evaluating GW{start_gw} to GW{end_gw}...")
    results = []

    for gw in range(start_gw, end_gw + 1):
        gw_result = evaluate_gameweek(gw)
        if "error" not in gw_result:
            results.append(gw_result)

    if not results:
        return {"error": "No valid gameweek results"}

    # Aggregate metrics
    n = len(results)
    summary = {
        "gameweeks_evaluated": n,
        "range": f"GW{start_gw}-GW{end_gw}",
        "agent_avg_points": round(sum(r["agent_points"] for r in results) / n, 1),
        "optimal_avg_points": round(sum(r["optimal_points"] for r in results) / n, 1),
        "form_avg_points": round(sum(r["form_baseline_points"] for r in results) / n, 1),
        "popularity_avg_points": round(sum(r["popularity_baseline_points"] for r in results) / n, 1),
        "random_avg_points": round(sum(r["random_baseline_points"] for r in results) / n, 1),
        "avg_efficiency": round(sum(r["efficiency"] for r in results) / n, 1),
        "win_rate_vs_form": round(sum(1 for r in results if r["beats_form"]) / n * 100, 1),
        "win_rate_vs_popularity": round(sum(1 for r in results if r["beats_popularity"]) / n * 100, 1),
        "win_rate_vs_random": round(sum(1 for r in results if r["beats_random"]) / n * 100, 1),
        "per_gameweek": results,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"eval_gw{start_gw}_to_gw{end_gw}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to {filepath}")

    return summary


def print_eval_report(summary: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 65)
    print(f"📊 EVALUATION REPORT — {summary['range']}")
    print("=" * 65)
    print(f"\n{'Method':<25} {'Avg Pts/GW':<15} {'vs Optimal'}")
    print("-" * 55)

    optimal = summary["optimal_avg_points"]
    rows = [
        ("Random baseline", summary["random_avg_points"]),
        ("Form-only baseline", summary["form_avg_points"]),
        ("Popularity baseline", summary["popularity_avg_points"]),
        ("FantasyEdge Agent", summary["agent_avg_points"]),
        ("Theoretical optimal", optimal),
    ]

    for name, pts in rows:
        pct = round(pts / optimal * 100, 1) if optimal > 0 else 0
        marker = " ⭐" if name == "FantasyEdge Agent" else ""
        print(f"  {name:<23} {pts:<15} {pct}%{marker}")

    print(f"\n  Agent efficiency: {summary['avg_efficiency']}%")
    print(f"  Win rate vs form baseline: {summary['win_rate_vs_form']}%")
    print(f"  Win rate vs popularity: {summary['win_rate_vs_popularity']}%")
    print(f"  Win rate vs random: {summary['win_rate_vs_random']}%")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    # Default: evaluate last 5 completed gameweeks
    bootstrap = fetch_bootstrap()
    if bootstrap:
        current_gw = None
        for event in bootstrap["events"]:
            if event["is_current"]:
                current_gw = event["id"]
                break

        if current_gw:
            start = max(1, current_gw - 5)
            end = current_gw - 1  # Only completed gameweeks
            summary = evaluate_multiple_gameweeks(start, end)
            if "error" not in summary:
                print_eval_report(summary)
