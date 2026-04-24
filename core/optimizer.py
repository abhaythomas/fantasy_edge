"""
core/optimizer.py — Squad Optimizer (Constraint Solver)

Selects the optimal 15-player squad within FPL constraints.
This is ALGORITHMIC — the LLM never touches squad selection.

Constraints enforced:
1. Budget: £100.0m total
2. Positions: exactly 2 GKP, 5 DEF, 5 MID, 3 FWD
3. Max 3 players from any single team
4. Starting XI: 11 players with valid formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD)

Method: Greedy selection with constraint checking.
Each position is filled by taking the highest-scoring available player
that doesn't violate any constraint.

Locked players: The optimizer can be given players that MUST be in the squad
(e.g., "I already have Salah, build around him"). These are placed first,
and remaining budget/slots are filled optimally.
"""

import pandas as pd
from typing import Optional
from core.fpl_data import SQUAD_RULES


def optimize_squad(
    players_df: pd.DataFrame,
    locked_players: Optional[list] = None,
    budget: float = 100.0,
    exclude_players: Optional[list] = None,
) -> dict:
    """
    Select the optimal 15-player squad.

    Args:
        players_df: DataFrame with scored players (must have 'score' column)
        locked_players: list of player names that MUST be included
        budget: total budget in millions (default 100.0)
        exclude_players: list of player names to exclude

    Returns:
        dict with 'squad', 'starting_xi', 'bench', 'captain', 'vice_captain',
        'total_cost', 'remaining_budget', 'formation', 'valid', 'issues'
    """
    locked_players = locked_players or []
    exclude_players = exclude_players or []
    budget_units = int(budget * 10)  # Convert to FPL internal units

    required = SQUAD_RULES["positions"].copy()  # {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    max_per_team = SQUAD_RULES["max_per_team"]

    # Filter to available players with scores
    candidates = players_df[
        (players_df["available"]) &
        (players_df["minutes"] >= 90) &
        (~players_df["name"].isin(exclude_players))
    ].copy()

    squad = []
    team_count = {}
    spent = 0
    issues = []

    # ── Step 1: Place locked players first ────────────────────────────
    for name in locked_players:
        match = candidates[candidates["name"].str.lower() == name.lower()]
        if match.empty:
            match = candidates[candidates["full_name"].str.lower() == name.lower()]
        if match.empty:
            issues.append(f"Locked player '{name}' not found or unavailable")
            continue

        player = match.iloc[0].to_dict()
        pos = player["position"]
        team_id = player["team_id"]
        price_units = int(player["price"] * 10)

        if required.get(pos, 0) <= 0:
            issues.append(f"Can't lock {name} — no {pos} slots remaining")
            continue

        squad.append(player)
        team_count[team_id] = team_count.get(team_id, 0) + 1
        spent += price_units
        required[pos] -= 1

    # ── Step 2: Fill remaining slots greedily ─────────────────────────
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        slots_needed = required[pos]
        if slots_needed <= 0:
            continue

        pos_candidates = candidates[
            (candidates["position"] == pos) &
            (~candidates["name"].isin([p["name"] for p in squad]))
        ].sort_values("score", ascending=False)

        filled = 0
        for _, player in pos_candidates.iterrows():
            if filled >= slots_needed:
                break

            team_id = player["team_id"]
            price_units = int(player["price"] * 10)

            # Check team limit
            if team_count.get(team_id, 0) >= max_per_team:
                continue

            # Check budget — reserve minimum price for remaining unfilled slots
            remaining_slots = sum(required.values()) - len(squad) + sum(SQUAD_RULES["positions"].values()) - sum(required.values()) - filled
            # Rough reserve: £4.0m per remaining slot
            total_remaining_slots = 15 - len(squad) - filled
            reserve = max(0, (total_remaining_slots - 1)) * 40
            if spent + price_units > budget_units - reserve:
                continue

            squad.append(player.to_dict())
            team_count[team_id] = team_count.get(team_id, 0) + 1
            spent += price_units
            filled += 1

        if filled < slots_needed:
            issues.append(f"Could only fill {filled}/{slots_needed} {pos} slots")

    # ── Step 3: Select Starting XI ────────────────────────────────────
    starting_xi, bench = _select_starting_xi(squad)

    # ── Step 4: Pick Captain & Vice Captain ───────────────────────────
    outfield_starters = sorted(
        [p for p in starting_xi if p["position"] != "GKP"],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )
    captain = outfield_starters[0] if outfield_starters else starting_xi[0]
    vice_captain = outfield_starters[1] if len(outfield_starters) > 1 else captain

    # ── Calculate totals ──────────────────────────────────────────────
    total_cost = round(sum(p["price"] for p in squad), 1)
    remaining = round(budget - total_cost, 1)

    # Determine formation
    formation = _get_formation(starting_xi)

    # Validate
    valid = len(squad) == 15 and len(issues) == 0 and total_cost <= budget

    return {
        "squad": squad,
        "starting_xi": starting_xi,
        "bench": bench,
        "captain": captain,
        "vice_captain": vice_captain,
        "total_cost": total_cost,
        "remaining_budget": remaining,
        "formation": formation,
        "valid": valid,
        "issues": issues,
    }


def _select_starting_xi(squad: list) -> tuple:
    """
    Select the best starting XI from a 15-player squad.
    Must have exactly 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, total 11.
    """
    mins = SQUAD_RULES["starting_min"]
    maxs = SQUAD_RULES["starting_max"]

    # Start with best GKP
    gkps = sorted([p for p in squad if p["position"] == "GKP"],
                  key=lambda x: x.get("score", 0), reverse=True)
    starting = [gkps[0]] if gkps else []
    bench = [gkps[1]] if len(gkps) > 1 else []

    # Sort outfield by score
    outfield = sorted([p for p in squad if p["position"] != "GKP"],
                     key=lambda x: x.get("score", 0), reverse=True)

    pos_count = {"DEF": 0, "MID": 0, "FWD": 0}

    # First pass: fill minimums
    remaining = []
    for p in outfield:
        pos = p["position"]
        if pos_count[pos] < mins[pos] and len(starting) < 11:
            starting.append(p)
            pos_count[pos] += 1
        else:
            remaining.append(p)

    # Second pass: fill to 11 with best available (respecting max per position)
    for p in remaining:
        if len(starting) >= 11:
            bench.append(p)
            continue
        pos = p["position"]
        if pos_count[pos] < maxs[pos]:
            starting.append(p)
            pos_count[pos] += 1
        else:
            bench.append(p)

    return starting, bench


def _get_formation(starting_xi: list) -> str:
    """Get formation string like '3-5-2'."""
    pos_count = {"DEF": 0, "MID": 0, "FWD": 0}
    for p in starting_xi:
        if p["position"] in pos_count:
            pos_count[p["position"]] += 1
    return f"{pos_count['DEF']}-{pos_count['MID']}-{pos_count['FWD']}"


def format_squad_summary(result: dict) -> str:
    """
    Format the squad selection as a readable string.
    The agent uses this to present results to the user.
    """
    lines = []
    budget_limit = round(result.get("total_cost", 0) + result.get("remaining_budget", 0), 1)
    captain_name = result.get("captain", {}).get("name", "N/A")
    vice_name = result.get("vice_captain", {}).get("name", "N/A")

    lines.append(f"Formation: {result.get('formation', 'N/A')} | Budget: £{result.get('total_cost', 0)}m / "
                 f"£{budget_limit}m (£{result.get('remaining_budget', 0)}m remaining)")
    lines.append(f"Captain: {captain_name} | Vice: {vice_name}")
    lines.append("")

    # Starting XI by position
    lines.append("STARTING XI:")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_players = [p for p in result.get("starting_xi", []) if p["position"] == pos]
        for p in sorted(pos_players, key=lambda x: x.get("score", 0), reverse=True):
            captain_tag = " (C)" if p["name"] == captain_name else ""
            vice_tag = " (VC)" if p["name"] == vice_name else ""
            conf = p.get("confidence", "")
            lines.append(
                f"  {pos} | {p['name']}{captain_tag}{vice_tag} | {p['team']} | "
                f"£{p['price']}m | Score: {p.get('score', 0):.1f} | [{conf}]"
            )

    lines.append("\nBENCH:")
    for p in result.get("bench", []):
        lines.append(f"  {p['position']} | {p['name']} | {p['team']} | £{p['price']}m")

    if result["issues"]:
        lines.append(f"\n⚠️ Issues: {'; '.join(result['issues'])}")

    return "\n".join(lines)
