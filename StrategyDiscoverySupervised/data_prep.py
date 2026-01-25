#!/usr/bin/env python3
# Simple tool to compute per-civilization winrates from a .jsonl of collected game records.
# Example Call:
# python DataPreperation/data_prep.py collected_games_with_summary_v2_2026-01-21.jsonl
import argparse
import glob
import json
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import csv
import math
from collections import defaultdict
from typing import Dict


def process_file(path: str, stats: Dict[str, Dict[str, int]], h2h) -> None:
    """
    Process a single file containing a .jsonl of game records.

    Skips games shorter than 2 minutes (120s) when duration is available.

    Updates per-civilization stats (games, wins) and head-to-head stats (games, wins) for each game.

    :param path: Path to a .jsonl file containing game records.
    :param stats: A dictionary of per-civilization stats to update.
    :param h2h: A dictionary of head-to-head stats to update.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] {path}:{lineno} JSON decode failed; skipping")
                continue

            # Skip games shorter than 2 minutes (120s) when duration is available
            dur = _game_duration_seconds(obj)
            if dur is not None and dur < 120:
                print(f"[INFO] Skipping short game ({dur}s) in {path}:{lineno}")
                continue


            game = obj.get("game") or obj  # tolerate either top-level game or whole object
            teams = game.get("teams") or game.get("players") or []
            # Normalize teams -> flat list of player entries
            
            leaderboard = game.get('leaderboard')
            if leaderboard != "rm_solo":
                print(f"[INFO] Skipping not 1v1 {leaderboard}")
                continue
            
            entries = []
            if isinstance(teams, list) and teams and all(isinstance(t, list) for t in teams):
                # teams is list-of-teams, each team is list of player entries
                for team in teams:
                    entries.extend(team)
            elif isinstance(teams, list):
                entries = teams
            elif isinstance(teams, dict):
                # dict keyed by something -> take values
                for v in teams.values():
                    if isinstance(v, list):
                        entries.extend(v)
                    elif isinstance(v, dict):
                        entries.append(v)

            # gather normalized player infos for this game (for per-civ and head-to-head)
            players_info = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                player = entry.get("player") if "player" in entry else entry
                if not isinstance(player, dict):
                    continue
                civ = player.get("civilization") or player.get("civilisation")
                if not civ:
                    continue
                civ = str(civ).lower()
                result = (player.get("result") or "").strip().lower()
                won = result in ("win", "won", "victory")
                # update per-civ totals
                stats[civ]["games"] += 1
                if won:
                    stats[civ]["wins"] += 1
                # collect for head-to-head
                players_info.append({"civ": civ, "won": won})

            # head-to-head: count each winner-vs-loser pair once per match (works for 1v1 and team games)
            winner_civs = {p["civ"] for p in players_info if p["won"]}
            loser_civs = {p["civ"] for p in players_info if not p["won"]}
            if winner_civs and loser_civs:
                for w in winner_civs:
                    for l in loser_civs:
                        # increment winner -> loser
                        h2h[w][l]["games"] += 1
                        h2h[w][l]["wins"] += 1
                        # add opposite direction (loser -> winner) as a played game (no win)
                        h2h[l][w]["games"] += 1

def h2h_to_dataframe(h2h):
    """
    Convert a head-to-head dictionary to a pandas DataFrame.

    The DataFrame contains the winrates of each civilization against others.
    The index and columns of the DataFrame are the civilization names.

    :param h2h: A dictionary of head-to-head game records.
    :return: A pandas DataFrame of winrates between civilizations.
    """
    if pd is None:
        raise RuntimeError("pandas is required for DataFrame export")
    civs = sorted(h2h.keys())
    mat = []
    for r in civs:
        row = []
        for c in civs:
            rec = h2h.get(r, {}).get(c, {"wins": 0, "games": 0})
            row.append(rec["wins"] / rec["games"] if rec["games"] else float("nan"))
        mat.append(row)
    return pd.DataFrame(mat, index=civs, columns=civs)

def save_h2h_heatmap(h2h, out_png="h2h_heatmap.png"):
    """
    Save a heatmap of the head-to-head winrates between civilizations.

    The heatmap is a square matrix where each cell (row, column) represents the winrate of the row civilization against the column civilization.
    The winrates are centered around 0.5 (50% winrate) and color-coded with a colormap.
    The heatmap is annotated with the winrate values in each cell.

    The heatmap is saved to a PNG file with a specified filename.

    :param h2h: A dictionary of head-to-head game records.
    :param out_png: The filename of the output PNG file (default: "h2h_heatmap.png").
    """
    if pd is None or sns is None:
        print("[INFO] install pandas and seaborn to save heatmap (pip install pandas seaborn matplotlib)")
        return
    df = h2h_to_dataframe(h2h)
    plt.figure(figsize=(max(6, len(df)/2), max(4, len(df)/3)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="vlag", center=0.5, linewidths=.5)
    plt.title("Head-to-head winrate (row vs column)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] heatmap saved to {out_png}")


def h2h_to_games_dataframe(h2h):
    """Return a pandas DataFrame of head-to-head *game counts* (row civ vs column civ)."""
    if pd is None:
        raise RuntimeError("pandas is required for DataFrame export")
    civs = sorted(h2h.keys())
    mat = []
    for r in civs:
        row = []
        for c in civs:
            rec = h2h.get(r, {}).get(c, {"games": 0})
            row.append(rec.get("games", 0))
        mat.append(row)
    return pd.DataFrame(mat, index=civs, columns=civs)


def save_h2h_games_heatmap(h2h, out_png="h2h_games_heatmap.png", min_games_threshold=30):
    """Save a heatmap of the number of games for each pair (row vs column).

    Cells with fewer than `min_games_threshold` games are highlighted in red.
    """
    if pd is None or sns is None:
        print("[INFO] install pandas and seaborn to save heatmap (pip install pandas seaborn matplotlib)")
        return
    df = h2h_to_games_dataframe(h2h)
    if df.empty:
        print("[INFO] No head-to-head data to save.")
        return

    import numpy as np
    from matplotlib.patches import Rectangle

    plt.figure(figsize=(max(6, len(df)/2), max(4, len(df)/3)))
    ax = sns.heatmap(df, annot=True, fmt="d", cmap="Blues", linewidths=.5, linecolor='gray', cbar_kws={'label': 'games'})

    # Highlight cells with fewer than threshold games in semi-transparent red
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if not np.isfinite(val):
                continue
            if val < min_games_threshold:
                rect = Rectangle((j, i), 1, 1, facecolor='red', alpha=0.5, linewidth=0, zorder=2)
                ax.add_patch(rect)

    plt.title(f"Head-to-head games (row vs column) — cells < {min_games_threshold} highlighted red")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] games heatmap saved to {out_png}")


def _phase_from_time(t: int) -> str:
    """Return phase label for a timestamp (seconds).
    EARLY: 0 - 8 min (0-480s)
    MID: 8 - 20 min (480-1200s)
    LATE: 20+ min (>1200s)
    """
    if t is None:
        return "EARLY"
    if t < 480:
        return "EARLY"
    if t < 1200:
        return "MID"
    return "LATE"


def _clean_entity_from_icon(icon: str) -> str:
    """Extract a human-readable entity name from the `icon` path (e.g. '.../barracks' -> 'Barracks')."""
    if not icon:
        return ""
    name = icon.split('/')[-1]
    # remove common prefixes and tidy
    for prefix in ("building_", "unit_", "building", "race_", "icons", "hud", "races"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace('_', ' ').strip()
    return name.title()


def _game_duration_seconds(obj: dict) -> int | None:
    """Return game duration in seconds if available, otherwise None.

    Checks (in order): summary.duration, game.duration, (summary.finished_at - summary.started_at)
    """
    summary = obj.get('summary') or {}
    dur = summary.get('duration') or (obj.get('game') or {}).get('duration')
    if isinstance(dur, (int, float)):
        try:
            return int(dur)
        except Exception:
            pass
    # attempt to compute from timestamps if provided (unix seconds)
    started = summary.get('started_at')
    finished = summary.get('finished_at')
    if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
        try:
            return int(finished - started)
        except Exception:
            pass
    return None

def _build_resource_snapshots(player: dict):
    """
    Returns a dict: profile_id -> list of snapshots sorted by time
    Each snapshot: {'time': t, 'wood': x, 'food': y, 'gold': z, 'stone': w, ...}
    """
    res_data = player.get('resources') or {}
    times = res_data.get('timestamps') or []
    if not times:
        return {}

    snapshots = {}
    for i, t in enumerate(times):
        snapshot = {}
        for key in ['wood', 'food', 'gold', 'stone', 'wood_per_min', 'food_per_min', 'gold_per_min', 'stone_per_min']:
            vals = res_data.get(key) or []
            snapshot[key] = vals[i] if i < len(vals) else 0
        snapshots[t] = snapshot
    
    return snapshots


def _get_snapshot_for_time(snapshots, evt_time):
    # timestamps sorted once
    eligible_times = [t for t in snapshots.keys() if t <= evt_time]
    if not eligible_times:
        return {}  # no snapshot available
    latest_time = max(eligible_times)
    return snapshots[latest_time]


def extract_events_from_obj(obj: dict):
    """Return a list of event dicts extracted from a single game record JSON object.

    Each event: {game_id, profile_id, time (s), event (BUILD/FINISH/DESTROY/UNKNOWN), entity, map, player_civ, enemy_civ}
    """
    evs = []
    # try to find summary.players[] first (preferred)
    summary = obj.get('summary') or {}
    game_id = summary.get('game_id') or (obj.get('game') or {}).get('game_id')
    players = summary.get('players') or []
    game_map = summary.get('map_name') or (obj.get('game') or {}).get('map')

    game = obj.get('game')
    leaderboard = game.get('leaderboard')
    if leaderboard != "rm_solo":
        print(f"[INFO] Skipping not 1v1 {leaderboard}")
        return evs

    # fallback: try to find players under top-level game entry
    if not players and 'game' in obj:
        teams = obj['game'].get('teams') or obj['game'].get('players') or []
        entries = []
        if isinstance(teams, list) and teams and all(isinstance(t, list) for t in teams):
            for team in teams:
                entries.extend(team)
        elif isinstance(teams, list):
            entries = teams
        elif isinstance(teams, dict):
            for v in teams.values():
                if isinstance(v, list):
                    entries.extend(v)
                elif isinstance(v, dict):
                    entries.append(v)
        for entry in entries:
            p = entry.get('player') if 'player' in entry else entry
            if isinstance(p, dict):
                players.append(p)

    # Build helper maps: profile_id -> civ, profile_id -> team (if available), profile_id -> result/won
    profile_to_civ = {}
    profile_to_team = {}
    profile_to_result = {}
    profile_to_won = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get('profile_id') or p.get('profileId')
        civ = p.get('civilization') or p.get('civilisation') or p.get('civilization_attrib') or ''
        team = p.get('team')
        result = p.get('result') or p.get('outcome')
        if civ is not None:
            civ = str(civ).lower()
        if result is not None and isinstance(result, str):
            res_norm = result.strip().lower()
        else:
            res_norm = ''
        won_flag = res_norm in ('win', 'won', 'victory')
        if pid is not None:
            profile_to_civ[pid] = civ
            profile_to_team[pid] = team
            profile_to_result[pid] = res_norm
            profile_to_won[pid] = bool(won_flag)

    for player in players:
        if not isinstance(player, dict):
            continue
        profile_id = player.get('profile_id') or player.get('profileId')
        player_civ = profile_to_civ.get(profile_id) or (player.get('civilization') or player.get('civilisation') or '')
        if player_civ is not None:
            player_civ = str(player_civ).lower()

        # determine enemy civ: all unique civs of players not on this player's team
        enemy_civs = set()
        for pid, civ in profile_to_civ.items():
            if pid == profile_id:
                continue
            team_p = profile_to_team.get(pid)
            team_self = profile_to_team.get(profile_id)
            if team_p is not None and team_self is not None:
                if team_p != team_self and civ:
                    enemy_civs.add(civ)
            else:
                # no team info, assume other players are opponents
                if civ:
                    enemy_civs.add(civ)
        enemy_civs_list = sorted(enemy_civs)
        enemy_civs_joined = ';'.join(enemy_civs_list)

        build_order = player.get('build_order') or player.get('buildOrder') or []
        if not isinstance(build_order, list):
            continue

        resource_snapshot = _build_resource_snapshots(player)
       
        for item in build_order:
            icon = item.get('icon') or ''
            entity = _clean_entity_from_icon(icon)

            # types to map => event name
            for key, name in (('constructed', 'BUILD'), ('finished', 'FINISH'), ('destroyed', 'DESTROY')):
                times = item.get(key) or []
                if isinstance(times, list):
                    for t in times:
                        cur_snapshot = _get_snapshot_for_time(resource_snapshot, int(t))

                        e = {
                            'game_id': game_id,
                            'profile_id': profile_id,
                            'time': int(t) if t is not None else 0,
                            'event': name,
                            'entity': entity,
                            'player_civ': player_civ,
                            'player_result': profile_to_result.get(profile_id, ''),
                            'player_won': 1 if profile_to_won.get(profile_id) else 0,
                            'enemy_civ': enemy_civs_joined,
                            'map': game_map,
                        }

                        res = {
                            'wood': cur_snapshot['wood'],
                            'stone': cur_snapshot['stone'],
                            'food': cur_snapshot['food'],
                            'gold': cur_snapshot['gold'],
                            'food_per_min': cur_snapshot['food_per_min'],
                            'gold_per_min': cur_snapshot['gold_per_min'],
                            'stone_per_min': cur_snapshot['stone_per_min'],
                            'wood_per_min': cur_snapshot['wood_per_min'],
                            # 'oliveoil': cur_snapshot['oliveoil'],
                            # 'oliveoil_per_min': cur_snapshot['oliveoil_per_min'],
                            # 'food_gathered': cur_snapshot['food_gathered'],
                            # 'gold_gathered': cur_snapshot['gold_gathered'],
                            # 'stone_gathered': cur_snapshot['stone_gathered'],
                            # 'wood_gathered': cur_snapshot['wood_gathered'],
                            # 'oliveoil_gathered': cur_snapshot['oliveoil_gathered'],
                            # 'military': cur_snapshot['military'],
                            # 'economy': cur_snapshot['economy'],
                            # 'technology': cur_snapshot['technology'],
                            # 'society': cur_snapshot['society']
                        }

                        evs.append(e | res)

            # unknown may be a dict of lists
            unknown = item.get('unknown') or {}
            if isinstance(unknown, dict):
                for val in unknown.values():
                    if isinstance(val, list):
                        for t in val:
                            cur_snapshot = _get_snapshot_for_time(resource_snapshot, int(t))

                            e = {
                                'game_id': game_id,
                                'profile_id': profile_id,
                                'time': int(t) if t is not None else 0,
                                'event': 'UNKNOWN',
                                'entity': entity,
                                'player_civ': player_civ,
                                'player_result': profile_to_result.get(profile_id, ''),
                                'player_won': 1 if profile_to_won.get(profile_id) else 0,
                                'enemy_civ': enemy_civs_joined,
                                'map': game_map,
                            }

                            res = {
                                'wood': cur_snapshot['wood'],
                                'stone': cur_snapshot['stone'],
                                'food': cur_snapshot['food'],
                                'gold': cur_snapshot['gold'],
                                'food_per_min': cur_snapshot['food_per_min'],
                                'gold_per_min': cur_snapshot['gold_per_min'],
                                'stone_per_min': cur_snapshot['stone_per_min'],
                                'wood_per_min': cur_snapshot['wood_per_min'],
                                # 'oliveoil': cur_snapshot['oliveoil'],
                                # 'oliveoil_per_min': cur_snapshot['oliveoil_per_min'],
                                # 'food_gathered': cur_snapshot['food_gathered'],
                                # 'gold_gathered': cur_snapshot['gold_gathered'],
                                # 'stone_gathered': cur_snapshot['stone_gathered'],
                                # 'wood_gathered': cur_snapshot['wood_gathered'],
                                # 'oliveoil_gathered': cur_snapshot['oliveoil_gathered'],
                                # 'military': cur_snapshot['military'],
                                # 'economy': cur_snapshot['economy'],
                                # 'technology': cur_snapshot['technology'],
                                # 'society': cur_snapshot['society']
                            }
                            evs.append(e | res)

    return evs


def prepare_transformer_csv(files, out_csv: str) -> None:
    """Extract events from given file(s) and write a CSV suitable for transformer training.

    Output columns: game_id, profile_id, event, entity, time, delta_time (s), delta_time_scaled (log1p), phase, player_civ, player_result, player_won, enemy_civ
    """
    if isinstance(files, str):
        files = [files]

    all_events = []
    for fp in files:
        for path in sorted(glob.glob(fp)):
            print(f"[INFO] Extracting events from {path}")
            if path.endswith('.jsonl'):
                with open(path, 'r', encoding='utf-8') as fh:
                    for lineno, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[WARN] {path}:{lineno} JSON decode failed; skipping")
                            continue
                        # filter short games when duration is available
                        dur = _game_duration_seconds(obj)
                        if dur is not None and dur < 120:
                            print(f"[INFO] Skipping short game ({dur}s) in {path}:{lineno}")
                            continue
                        all_events.extend(extract_events_from_obj(obj))
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        obj = json.load(fh)
                except Exception as exc:
                    print(f"[WARN] Failed to parse {path}: {exc}")
                    continue
                # filter short games when duration is available
                dur = _game_duration_seconds(obj)
                if dur is not None and dur < 120:
                    print(f"[INFO] Skipping short game ({dur}s) in {path}")
                    continue
                all_events.extend(extract_events_from_obj(obj))

    # group by (game_id, profile_id) so delta is computed per player timeline
    from collections import defaultdict
    groups = defaultdict(list)
    for e in all_events:
        key = (e.get('game_id'), e.get('profile_id'))
        groups[key].append(e)

    with open(out_csv, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=['game_id', 'profile_id', 'event', 'entity', 'time', 'delta_time', 'delta_time_scaled', 'phase', 'player_civ', 'player_result', 'player_won', 'enemy_civ', 'map', 'wood', 'wood_per_min', 'stone', 'stone_per_min', 'food', 'food_per_min', 'gold', 'gold_per_min'])
        writer.writeheader()
        for key, evs in groups.items():
            evs_sorted = sorted(evs, key=lambda x: x.get('time', 0))
            prev_time = 0
            for ev in evs_sorted:
                t = ev.get('time', 0) or 0
                # cutoff after 30 minutes (1800s)
                if t > 1800:
                    break
                delta = int(t - prev_time)
                prev_time = t
                delta_scaled = math.log1p(delta) if delta >= 0 else 0.0
                phase = _phase_from_time(t)
                writer.writerow({
                    'game_id': ev.get('game_id'),
                    'profile_id': ev.get('profile_id'),
                    'event': ev.get('event'),
                    'entity': ev.get('entity'),
                    'time': t,
                    'delta_time': delta,
                    'delta_time_scaled': f"{delta_scaled:.6f}",
                    'phase': phase,
                    'player_civ': ev.get('player_civ'),
                    'player_result': ev.get('player_result'),
                    'player_won': ev.get('player_won'),
                    'enemy_civ': ev.get('enemy_civ'),
                    'map': ev.get('map'),
                    'wood': ev.get('wood'),
                    'stone': ev.get('stone'),
                    'food': ev.get('food'),
                    'gold': ev.get('gold'),
                    'food_per_min': ev.get('food_per_min'),
                    'gold_per_min': ev.get('gold_per_min'),
                    'stone_per_min': ev.get('stone_per_min'),
                    'wood_per_min': ev.get('wood_per_min'),
                    # 'oliveoil': ev.get('oliveoil'),
                    # 'oliveoil_per_min': ev.get('oliveoil_per_min'),
                    # 'food_gathered': ev.get('food_gathered'),
                    # 'gold_gathered': ev.get('gold_gathered'),
                    # 'stone_gathered': ev.get('stone_gathered'),
                    # 'wood_gathered': ev.get('wood_gathered'),
                    # 'oliveoil_gathered': ev.get('oliveoil_gathered'),
                    # 'military': ev.get('military'),
                    # 'economy': ev.get('economy'),
                    # 'technology': ev.get('technology'),
                    # 'society': ev.get('society')
                })
    print(f"[INFO] Events CSV written to {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Compute per-civilization winrates from collected .jsonl game records.")
    p.add_argument("paths", nargs="*", help="Path(s) or glob(s) to .jsonl files. If omitted, all *.jsonl in CWD are used.")
    p.add_argument("--winrate-heatmap", dest="winrate_heatmap", default="h2h_heatmap.png",
                   help="Write winrate heatmap PNG (default: h2h_heatmap.png). Use empty string to skip.")
    p.add_argument("--games-heatmap", dest="games_heatmap", default="h2h_games_heatmap.png",
                   help="Write games-count heatmap PNG (default: h2h_games_heatmap.png). Use empty string to skip.")
    p.add_argument("--min-games", dest="min_games", type=int, default=10,
                   help="Minimum games threshold to highlight cells in games heatmap (default: 10).")
    p.add_argument("--export-events", dest="export_events", default="transformer_input_test.csv",
                   help="Write transformer-ready events CSV to given path (e.g. events.csv). If empty, no events file is written.")
    args = p.parse_args()

    files = []
    if args.paths:
        for pat in args.paths:
            files.extend(sorted(glob.glob(pat)))
    else:
        files = sorted(glob.glob("*.jsonl"))

    if not files:
        print("No .jsonl files found (pass one or more paths/globs).")
        sys.exit(1)

    # If requested, export events CSV (does not require computing civ stats)
    if args.export_events:
        prepare_transformer_csv(files, args.export_events)
        # continue to compute stats too (optional)

    stats = defaultdict(lambda: {"wins": 0, "games": 0})
    # head-to-head: civ -> opponent_civ -> {"wins":int, "games":int}
    h2h = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "games": 0}))
    for fp in files:
        print(f"[INFO] Processing {fp}")
        process_file(fp, stats, h2h)

    if not stats:
        print("No civilization data found.")
        return

    rows = []
    total_games = 0
    total_wins = 0
    for civ, v in stats.items():
        wins = v["wins"]
        games = v["games"]
        total_games += games
        total_wins += wins
        winrate = (wins / games * 100) if games else 0.0
        rows.append((civ, wins, games, winrate))

    # sort by games (desc) so frequent civs appear first; adjust if you prefer sort by winrate
    rows.sort(key=lambda r: r[2], reverse=True)

    print("\nPer-civilization winrates:")
    for civ, wins, games, winrate in rows:
        print(f"{civ}: {winrate:.1f}% ({wins}/{games})")

    print(f"\nOverall: {total_wins}/{total_games} wins → {total_wins/total_games*100:.1f}%")

    # --- new: per-civ head-to-head summaries ---
    print("\nPer-civilization vs all other civs (excluding mirrors):")
    civs = sorted(h2h.keys(), key=lambda c: -sum(v["games"] for v in h2h[c].values()))
    for civ in civs:
        total_games_vs_others = 0
        total_wins_vs_others = 0
        for opp, rec in h2h[civ].items():
            if opp == civ:
                continue
            total_games_vs_others += rec["games"]
            total_wins_vs_others += rec["wins"]
        if total_games_vs_others:
            wr = total_wins_vs_others / total_games_vs_others * 100
            print(f"{civ}: {wr:.1f}% ({total_wins_vs_others}/{total_games_vs_others})")
        else:
            print(f"{civ}: no non-mirror matches")

    print("\nPairwise head-to-head (showing opponents with >=1 game):")
    for civ in civs:
        opps = [(opp, rec["wins"], rec["games"]) for opp, rec in h2h[civ].items() if rec["games"] > 0]
        if not opps:
            continue
        # sort opponents by games desc
        opps.sort(key=lambda x: x[2], reverse=True)
        print(f"\n{civ} vs:")
        for opp, wins, games in opps:
            wr = wins / games * 100 if games else 0.0
            print(f"  {opp}: {wr:.1f}% ({wins}/{games})")

    # save full matrix
    if args.winrate_heatmap:
        save_h2h_heatmap(h2h, out_png=args.winrate_heatmap)
    if args.games_heatmap:
        save_h2h_games_heatmap(h2h, out_png=args.games_heatmap, min_games_threshold=args.min_games)


if __name__ == "__main__":
    main()
