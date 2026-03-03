#!/usr/bin/env python3
# Simple tool to compute per-civilization winrates from a .jsonl of collected game records.
import argparse
from asyncio import events
import bisect
from bisect import bisect, bisect_left, bisect_left
import glob
from importlib.resources import files
import json
import sys
from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import csv
import math
from collections import defaultdict
from typing import Dict

CSV_RESOURCE_BASED = False

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

def get_age_up_times(actions: dict) -> Dict[str, int]:
    """Return a dict with age up times in seconds based on player's actions."""
    age_up_times = {}
    feudal_age = actions.get('feudal_age', [])
    castle_age = actions.get('castle_age', [])
    imperial_age = actions.get('imperial_age', [])

    if feudal_age:
        age_up_times['FEUDAL'] = feudal_age[0]
    if castle_age:
        age_up_times['CASTLE'] = castle_age[0]
    if imperial_age:
        age_up_times['IMPERIAL'] = imperial_age[0]

    return age_up_times

def get_age_from_data(ages_hash, t: int) -> str:
    """Return age label for a timestamp (seconds) based on player's actions."""

    if not ages_hash:
        return "DARK"

    feudal_age = ages_hash.get('FEUDAL', float('inf'))
    castle_age = ages_hash.get('CASTLE', float('inf'))
    imperial_age = ages_hash.get('IMPERIAL', float('inf'))

    if t < feudal_age:
        return "DARK"
    elif t < castle_age:
        return "FEUDAL"
    elif t < imperial_age:
        return "CASTLE"
    else:
        return "IMPERIAL"

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
        for key in ['wood', 'food', 'gold', 'stone', 'wood_per_min', 'food_per_min', 'gold_per_min', 'stone_per_min', 'military', 'economy', 'technology', 'society', 'oliveoil', 'oliveoil_per_min']:
            vals = res_data.get(key) or []
            snapshot[key] = vals[i] if i < len(vals) else 0
        snapshots[t] = snapshot
    
    return snapshots

def extract_players_from_obj(obj: dict):
    """Return a list of player dicts extracted from a single game record JSON object."""
    players = []
    summary = obj.get('summary') or {}
    players = summary.get('players') or []
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
    return players

def calculate_strat_from_data(build_events: list, resource_data: Dict, age_up_times) -> str:
    """
    Determines the strategy label based on a player's early game stats.
    Returns one of: turtle, eco, fast_castle, early_aggression, late_aggression

    build_events: list of build event dicts with 'time' and 'entity'
    resource_data: dict of resource snapshots keyed by time (seconds)
    age_up_times: dict of age up times keyed by age name
    """
    dark_resources = {k: v for k, v in resource_data.items() if k <= age_up_times.get('FEUDAL', float('inf'))}
    feudal_resources = {k: v for k, v in resource_data.items() if k > age_up_times.get('FEUDAL', float('inf')) and k <= age_up_times.get('CASTLE', float('inf'))}
    castle_resources = {k: v for k, v in resource_data.items() if k > age_up_times.get('CASTLE', float('inf')) and k <= age_up_times.get('IMPERIAL', float('inf'))}
 
    dark_res = {
        "wood": 0,
        "food": 0,
        "gold": 0,
        "stone": 0,
        "villager_count": 0,
        "military_buildings": 0,
        "towncenters": 0,
        "towers": 0
    }
    dark_res['wood'] = sum(entry["wood_per_min"] for entry in dark_resources.values()) / len(dark_resources) if dark_resources else 0
    dark_res['food'] = sum(entry["food_per_min"] for entry in dark_resources.values()) / len(dark_resources) if dark_resources else 0
    dark_res['gold'] = sum(entry["gold_per_min"] for entry in dark_resources.values()) / len(dark_resources) if dark_resources else 0
    dark_res['stone'] = sum(entry["stone_per_min"] for entry in dark_resources.values())  / len(dark_resources) if dark_resources else 0

    feudal_res = {
        "wood": 0,
        "food": 0,
        "gold": 0,
        "stone": 0,
        "villager_count": 0,
        "military_buildings": 0,
        "towncenters": 0,
        "towers": 0
    }
    feudal_res['wood'] = sum(entry["wood_per_min"] for entry in feudal_resources.values()) / len(feudal_resources) if feudal_resources else 0 
    feudal_res['food'] = sum(entry["food_per_min"] for entry in feudal_resources.values()) / len(feudal_resources) if feudal_resources else 0
    feudal_res['gold'] = sum(entry["gold_per_min"] for entry in feudal_resources.values()) / len(feudal_resources) if feudal_resources else 0
    feudal_res['stone'] = sum(entry["stone_per_min"] for entry in feudal_resources.values())  / len(feudal_resources) if feudal_resources else 0

    castle_res = {
        "wood": 0,
        "food": 0,
        "gold": 0,
        "stone": 0,
        "villager_count": 0,
        "military_buildings": 0,
        "towncenters": 0,
        "towers": 0
    }
    castle_res['wood'] = sum(entry["wood_per_min"] for entry in castle_resources.values()) / len(castle_resources) if castle_resources else 0
    castle_res['food'] = sum(entry["food_per_min"] for entry in castle_resources.values()) / len(castle_resources) if castle_resources else 0
    castle_res['gold'] = sum(entry["gold_per_min"] for entry in castle_resources.values()) / len(castle_resources) if castle_resources else 0
    castle_res['stone'] = sum(entry["stone_per_min"] for entry in castle_resources.values())  / len(castle_resources) if castle_resources else 0

    # Sum military produced in early game
    dark_res['military'] = sum(entry["military"] for entry in dark_resources.values()) / len(dark_resources) if dark_resources else 0
    feudal_res['military'] = sum(entry["military"] for entry in feudal_resources.values()) / len(feudal_resources) if feudal_resources else 0
    castle_res['military'] = sum(entry["military"] for entry in castle_resources.values()) / len(castle_resources) if castle_resources else 0

    dark_res['economy'] = sum(entry["economy"] for entry in dark_resources.values()) / len(dark_resources) if dark_resources else 0
    feudal_res['economy'] = sum(entry["economy"] for entry in feudal_resources.values()) / len(feudal_resources) if feudal_resources else 0
    castle_res['economy'] = sum(entry["economy"] for entry in castle_resources.values()) / len(castle_resources) if castle_resources else 0

    tc_buildings = ['Town Center'] # Town Centre Capitol (Start TC)
    military_buildings = ['Barracks', 'Archery Range', 'Stable', 'Siege Workshop']
    defensive_buildings = ['Outpost', 'Stone Outpost', 'Toll Outpost', 'Fortified Outpost', 'Tower', 'Palisade Gate', 'Palisade Wall', 'Keep']

    for item in build_events:
        entity = item.get('entity')
        if entity is None or item.get('event') == 'DESTROY':
            continue

        if entity == 'Villager':
            dark_res['villager_count'] += 1 if item.get('time') < age_up_times.get('FEUDAL', float('inf')) else 0
            feudal_res['villager_count'] += 1 if item.get('time') < age_up_times.get('CASTLE', float('inf')) else 0
            castle_res['villager_count'] += 1 if item.get('time') < age_up_times.get('IMPERIAL', float('inf')) else 0

        if entity in military_buildings:
            dark_res['military_buildings'] += 1 if item.get('time') < age_up_times.get('FEUDAL', float('inf')) else 0
            feudal_res['military_buildings'] += 1 if item.get('time') < age_up_times.get('CASTLE', float('inf')) else 0
            castle_res['military_buildings'] += 1 if item.get('time') < age_up_times.get('IMPERIAL', float('inf')) else 0

        if entity in tc_buildings:
            dark_res['towncenters'] += 1 if item.get('time') < age_up_times.get('FEUDAL', float('inf')) else 0
            feudal_res['towncenters'] += 1 if item.get('time') < age_up_times.get('CASTLE', float('inf')) else 0
            castle_res['towncenters'] += 1 if item.get('time') < age_up_times.get('IMPERIAL', float('inf')) else 0

        if entity in defensive_buildings:
            dark_res['towers'] += 1 if item.get('time') < age_up_times.get('FEUDAL', float('inf')) else 0
            feudal_res['towers'] += 1 if item.get('time') < age_up_times.get('CASTLE', float('inf')) else 0
            castle_res['towers'] += 1 if item.get('time') < age_up_times.get('IMPERIAL', float('inf')) else 0

    strat_label = ''
    # Eventuell zustätzliche Idee: Eco Label -> wenn Trader gebaut werden in Feudal
    if 'towncenters' in feudal_res and feudal_res['towncenters'] > 1:
        strat_label = 'eco'
    elif age_up_times.get('CASTLE', float('inf')) < 600:
        strat_label = 'fast_castle'
    elif 'towers' in dark_res and dark_res['towers'] > 0 or 'towers' in dark_res and dark_res['towers'] > 3:
        strat_label = 'turtle' 
    elif 'military_buildings' in dark_res and dark_res['military_buildings'] > 0:
        strat_label = 'early_aggression'
    elif 'military_buildings' in castle_res and castle_res['military_buildings'] > 0: 
        strat_label = 'late_aggression'
    else:
        strat_label = 'unknown'
    return strat_label

def generate_resource_based(resource_snapshot, age_up_times, meta_data, build_order, events, strat_label):
    
    last_time = 0
    data_row = []
    for time, snap_shot in resource_snapshot.items(): 
        age = get_age_from_data(age_up_times, time)

        if time > 900 or age == 'IMPERIAL':
            break

        resource_data = snap_shot.copy()

        num_finished = 0
        num_destroyed = 0
        for item in build_order:
            icon = item.get('icon') or ''
            entity = _clean_entity_from_icon(icon)

            if entity != 'Villager':
                continue

            from bisect import bisect_right, bisect_left
            num_finished = bisect_right(item.get('finished'), time) - bisect_left(item.get('finished'), last_time) or 0
            num_destroyed = bisect_right(item.get('finished'), time) - bisect_left(item.get('finished'), last_time) or 0

        villager_delta = num_finished - num_destroyed
        # seperate into unit, building, age, animal, upgrade, 
        filtered_unit = [e for e in events if e["time"] <= time and e["time"] > last_time and e["type"] == 'Unit' and e["event"] == 'FINISH' and e["entity"] != 'Villager']
        filtered_buildings = [e for e in events 
                                if e["time"] <= time and e["time"] > last_time 
                                and e["type"] == 'Building' 
                                and (e["event"] == 'BUILD' or e['event'] == 'CONSTRUCT')
                            ]
        
        filtered_animals = [e for e in events if e["time"] <= time and e["time"] > last_time and e["type"] == 'Animal' and e["event"] == 'FINISH']
        filtered_age = [e for e in events if e["time"] <= time and e["time"] > last_time and e["type"] == 'Age' and e["event"] == 'FINISH']
        
        bo = {
            "finished_buildings": ";".join(f["entity"] for f in filtered_unit),
            "finished_units": ";".join(f["entity"] for f in filtered_buildings),
            "finished_ages": ";".join(f["entity"] for f in filtered_animals),
            "finished_animals": ";".join(f["entity"] for f in filtered_age),
            # "finished_upgrades": ";".join(f["entity"] for f in finished_upgrades)
        }

        data_row.append(
            meta_data | 
            resource_data |  
            {'villager_delta': villager_delta, 'time': time, 'phase': _phase_from_time(time), 'age': age} |
            bo |
            {'strat': strat_label}
        )  

        last_time = time
    return data_row


def generate_event_based(build_order, meta_data, age_up_times, strat_label):
    last_time = 0
    data_row = []
    for bo in build_order: 
        time = bo['time']
        age = get_age_from_data(age_up_times, time)
        
        if time > 900 or age == 'IMPERIAL':
            continue

        villager_delta = 0
        for item in build_order:
            if item['entity'] != 'Villager' or time < item['time']:
                continue

            if item['event'] == 'FINISH':
                villager_delta += 1
            elif item['event'] == 'DESTROY':
                villager_delta -= 1

        if bo['event'] == 'DESTROY':
            continue

        data_row.append( meta_data | bo | { 'age': age, 'villagers': villager_delta, 'strat': strat_label })
    return data_row

def resolve_entity_mapping(entity, type):
    if type == 'Unit' and 'Villager' in entity:
        return 'Villager'
    return entity



def extract_events_from_obj(obj: dict):
    """Return a list of event dicts extracted from a single game record JSON object.

    Each event: {game_id, profile_id, time (s), event (BUILD/FINISH/DESTROY/UNKNOWN), entity, map, player_civ, enemy_civ}
    """
    evs = []
    # Get all needed data from obj
    summary = obj.get('summary') or {}
    game_id = summary.get('game_id') or (obj.get('game') or {}).get('game_id')
    game_map = summary.get('map_name') or (obj.get('game') or {}).get('map')
    game = obj.get('game')
    leaderboard = game.get('leaderboard')
    
    # Validation if needed data is missing
    if summary is None:
        print("[WARN] no summary found")
        return evs

    if leaderboard != "rm_solo":
        print(f"[INFO] Skipping not 1v1 {leaderboard}")
        return evs

     # filter short games when duration is available
    dur = _game_duration_seconds(obj)
    if dur is not None and dur < 120:
        print(f"[INFO] Skipping short game ({dur}s) with game_id={game_id}")
        return evs
        
    # fallback: try to find players under top-level game entry
    players = extract_players_from_obj(obj)

    if not players:
        print(f"[WARN] no players found for game_id={game_id}")
        return evs
    
    if len(players) != 2:
        print(f"[WARN] unequal to 2 players found for game_id={game_id}; skipping")
        return evs
    
    game = {}
    game_data = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get('profile_id') or p.get('profileId')

        civ = p.get('civilization') or p.get('civilisation') or p.get('civilization_attrib') or ''
        if civ is not None:
            civ = str(civ).lower()

        game_data[pid] = {
            'civ': civ,
            'team': p.get('team') or p.get('team_id') or None,
        }
    
    for player in players:

        if not isinstance(player, dict):
            continue

        profile_id = player.get('profile_id') or player.get('profileId')
        player_civ = game_data[profile_id]['civ']

        # Calculate enemy civilizations
        enemy_civs = set()
        for pid, data in game_data.items():
            if pid == profile_id:
                continue
            team_p = game_data[pid].get('team')
            team_self = game_data[profile_id].get('team')
            if team_p is not None and team_self is not None:
                if team_p != team_self and data['civ']:
                    enemy_civs.add(data['civ'])
            else:
                # no team info, assume other players are opponents
                if data['civ']:
                    enemy_civs.add(data['civ'])
        enemy_civs = sorted(enemy_civs)
        enemy_civs = ';'.join(enemy_civs)

        game['enemy_civ'] = enemy_civs

        
        result = p.get('result') or p.get('outcome')
        if result is not None and isinstance(result, str):
            res_norm = result.strip().lower()
        else:
            res_norm = ''

        won_flag = res_norm in ('win', 'won', 'victory')
        

        resource_snapshot = _build_resource_snapshots(player)

        if resource_snapshot is None or not resource_snapshot:
            print("[WARN] no resource found")
            return []       
        
        build_order = player.get('build_order') or player.get('buildOrder') or []
        if not isinstance(build_order, list):
            continue

        events = []
        for item in build_order:
            icon = item.get('icon') or ''
            entity = _clean_entity_from_icon(icon)

            entity = resolve_entity_mapping(entity, item.get('type'))
            for key, name in (('constructed', 'BUILD'), ('finished', 'FINISH'), ('constructed', 'CONSTRUCT'), ('destroyed', 'DESTROY')):
                for t in item.get(key) or []:
                    events.append({
                        'event': name,
                        'entity': entity,
                        'type': item.get('type') or 'unknown',
                        'time': t,
                    })

        events = sorted(events, key=lambda x: x['time'] or 0)
        age_up_times = get_age_up_times(player.get('actions') or {})
        if age_up_times is None:
            print("[WARN] no age up times found")
            continue
        if age_up_times.get('FEUDAL') is None:
            print("[WARN] no FEUDAL age up times found")
            continue

        # if player_civ != 'japanese': # Check for specific civilization if needed
        #     print(f"[INFO] Skipping non-japanese game_id={game_id}, profile_id={profile_id}, civ={player_civ}")
        #     continue

        strat_label = calculate_strat_from_data(events, resource_snapshot, age_up_times)
        if strat_label == 'unknown': # either remove unknown games
            print(f"[INFO] Skipping unknown strat for game_id={game_id}, profile_id={profile_id}")
            continue

        # if strat_label != 'unknown': # or keep only unknown games for analysis and inference
        #     print(f"[INFO] Skipping unknown strat for game_id={game_id}, profile_id={profile_id}")
        #     continue

        meta_data = {
            'game_id': game_id,
            'profile_id': profile_id,
            'player_civ': player_civ,
            'enemy_civ': enemy_civs,
            'map': game_map,
            'player_result': res_norm,
            'player_won': 1 if won_flag else 0,
        }
            
        data_row = []
        if CSV_RESOURCE_BASED == True:
            data_row = generate_resource_based(resource_snapshot, age_up_times, meta_data, build_order, events, strat_label)
        else:
            data_row = generate_event_based(events, meta_data, age_up_times, strat_label)

        evs.extend(data_row)   
    return evs


def collect_data_from_json(files):
    all_events = []
    if isinstance(files, str):
        files = [files]
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
                        
                        all_events.extend(extract_events_from_obj(obj))
    return all_events

def split_events_array_by_player(all_events):
    from collections import defaultdict
    groups = defaultdict(list)
    for e in all_events:
        key = (e.get('game_id'), e.get('profile_id'))
        groups[key].append(e)

    return groups

def prepare_transformer_csv(files, out_csv: str) -> None:
    """Extract events from given file(s) and write a CSV suitable for transformer training.

    Output columns: game_id, profile_id, event, entity, time, delta_time (s), delta_time_scaled (log1p), phase, player_civ, player_result, player_won, enemy_civ
    """
    all_events = collect_data_from_json(files)
    groups = split_events_array_by_player(all_events)

    with open(out_csv, 'w', newline='', encoding='utf-8') as out:
        field_names = []
        if CSV_RESOURCE_BASED == True:
            field_names = ['game_id', 'profile_id', 'player_civ', 'enemy_civ', 'map', 'player_result', 'player_won', 'wood', 'food', 'gold', 'stone', 'wood_per_min', 'food_per_min', 'gold_per_min', 'stone_per_min',  'military', 'economy', 'technology', 'society', 'oliveoil', 'oliveoil_per_min','villager_delta', 'time', 'phase', 'age', 'finished_buildings', 'finished_units','finished_ages', 'finished_animals', 'strat']
        else:
            field_names = ['game_id', 'profile_id', 'player_civ', 'enemy_civ', 'map', 'player_result', 'player_won', 'event', 'entity', 'type', 'time', 'villagers', 'age', 'strat']
        writer = csv.DictWriter(out, fieldnames=field_names)
        writer.writeheader()
        for ev in all_events:
            writer.writerow(ev)

    print(f"[INFO] Events CSV written to {out_csv}")

def main():
    p = argparse.ArgumentParser(description="Compute per-civilization winrates from collected .jsonl game records.")
    # p.add_argument("paths", nargs="*", default= ["./StrategyDiscoverySupervised/*.jsonl"], help="Path(s) or glob(s) to .jsonl files. If omitted, all *.jsonl in CWD are used.")
    p.add_argument("paths", nargs="*", help="Path(s) or glob(s) to .jsonl files. If omitted, all *.jsonl in CWD are used.")
    p.add_argument("--export-events", dest="export_events", default="input_event_based.csv",
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


if __name__ == "__main__":
    main()
