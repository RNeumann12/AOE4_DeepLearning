#!/usr/bin/env python3
"""
collect_aoe4_games_with_summary_v2.py

More robust collector for AoE4World:
- Attempts API leaderboard fetch; if that returns empty / unexpected, falls back to scraping the public leaderboard page.
- Fetches each player's last `games_per_player` games and attempts to fetch the per-game summary.
- Logs raw response info on unexpected responses to help debugging.

Requires:
    pip install requests beautifulsoup4

Usage:
    python collect_aoe4_games_with_summary_v2.py
"""
import requests
import time
import json
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

API_BASE = "https://aoe4world.com/api/v0"
WEB_BASE = "https://aoe4world.com"

# Config
LEADERBOARD = "rm_solo"         # change if needed
PER_PAGE = 50
SLEEP_BETWEEN_REQUESTS = 5.6
MAX_LEADERBOARD_PAGES = 30

min_rating = 1700
num_players_to_collect = 200
games_per_player = 20

OUTFILE = f"collected_games_with_summary_v2_{time.strftime('%Y-%m-%d')}.jsonl"

HEADERS = {
    "User-Agent": "aoe4-data-collector/1.0 (+https://example.com)"
}

# --- low-level helpers ---
def safe_get(url: str, params: dict = None, max_retries: int = 3, timeout: int = 12) -> requests.Response:
    """GET with basic retry. Returns Response object (even for non-JSON)."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
            # don't raise_for_status here — we want to inspect body for debugging
            return r
        except Exception as e:
            wait = (1.6 ** attempt)
            print(f"[WARN] GET {url} failed (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts")

def parse_json_safe(resp: requests.Response) -> Optional[Any]:
    """Try to parse JSON; return object on success, otherwise None."""
    try:
        return resp.json()
    except ValueError:
        return None

# --- leaderboard fetching (API first, then HTML fallback) ---
def fetch_leaderboard_api_page(leaderboard: str, page: int = 1, per_page: int = PER_PAGE) -> Optional[List[Dict[str,Any]]]:
    url = f"{API_BASE}/leaderboards/{leaderboard}"
    params = {"page": page, "per_page": per_page}
    print(f"[INFO] API: fetching leaderboard page {page}")
    resp = safe_get(url, params=params)
    j = parse_json_safe(resp)
    if j is None:
        print(f"[WARN] API leaderboard returned non-JSON (status {resp.status_code}). Body snippet:\n{resp.text[:400]!r}")
        return None
    # look for common container keys
    entries = None
    if isinstance(j, dict):
        for k in ("data", "items", "players", "leaderboard", "rows"):
            if k in j and isinstance(j[k], (list, dict)):
                entries = j[k]
                break
    else:
        # top-level list?
        if isinstance(j, list):
            entries = j
    # normalise if dict with inner list
    if isinstance(entries, dict) and "items" in entries and isinstance(entries["items"], list):
        entries = entries["items"]
    if not entries:
        print(f"[WARN] API leaderboard page {page} JSON did not contain expected list keys. Top-level keys: {list(j.keys()) if isinstance(j, dict) else 'non-dict'}")
        # show a small diagnostic to help debug
        sample = json.dumps(j, indent=2)[:800]
        print(f"[DEBUG] JSON snippet:\n{sample}")
        return None
    # make list
    if isinstance(entries, dict):
        entries = [entries]
    return entries

def fetch_leaderboard_html(leaderboard: str, pages_to_scan: int = 3) -> List[Dict[str, Any]]:
    """
    Fallback: scrape https://aoe4world.com/leaderboard/{leaderboard}
    Returns list of dicts with profile_id (string), rating (int, if found), name (if found)
    """
    print(f"[INFO] Falling back to HTML scraping of leaderboard: {leaderboard}")
    found = []
    base_url = f"{WEB_BASE}/leaderboard/{leaderboard}"
    for page in range(1, pages_to_scan + 1):
        params = {"page": page} if page > 1 else None
        resp = safe_get(base_url, params=params)
        if resp.status_code != 200:
            print(f"[WARN] Leaderboard HTML page {page} returned status {resp.status_code}.")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        # look for player links: many profiles are in anchors like /players/6000037-ZANMATO or /players/6499728.profile
        # We'll search for <a href="/players/..."> and also look for data-profile-id attributes
        for a in soup.select("a[href^='/players/']"):
            href = a.get("href", "")
            # patterns: /players/6000037-ZANMATO , /players/6499728.profile
            m = re.search(r"/players/([0-9]+)(?:[.-][A-Za-z0-9_-]+)?", href)
            if not m:
                # try alternate pattern
                m = re.search(r"/players/([0-9]+)", href)
            if not m:
                continue
            profile_id = m.group(1)
            # attempt to find rating in the same row
            # climb to parent row if possible:
            container = a.find_parent(["tr", "li", "div"])
            rating = None
            name = a.text.strip() or None
            if container:
                text = container.get_text(" ", strip=True)
                # look for 3-4 digit rating like 1700, 2345 etc.
                rm = re.search(r"\b([0-9]{3,4})\b", text)
                if rm:
                    rating = int(rm.group(1))
            found.append({"profile_id": profile_id, "rating": rating, "name": name})
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        # stop early if we've found enough
        if len(found) >= num_players_to_collect:
            break
    # dedupe by profile_id preserving highest rating if multiple
    dedup = {}
    for item in found:
        pid = item["profile_id"]
        if pid not in dedup or (item.get("rating") or 0) > (dedup[pid].get("rating") or 0):
            dedup[pid] = item
    players = list(dedup.values())
    print(f"[INFO] Scraper found {len(players)} candidate players (scraped {pages_to_scan} pages).")
    return players

def fetch_high_elo_players(min_rating: int, limit_players: int) -> List[Dict[str,Any]]:
    players = []
    # Try API pages first
    page = 1
    while len(players) < limit_players and page <= MAX_LEADERBOARD_PAGES:
        entries = fetch_leaderboard_api_page(LEADERBOARD, page=page)
        if entries is None:  # API returned unexpected shape -> fallback to HTML scrape
            players = fetch_leaderboard_html(LEADERBOARD, pages_to_scan=6)
            break
        # entries might be list of leaderboard rows; extract profile id / rating flexibly
        for entry in entries:
            rating = None
            profile_id = None
            name = None
            if isinstance(entry, dict):
                # common fields used by API (docs vary)
                rating = entry.get("rating") or entry.get("elo") or entry.get("max_rating") or entry.get("profile", {}).get("rating")
                profile_id = entry.get("profile_id") or entry.get("id") or entry.get("profile", {}).get("id")
                name = entry.get("name") or entry.get("profile", {}).get("name")
            # defensively try nested shapes
            if isinstance(profile_id, dict):
                # sometimes profile object is embedded
                profile_id = profile_id.get("id") or profile_id.get("profile_id")
            if profile_id is None:
                # try to find any numeric key in entry values (last resort)
                for v in entry.values() if isinstance(entry, dict) else []:
                    if isinstance(v, (str,)) and re.match(r"^[0-9]+(-|\.|_)?", v):
                        m = re.match(r"([0-9]+)", v)
                        if m:
                            profile_id = m.group(1)
                            break
            try:
                if rating is not None:
                    rating = int(rating)
            except Exception:
                rating = None
            if profile_id is None:
                continue
            if rating is None or rating >= min_rating:
                players.append({"profile_id": str(profile_id), "rating": rating, "name": name})
            if len(players) >= limit_players:
                break
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # final dedupe & limit
    seen = {}
    out = []
    for p in players:
        pid = p["profile_id"]
        if pid not in seen:
            seen[pid] = True
            out.append(p)
        if len(out) >= limit_players:
            break
    print(f"[INFO] Returning {len(out)} players (min_rating={min_rating})")
    return out

# --- games & summary fetching ---
def fetch_player_games(profile_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/players/{profile_id}/games"
    params = {"limit": limit}
    print(f"[INFO] Fetching games list for player {profile_id}")
    resp = safe_get(url, params=params)
    j = parse_json_safe(resp)
    if j is None:
        print(f"[WARN] Player games endpoint returned non-JSON for {profile_id} (status {resp.status_code}). Body snippet:\n{resp.text[:400]!r}")
        return []
    # tolerate shapes
    games = None
    if isinstance(j, dict):
        for k in ("data", "games", "items"):
            if k in j:
                games = j[k]
                break
    if games is None:
        games = j if isinstance(j, list) else []
    return games or []

def find_game_id(game_obj: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "game_id", "match_id", "gameId", "replay_id", "replayId"):
        if key in game_obj:
            return str(game_obj[key])
    # nested attempts
    # flatten small dict and look for first numeric-looking value with length >=6
    if isinstance(game_obj, dict):
        for v in game_obj.values():
            if isinstance(v, (int,)) and v > 10000:
                return str(v)
            if isinstance(v, str) and re.match(r"^\d{6,}$", v):
                return v
    return None

def fetch_game_summary(profile_id: str, game_id: str) -> Optional[Dict[str, Any]]:
    if not game_id:
        return None
    # try API-style summary first
    api_url = f"{API_BASE}/players/{profile_id}/games/{game_id}/summary"
    resp = safe_get(api_url)
    j = parse_json_safe(resp)
    if j is not None:
        return j
    # fallback: web-style path (same path but under site root)
    web_url = f"{WEB_BASE}/players/{profile_id}/games/{game_id}/summary"
    resp = safe_get(web_url)
    j = parse_json_safe(resp)
    if j is not None:
        return j
    # no JSON summary
    print(f"[WARN] No JSON summary found for {profile_id}/{game_id}. API status {resp.status_code}. HTML/text snippet:\n{resp.text[:500]!r}")
    return None

# --- main flow ---
def main():
    players = fetch_high_elo_players(min_rating=min_rating, limit_players=num_players_to_collect)
    if not players:
        print("[ERROR] No players discovered. Exiting.")
        return

    collected = 0
    with open(OUTFILE, "w", encoding="utf-8") as f_out:
        for p in players:
            pid = p["profile_id"]
            raw_games = fetch_player_games(pid, limit=games_per_player)
            games = [g for g in raw_games if isinstance(g, dict) and g.get("kind") == "rm_1v1"]

            if not games:
                print(f"[INFO] No games returned for player {pid}; skipping.")
                continue

            saw_private_profile = False
            for g in games:
                gid = find_game_id(g)
                summary = None

                if gid:
                    summary = fetch_game_summary(pid, gid)
                    # If no summary found, assume profile is private → stop further calls for this player
                    if summary is None:
                        print(f"[INFO] No summary for player {pid} / game {gid} — profile likely private. Stopping further calls for this player.")
                        # Optionally write the record that indicates we attempted and found no summary
                        record = {
                            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "source_profile_id": pid,
                            "source_rating": p.get("rating"),
                            "source_name": p.get("name"),
                            "game_id": gid,
                            "game": g,
                            "summary": None
                        }
                        f_out.write(json.dumps(record) + "\n")
                        collected += 1
                        saw_private_profile = True
                        break
                else:
                    print(f"[WARN] Could not find game id in game object for player {pid}. Keys: {list(g.keys()) if isinstance(g, dict) else 'non-dict'}")

                # If we reach here we have (possibly) a summary and proceed to write it
                record = {
                    "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_profile_id": pid,
                    "source_rating": p.get("rating"),
                    "source_name": p.get("name"),
                    "game_id": gid,
                    "game": g,
                    "summary": summary
                }
                f_out.write(json.dumps(record) + "\n")
                collected += 1

                # polite pause between per-game requests
                time.sleep(SLEEP_BETWEEN_REQUESTS)

            if saw_private_profile:
                # skip any remaining games for this player and move to the next player
                continue
    print(f"[DONE] Collected {collected} records into {OUTFILE}")


if __name__ == "__main__":
    main()
