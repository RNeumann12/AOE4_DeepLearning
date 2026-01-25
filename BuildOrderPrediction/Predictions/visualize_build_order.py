#!/usr/bin/env python3
"""
Build Order Visualization Tool
Generates a visual flowchart from prediction files using AoE4 World icons.
"""

import argparse
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import requests

# Cache directory for downloaded icons
CACHE_DIR = Path(__file__).parent / ".icon_cache"

# Icon mapping: entity name patterns -> (category, icon_name)
# Categories: units, buildings, technologies
ICON_MAPPING: Dict[str, Tuple[str, str]] = {
    # Units
    "villager": ("units", "villager-1"),
    "scout": ("units", "scout-1"),
    "spearman": ("units", "spearman-1"),
    "man-at-arms": ("units", "man-at-arms-1"),
    "man at arms": ("units", "man-at-arms-1"),
    "archer": ("units", "archer-1"),
    "longbowman": ("units", "longbowman-1"),
    "crossbowman": ("units", "crossbowman-1"),
    "horseman": ("units", "horseman-1"),
    "knight": ("units", "knight-1"),
    "lancer": ("units", "lancer-1"),
    "royal knight": ("units", "royal-knight-1"),
    "monk": ("units", "monk-1"),
    "prelate": ("units", "prelate-1"),
    "imam": ("units", "imam-1"),
    "trader": ("units", "trader-1"),
    "fishing boat": ("units", "fishing-boat-1"),
    "transport ship": ("units", "transport-ship-1"),
    "galley": ("units", "galley-1"),
    "war galley": ("units", "war-galley-1"),
    "hulk": ("units", "hulk-1"),
    "carrack": ("units", "carrack-1"),
    "springald": ("units", "springald-1"),
    "mangonel": ("units", "mangonel-1"),
    "bombard": ("units", "bombard-1"),
    "trebuchet": ("units", "counterweight-trebuchet-1"),
    "battering ram": ("units", "battering-ram-1"),
    "siege tower": ("units", "siege-tower-1"),
    "ribauldequin": ("units", "ribauldequin-1"),
    "cannon": ("units", "cannon-1"),
    "handcannoneer": ("units", "handcannoneer-1"),
    "grenadier": ("units", "grenadier-1"),
    "streltsy": ("units", "streltsy-1"),
    "zhuge nu": ("units", "zhuge-nu-1"),
    "fire lancer": ("units", "fire-lancer-1"),
    "palace guard": ("units", "palace-guard-1"),
    "nest of bees": ("units", "nest-of-bees-1"),
    
    # Buildings - Economy
    "house": ("buildings", "house"),
    "town center": ("buildings", "town-center"),
    "capital town center": ("buildings", "capital-town-center"),
    "mill": ("buildings", "mill"),
    "lumber camp": ("buildings", "lumber-camp"),
    "mining camp": ("buildings", "mining-camp"),
    "gold mining camp": ("buildings", "mining-camp"),
    "stone mining camp": ("buildings", "mining-camp"),
    "farm": ("buildings", "farm"),
    "market": ("buildings", "market"),
    "dock": ("buildings", "dock"),
    
    # Buildings - Military
    "barracks": ("buildings", "barracks"),
    "archery range": ("buildings", "archery-range"),
    "stable": ("buildings", "stable"),
    "siege workshop": ("buildings", "siege-workshop"),
    "blacksmith": ("buildings", "blacksmith"),
    "monastery": ("buildings", "monastery"),
    "keep": ("buildings", "keep"),
    "outpost": ("buildings", "outpost"),
    "stone wall": ("buildings", "stone-wall"),
    "palisade wall": ("buildings", "palisade"),
    "palisade gate": ("buildings", "palisade-gate"),
    "stone wall gate": ("buildings", "stone-wall-gate"),
    "stone wall tower": ("buildings", "stone-wall-tower"),
    
    # Landmarks - English
    "westminster abbey": ("buildings", "abbey-of-kings"),  # fallback to similar landmark
    "council hall": ("buildings", "council-hall"),
    "abbey of kings": ("buildings", "abbey-of-kings"),
    "kings palace": ("buildings", "kings-palace"),
    "king's palace": ("buildings", "kings-palace"),
    "berkshire palace": ("buildings", "berkshire-palace"),
    "wynguard palace": ("buildings", "wynguard-palace"),
    "white tower": ("buildings", "the-white-tower"),
    
    # Landmarks - French
    "chamber of commerce": ("buildings", "chamber-of-commerce"),
    "school of cavalry": ("buildings", "school-of-cavalry"),
    "royal institute": ("buildings", "royal-institute"),
    "guild hall": ("buildings", "guild-hall"),
    "college of artillery": ("buildings", "college-of-artillery"),
    "red palace": ("buildings", "red-palace"),
    
    # Landmarks - Generic/Other civs
    "landmark": ("buildings", "keep"),  # fallback to keep icon
    
    # Special/Age markers
    "crown king": ("units", "king-2"),
    "king": ("units", "king-2"),
}

# Color scheme for different entity types
ENTITY_COLORS = {
    "villager": "#4CAF50",      # Green - Economy
    "house": "#9E9E9E",         # Gray - Support
    "military": "#F44336",       # Red - Military
    "building": "#2196F3",       # Blue - Buildings
    "landmark": "#FFD700",       # Gold - Landmarks
    "age": "#9C27B0",           # Purple - Age ups
    "default": "#607D8B",        # Blue Gray - Default
}


def get_entity_color(entity: str) -> str:
    """Get color based on entity type."""
    entity_lower = entity.lower()
    
    if "villager" in entity_lower:
        return ENTITY_COLORS["villager"]
    elif "house" in entity_lower:
        return ENTITY_COLORS["house"]
    elif any(x in entity_lower for x in ["age", "westminster", "council", "abbey", "palace", "college", "guild", "school", "chamber", "tower"]):
        return ENTITY_COLORS["landmark"]
    elif any(x in entity_lower for x in ["barracks", "archery", "stable", "siege", "blacksmith", "keep"]):
        return ENTITY_COLORS["military"]
    elif any(x in entity_lower for x in ["camp", "mill", "market", "dock", "farm", "town center"]):
        return ENTITY_COLORS["building"]
    else:
        return ENTITY_COLORS["default"]


def get_icon_url(entity: str) -> str:
    """Get the icon URL for an entity from aoe4world."""
    entity_lower = entity.lower().strip()
    
    # Try direct match first
    for pattern, (category, icon_name) in ICON_MAPPING.items():
        if pattern in entity_lower:
            return f"https://data.aoe4world.com/images/{category}/{icon_name}.png"
    
    # Default fallback icons
    if "age" in entity_lower and "display" in entity_lower:
        return None  # Skip age display markers
    
    return None  # No icon found


def download_icon(url: str, size: int = 64) -> Image.Image:
    """Download and cache an icon, return as PIL Image."""
    if url is None:
        return None
    
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Create cache filename from URL
    cache_name = url.split("/")[-1]
    cache_path = CACHE_DIR / cache_name
    
    try:
        if cache_path.exists():
            img = Image.open(cache_path)
        else:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img.save(cache_path)
        
        # Resize to standard size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Warning: Could not load icon from {url}: {e}")
        return None


def parse_prediction_file(filepath: str) -> List[Tuple[int, str]]:
    """Parse a prediction file and return list of (step, entity) tuples."""
    entries = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: "1      Villager" or similar
            match = re.match(r'(\d+)\s+(.+)', line)
            if match:
                step = int(match.group(1))
                entity = match.group(2).strip()
                entries.append((step, entity))
    
    return entries


def create_flowchart(
    entries: List[Tuple[int, str]],
    output_path: str,
    title: str = "Build Order Prediction",
    items_per_row: int = 8,
    show_numbers: bool = True,
    icon_size: int = 48,
    figsize: Tuple[int, int] = None,
):
    """Create a modern card-based build order visualization."""
    
    # Filter out non-displayable entries (like "Age Display Persistent")
    filtered = [(step, entity) for step, entity in entries 
                if "display persistent" not in entity.lower()]
    
    n_items = len(filtered)
    
    # Card dimensions
    card_width = 4.0
    card_height = 0.6
    card_spacing = 0.15
    items_per_column = 12
    
    n_columns = (n_items + items_per_column - 1) // items_per_column
    
    if figsize is None:
        figsize = (card_width * n_columns + 0.5, items_per_column * (card_height + card_spacing) + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#2d2d2d')
    ax.set_facecolor('#2d2d2d')
    
    ax.set_xlim(-0.2, n_columns * card_width + 0.2)
    ax.set_ylim(-items_per_column * (card_height + card_spacing) - 0.5, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(n_columns * card_width / 2, 0.5, title, 
            fontsize=14, fontweight='bold', color='white',
            ha='center', va='center')
    
    # Draw each entry as a card
    for idx, (step, entity) in enumerate(filtered):
        col = idx // items_per_column
        row = idx % items_per_column
        
        x = col * card_width + 0.1
        y = -row * (card_height + card_spacing) - 0.3
        
        # Get icon
        icon_url = get_icon_url(entity)
        icon = download_icon(icon_url, size=icon_size) if icon_url else None
        
        # Draw card background (rounded rectangle)
        card_bg = mpatches.FancyBboxPatch(
            (x, y - card_height), card_width - 0.3, card_height,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.08),
            facecolor='#3d3d3d', edgecolor='#4d4d4d', linewidth=1,
            zorder=1
        )
        ax.add_patch(card_bg)
        
        # Draw step number on the left
        if show_numbers:
            ax.text(x + 0.25, y - card_height/2, str(step), 
                   fontsize=9, color='#888888',
                   ha='center', va='center', zorder=2)
        
        # Draw icon background (rounded square)
        icon_bg_x = x + 0.5
        icon_bg_size = 0.45
        icon_bg = mpatches.FancyBboxPatch(
            (icon_bg_x, y - card_height + 0.075), icon_bg_size, icon_bg_size,
            boxstyle=mpatches.BoxStyle("Round", pad=0.01, rounding_size=0.05),
            facecolor='#5a4a3a', edgecolor='none',
            zorder=2
        )
        ax.add_patch(icon_bg)
        
        # Draw icon
        icon_center_x = icon_bg_x + icon_bg_size / 2
        icon_center_y = y - card_height / 2
        
        if icon:
            imagebox = OffsetImage(icon, zoom=0.35)
            ab = AnnotationBbox(imagebox, (icon_center_x, icon_center_y), 
                               frameon=False, zorder=3)
            ax.add_artist(ab)
        
        # Draw entity name
        ax.text(x + 1.1, y - card_height/2, entity, 
               fontsize=9, color='white',
               ha='left', va='center', zorder=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='#2d2d2d', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved visualization to: {output_path}")


def create_compact_timeline(
    entries: List[Tuple[int, str]],
    output_path: str,
    title: str = "Build Order",
    max_items: int = 15,
    icon_size: int = 48,
):
    """Create a compact horizontal card-based timeline (good for slides)."""
    
    # Filter and limit entries
    filtered = [(step, entity) for step, entity in entries 
                if "display persistent" not in entity.lower()][:max_items]
    
    n_items = len(filtered)
    
    # Card dimensions for horizontal layout
    card_width = 0.8
    card_height = 1.2
    card_spacing = 0.1
    
    figsize = (n_items * (card_width + card_spacing) + 1, 2.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#2d2d2d')
    ax.set_facecolor('#2d2d2d')
    
    ax.set_xlim(-0.3, n_items * (card_width + card_spacing) + 0.3)
    ax.set_ylim(-0.3, card_height + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(n_items * (card_width + card_spacing) / 2, card_height + 0.3, title, 
            fontsize=12, fontweight='bold', color='white',
            ha='center', va='center')
    
    for idx, (step, entity) in enumerate(filtered):
        x = idx * (card_width + card_spacing)
        y = 0
        
        # Get icon
        icon_url = get_icon_url(entity)
        icon = download_icon(icon_url, size=icon_size) if icon_url else None
        
        # Draw icon background (rounded square)
        icon_bg = mpatches.FancyBboxPatch(
            (x, y + 0.3), card_width, card_width,
            boxstyle=mpatches.BoxStyle("Round", pad=0.01, rounding_size=0.08),
            facecolor='#5a4a3a', edgecolor='#6a5a4a', linewidth=1,
            zorder=1
        )
        ax.add_patch(icon_bg)
        
        # Draw icon
        icon_center_x = x + card_width / 2
        icon_center_y = y + 0.3 + card_width / 2
        
        if icon:
            imagebox = OffsetImage(icon, zoom=0.6)
            ab = AnnotationBbox(imagebox, (icon_center_x, icon_center_y), 
                               frameon=False, zorder=2)
            ax.add_artist(ab)
        
        # Step number below icon
        ax.text(icon_center_x, y + 0.15, str(step), 
               fontsize=8, color='#888888',
               ha='center', va='center')
    
    # "..." if truncated
    if len(entries) > max_items:
        ax.text(n_items * (card_width + card_spacing), card_width / 2 + 0.3, "...", 
               ha='left', va='center', fontsize=14, color='#888')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#2d2d2d', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved compact timeline to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AoE4 build order predictions as flowcharts"
    )
    parser.add_argument(
        "input_file",
        help="Path to the prediction file (e.g., english_vs_french_dry_arabia.txt)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output image path (default: same name as input with .png extension)"
    )
    parser.add_argument(
        "--title",
        help="Chart title (default: derived from filename)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Create compact single-row timeline (better for slides)"
    )
    parser.add_argument(
        "--items-per-row",
        type=int,
        default=8,
        help="Items per row in flowchart mode (default: 8)"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=15,
        help="Max items in compact mode (default: 15)"
    )
    parser.add_argument(
        "--no-numbers",
        action="store_true",
        help="Hide step numbers"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    input_path = Path(args.input_file)
    if args.output:
        output_path = args.output
    else:
        suffix = "_timeline.png" if args.compact else "_flowchart.png"
        output_path = input_path.with_suffix("").as_posix() + suffix
    
    # Determine title
    if args.title:
        title = args.title
    else:
        # Convert filename to title: english_vs_french_dry_arabia -> English vs French (Dry Arabia)
        name = input_path.stem
        parts = name.split("_vs_")
        if len(parts) == 2:
            civ1 = parts[0].replace("_", " ").title()
            rest = parts[1].split("_", 1)
            civ2 = rest[0].title()
            map_name = rest[1].replace("_", " ").title() if len(rest) > 1 else ""
            title = f"{civ1} vs {civ2}"
            if map_name:
                title += f" ({map_name})"
        else:
            title = name.replace("_", " ").title()
    
    # Parse and visualize
    entries = parse_prediction_file(args.input_file)
    print(f"Parsed {len(entries)} build order steps")
    
    if args.compact:
        create_compact_timeline(
            entries,
            output_path,
            title=title,
            max_items=args.max_items,
        )
    else:
        create_flowchart(
            entries,
            output_path,
            title=title,
            items_per_row=args.items_per_row,
            show_numbers=not args.no_numbers,
        )


if __name__ == "__main__":
    main()
