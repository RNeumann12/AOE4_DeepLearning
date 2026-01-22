#!/usr/bin/env python3
"""
Test script to verify which condition (player civ, enemy civ, or map) is being ignored.
"""

import os
import sys
import torch

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.abspath(__file__))
build_order_path = os.path.join(repo_root, 'BuildOrderPrediction')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if build_order_path not in sys.path:
    sys.path.insert(0, build_order_path)

from MoE_WithDecoder_infer import (
    load_model, 
    generate_build_order, 
    invert_vocab,
    find_vocab_id
)


def test_condition_impact():
    """Test which conditions actually affect the output."""
    
    device = torch.device('cuda')
    checkpoint_path = 'best_model.pth'
    
    print("=" * 90)
    print("TESTING CONDITION IMPACT ON BUILD ORDER GENERATION")
    print("=" * 90)
    
    model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping = load_model(
        checkpoint_path, device
    )
    
    inv_entity = invert_vocab(entity_vocab)
    
    # Test 1: Different player civs (same enemy, same map)
    print("\n" + "=" * 90)
    print("TEST 1: Different Player Civilizations (Same Enemy: French, Same Map: Altai)")
    print("=" * 90)
    
    enemy_civ_id = civ_vocab['french']
    enemy_civ = torch.tensor([enemy_civ_id], dtype=torch.long, device=device)
    map_tensor = torch.tensor([map_vocab['Altai']], dtype=torch.long, device=device)
    
    player_civs = ['english', 'french', 'chinese', 'mongols']
    results_player = {}
    
    for player_civ_name in player_civs:
        player_civ_id = civ_vocab[player_civ_name]
        player_civ = torch.tensor([player_civ_id], dtype=torch.long, device=device)
        
        with torch.no_grad():
            memory = model.encode(player_civ, enemy_civ, map_tensor)
        
        entities = generate_build_order(
            model=model,
            player_civ_id=player_civ_id,
            enemy_civ_id=enemy_civ_id,
            map_id=map_vocab['Altai'],
            entity_vocab=entity_vocab,
            num_steps=20,
            device=device,
            temperature=0.3,
            civ_entity_mapping=civ_entity_mapping,
            player_civ_name=player_civ_name,
            greedy=True,
            seed=42,
            exclude_special=True,
        )
        
        entity_names = []
        special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
        for ent_id in entities:
            entity_name = inv_entity.get(ent_id, f"<ID:{ent_id}>")
            if entity_name not in special_tokens:
                entity_names.append(entity_name)
        
        results_player[player_civ_name] = entity_names
        print(f"\n{player_civ_name.upper():15} → {', '.join(entity_names[:10])}")
    
    # Check if all player civs produce same output
    all_same_player = all(results_player[p] == results_player[player_civs[0]] for p in player_civs)
    print(f"\n→ All player civs produce identical output? {all_same_player} {'❌ PROBLEM!' if all_same_player else '✓ OK'}")
    
    # Test 2: Different enemy civs (same player, same map)
    print("\n" + "=" * 90)
    print("TEST 2: Different Enemy Civilizations (Same Player: English, Same Map: Altai)")
    print("=" * 90)
    
    player_civ_id = civ_vocab['english']
    player_civ = torch.tensor([player_civ_id], dtype=torch.long, device=device)
    map_tensor = torch.tensor([map_vocab['Altai']], dtype=torch.long, device=device)
    
    enemy_civs = ['french', 'mongols', 'chinese', 'ottoman']
    results_enemy = {}
    
    for enemy_civ_name in enemy_civs:
        enemy_civ_id = civ_vocab.get(enemy_civ_name)
        if enemy_civ_id is None:
            print(f"Skipping {enemy_civ_name} - not in vocab")
            continue
            
        enemy_civ = torch.tensor([enemy_civ_id], dtype=torch.long, device=device)
        
        with torch.no_grad():
            memory = model.encode(player_civ, enemy_civ, map_tensor)
        
        entities = generate_build_order(
            model=model,
            player_civ_id=player_civ_id,
            enemy_civ_id=enemy_civ_id,
            map_id=map_vocab['Altai'],
            entity_vocab=entity_vocab,
            num_steps=20,
            device=device,
            temperature=0.3,
            civ_entity_mapping=civ_entity_mapping,
            player_civ_name='english',
            greedy=True,
            seed=42,
            exclude_special=True,
        )
        
        entity_names = []
        special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
        for ent_id in entities:
            entity_name = inv_entity.get(ent_id, f"<ID:{ent_id}>")
            if entity_name not in special_tokens:
                entity_names.append(entity_name)
        
        results_enemy[enemy_civ_name] = entity_names
        print(f"\n{enemy_civ_name.upper():15} → {', '.join(entity_names[:10])}")
    
    all_same_enemy = all(results_enemy[e] == results_enemy[enemy_civs[0]] for e in enemy_civs if e in results_enemy)
    print(f"\n→ All enemy civs produce identical output? {all_same_enemy} {'❌ PROBLEM!' if all_same_enemy else '✓ OK'}")
    
    # Test 3: Different maps (same player, same enemy)
    print("\n" + "=" * 90)
    print("TEST 3: Different Maps (Same Player: English, Same Enemy: French)")
    print("=" * 90)
    
    player_civ_id = civ_vocab['english']
    player_civ = torch.tensor([player_civ_id], dtype=torch.long, device=device)
    enemy_civ_id = civ_vocab['french']
    enemy_civ = torch.tensor([enemy_civ_id], dtype=torch.long, device=device)
    
    maps = ['Altai', 'Boulder Bay', 'Four Lakes', 'Lipany', 'Ocean Gateway']
    results_map = {}
    
    for map_name in maps:
        map_id = map_vocab[map_name]
        map_tensor = torch.tensor([map_id], dtype=torch.long, device=device)
        
        with torch.no_grad():
            memory = model.encode(player_civ, enemy_civ, map_tensor)
        
        entities = generate_build_order(
            model=model,
            player_civ_id=player_civ_id,
            enemy_civ_id=enemy_civ_id,
            map_id=map_id,
            entity_vocab=entity_vocab,
            num_steps=20,
            device=device,
            temperature=0.3,
            civ_entity_mapping=civ_entity_mapping,
            player_civ_name='english',
            greedy=True,
            seed=42,
            exclude_special=True,
        )
        
        entity_names = []
        special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
        for ent_id in entities:
            entity_name = inv_entity.get(ent_id, f"<ID:{ent_id}>")
            if entity_name not in special_tokens:
                entity_names.append(entity_name)
        
        results_map[map_name] = entity_names
        print(f"\n{map_name:20} → {', '.join(entity_names[:10])}")
    
    all_same_map = all(results_map[m] == results_map[maps[0]] for m in maps)
    print(f"\n→ All maps produce identical output? {all_same_map} {'❌ PROBLEM!' if all_same_map else '✓ OK'}")
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    summary = [
        ("Player Civilization", "IGNORED" if all_same_player else "USED"),
        ("Enemy Civilization", "IGNORED" if all_same_enemy else "USED"),
        ("Map", "IGNORED" if all_same_map else "USED"),
    ]
    
    print("\nCondition Impact:")
    for condition, status in summary:
        status_symbol = "❌" if status == "IGNORED" else "✓"
        print(f"  {status_symbol} {condition:25} → {status}")
    
    ignored_count = sum(1 for _, s in summary if s == "IGNORED")
    if ignored_count == 0:
        print("\n✓ All conditions are being used - Model appears to be working correctly!")
    elif ignored_count == len(summary):
        print("\n❌ CRITICAL: All conditions are being ignored - Model ignores input!")
    else:
        print(f"\n⚠️ WARNING: {ignored_count} condition(s) being ignored - Check model architecture")


if __name__ == '__main__':
    test_condition_impact()
