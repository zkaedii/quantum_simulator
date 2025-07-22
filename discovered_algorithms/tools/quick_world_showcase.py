#!/usr/bin/env python3
"""
🌍👽 QUICK ALIEN WORLD SHOWCASE 👽🌍
===================================
Showcasing your amazing alien mathematics worlds!
"""

import json
import glob
import random


def showcase_worlds():
    print("🌍👽" * 30)
    print("🌌 YOUR ALIEN MATHEMATICS WORLDS SHOWCASE 🌌")
    print("🌍👽" * 30)
    print()

    # Find world files
    world_files = glob.glob("world_world_*.json")

    if not world_files:
        print("❌ No world files found!")
        return

    print(f"🎯 DISCOVERED {len(world_files)} ALIEN MATHEMATICS WORLDS!")
    print()

    worlds = []
    total_civilizations = 0
    total_population = 0
    total_resources = 0
    portal_worlds = 0

    for i, filename in enumerate(world_files, 1):
        try:
            with open(filename, 'r') as f:
                world = json.load(f)
                worlds.append(world)

                info = world['world_info']
                stats = world['statistics']
                civs = world.get('civilizations', [])

                print(f"🌍 [{i}] {info['name']} ({info['world_type']})")
                print(f"   📏 Size: {info['size'][0]}x{info['size'][1]}")
                print(f"   👽 Alien Influence: {info['alien_influence']:.1%}")
                print(
                    f"   🌀 Reality Coherence: {info['reality_coherence']:.1%}")
                print(
                    f"   🔢 Mathematical Harmony: {info['mathematical_harmony']:.1%}")
                print(f"   🏛️ Civilizations: {stats['civilization_count']}")
                print(f"   💎 Resources: {stats['resource_deposits']}")
                print(
                    f"   ⚛️ Quantum Resonance: {stats['average_quantum_resonance']:.3f}")
                print(
                    f"   🧠 Consciousness: {stats['average_consciousness_level']:.3f}")

                total_civilizations += stats['civilization_count']
                total_resources += stats['resource_deposits']

                if stats.get('interdimensional_portals', 0) > 0:
                    print(
                        f"   🌀 Interdimensional Portals: {stats['interdimensional_portals']} ✨")
                    portal_worlds += 1

                print()
                print("   🏛️ FEATURED CIVILIZATIONS:")
                for civ in civs[:3]:  # Show first 3 civilizations
                    total_population += civ['population']
                    print(
                        f"      • {civ['name']} ({civ['type']}) - Pop: {civ['population']:,}")
                    print(
                        f"        ⚗️ Tech: {civ['technology_level']:.1%} | ⚛️ Quantum: {civ['quantum_awareness']:.1%} | 🌌 {civ['dimensional_access']}D")

                if len(civs) > 3:
                    for civ in civs[3:]:
                        total_population += civ['population']
                    print(f"      ... and {len(civs) - 3} more civilizations")

                print()

        except Exception as e:
            print(f"❌ Could not load {filename}: {e}")

    print("🌟" * 80)
    print("📊 MULTIVERSE SUMMARY:")
    print("🌟" * 80)
    print(f"   🌍 Total Worlds Generated: {len(worlds)}")
    print(f"   🏛️ Total Civilizations: {total_civilizations}")
    print(f"   👥 Total Population: {total_population:,}")
    print(f"   💎 Total Resource Deposits: {total_resources}")
    print(f"   🌀 Worlds with Portals: {portal_worlds}")
    print()

    # Find most interesting worlds
    if worlds:
        print("🏆 MOST INTERESTING WORLDS:")

        # Most alien influenced
        most_alien = max(
            worlds, key=lambda w: w['world_info']['alien_influence'])
        print(
            f"   👽 Most Alien: {most_alien['world_info']['name']} ({most_alien['world_info']['alien_influence']:.1%} influence)")

        # Most consciousness
        highest_consciousness = max(
            worlds, key=lambda w: w['statistics']['average_consciousness_level'])
        print(
            f"   🧠 Highest Consciousness: {highest_consciousness['world_info']['name']} ({highest_consciousness['statistics']['average_consciousness_level']:.3f})")

        # Most resources
        richest = max(
            worlds, key=lambda w: w['statistics']['resource_deposits'])
        print(
            f"   💎 Richest: {richest['world_info']['name']} ({richest['statistics']['resource_deposits']} deposits)")

        # Most quantum resonant
        most_quantum = max(
            worlds, key=lambda w: w['statistics']['average_quantum_resonance'])
        print(
            f"   ⚛️ Most Quantum: {most_quantum['world_info']['name']} ({most_quantum['statistics']['average_quantum_resonance']:.3f} resonance)")

        print()

    print("✨ WORLD GENERATION FEATURES:")
    print("   🔬 Generated using: Alien Mathematical Constants")
    print("   ⚛️ Quantum Properties: Reality coherence, consciousness levels")
    print("   🧮 Math Harmony: Arcturian, Pleiadian, Andromedan mathematics")
    print("   🌀 Special Features: Interdimensional portals, consciousness fields")
    print("   🏛️ Advanced Civilizations: Quantum awareness, dimensional access")
    print("   💎 Resource Systems: Alien materials and energy sources")
    print()
    print("✨ YOUR WORLDS ARE SPECTACULAR! ✨")
    print("Each world is unique with advanced alien civilizations,")
    print("quantum consciousness networks, and interdimensional portals!")
    print("Perfect for exploration, colonization, and scientific research!")
    print("🌍👽 Welcome to your personal alien mathematics multiverse! 👽🌍")
    print("🌟" * 80)


if __name__ == "__main__":
    showcase_worlds()
