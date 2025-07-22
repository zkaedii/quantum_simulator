#!/usr/bin/env python3
"""
🛸📊 VISUALIZE PURE ALIEN CREATIVE SESSION 📊🛸
===============================================
Specialized visualization of our alien creative session data!
"""

import os
from json_visualizer_demo import JSONVisualizer


def visualize_creative_session():
    """Visualize the pure alien creative session data"""

    print("🛸🌟 PURE ALIEN CREATIVE SESSION VISUALIZATION 🌟🛸")
    print("=" * 60)

    session_file = "pure_alien_creative_session_20250722_010503.json"

    if not os.path.exists(session_file):
        print(f"❌ File not found: {session_file}")
        return

    visualizer = JSONVisualizer()

    # Load the creative session
    print(f"📂 Loading: {session_file}")
    data = visualizer.load_json_file(session_file)

    if not data:
        print("❌ Failed to load session data")
        return

    print()
    print("🧠 CONSCIOUSNESS SESSION ANALYSIS:")
    print("-" * 40)

    # Extract key consciousness metrics
    session_info = data.get('session_info', {})
    final_stats = data.get('final_stats', {})

    if session_info:
        print(f"👤 User: {session_info.get('user_name', 'Unknown')}")
        print(f"🆔 Session: {session_info.get('session_id', 'Unknown')}")
        print(
            f"⏱️ Duration: {session_info.get('duration_minutes', 0)} minutes")

    if final_stats:
        consciousness = final_stats.get('consciousness_level', 0)
        reality_power = final_stats.get('reality_manipulation_power', 0)
        telepathic = final_stats.get('telepathic_ability', 0)

        print(f"🧠 Final Consciousness: {consciousness:.3f}")
        print(f"🎮 Reality Power: {reality_power:.3f}")
        print(f"📡 Telepathic Ability: {telepathic:.3f}")

    achievements = data.get('achievements', [])
    if achievements:
        print(f"🏆 Achievements: {len(achievements)}")
        for achievement in achievements[:3]:
            print(f"   ✨ {achievement}")
        if len(achievements) > 3:
            print(f"   ... and {len(achievements) - 3} more")

    print()
    print("🌟 FULL TREE VISUALIZATION:")
    print("-" * 40)

    # Show complete tree structure
    tree_view = visualizer.visualize_tree(data, max_depth=10)
    print(tree_view)

    print()
    print("📊 DETAILED STATISTICS:")
    print("-" * 40)
    stats = visualizer.create_statistics_view(data)
    print(stats)

    print()
    print("🛸 ALIEN DASHBOARD:")
    print("-" * 40)
    dashboard = visualizer.create_alien_dashboard(data)
    print(dashboard)

    print("🌟" * 60)
    print("✨ CREATIVE SESSION VISUALIZATION COMPLETE! ✨")
    print("🧠 Consciousness evolution successfully analyzed!")
    print("🛸 Alien mathematics session fully visualized!")
    print("🌟" * 60)


if __name__ == "__main__":
    visualize_creative_session()
