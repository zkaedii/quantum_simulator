#!/usr/bin/env python3
"""
🛸📊 FOCUSED JSON VISUALIZATION DEMO 📊🛸
==========================================
Demonstrating JSON visualization with our actual alien mathematics data!
"""

import os
from alien_json_visualization_engine import (
    AlienJSONVisualizationEngine, VisualizationMode
)


def run_focused_demo():
    """Run focused demonstration with actual alien mathematics data"""

    print("🛸📊" * 40)
    print("🌟 ALIEN MATHEMATICS JSON VISUALIZATION DEMO 🌟")
    print("🛸📊" * 40)
    print("Visualizing actual alien mathematics session data!")
    print()

    # Initialize the engine
    engine = AlienJSONVisualizationEngine()

    # Demo files to visualize
    demo_files = [
        "pure_alien_creative_session_20250722_010503.json",
        "babylonian_cuneiform_session_20250721_093002.json",
        "norse_viking_mega_session_20250721_095532.json",
        "quantum_casino_packages_summary_20250721_133859.json"
    ]

    for demo_file in demo_files:
        if os.path.exists(demo_file):
            print(f"🔍 VISUALIZING: {demo_file}")
            print("=" * 60)

            # Load the data
            data = engine.load_json_file(demo_file)

            if data:
                # Show tree view (first 1500 chars)
                print("🌳 TREE VIEW:")
                print("-" * 30)
                tree_view = engine.visualize_json(
                    data, VisualizationMode.TREE_VIEW)
                print(tree_view[:1500])
                if len(tree_view) > 1500:
                    print(f"... (showing first 1500 of "
                          f"{len(tree_view)} characters)")
                print()

                # Show alien dashboard
                print("🛸 ALIEN MATHEMATICS DASHBOARD:")
                print("-" * 30)
                dashboard = engine.visualize_json(
                    data, VisualizationMode.ALIEN_DASHBOARD)
                print(dashboard)

                # Show statistics
                print("📊 STATISTICAL ANALYSIS:")
                print("-" * 30)
                stats = engine.visualize_json(
                    data, VisualizationMode.STATISTICAL)
                print(stats)

                print("🌟" * 60)
                print()
                break  # Just demo the first available file

    print("✨ JSON VISUALIZATION DEMO COMPLETE! ✨")
    print("🛸 The JSON visualization engine is ready for any alien "
          "mathematics data!")


if __name__ == "__main__":
    run_focused_demo()
