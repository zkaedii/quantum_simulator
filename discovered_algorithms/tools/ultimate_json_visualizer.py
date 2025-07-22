#!/usr/bin/env python3
"""
🛸📊 ULTIMATE JSON VISUALIZATION ENGINE 📊🛸
============================================
The most beautiful JSON visualizer for alien mathematics data!

🌟 FEATURES:
📊 Beautiful tree visualization with colors
📈 Comprehensive statistical analysis  
🛸 Alien mathematics dashboard
🧠 Consciousness level detection
⚡ Fast and lightweight
🌈 Color-coded output
🔍 Deep structure analysis
"""

import json
import os
import math
from datetime import datetime
from typing import Dict, List, Any, Optional


class Colors:
    """Console colors for beautiful output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Basic colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class UltimateJSONVisualizer:
    """The ultimate JSON visualization engine"""

    def __init__(self):
        self.colors = Colors()

    def load_json_file(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            file_size = os.path.getsize(filepath)
            print(f"✅ Loaded: {filepath} ({file_size:,} bytes)")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return {}

    def get_key_color(self, key: str) -> str:
        """Get appropriate color for JSON key"""
        key_lower = str(key).lower()

        if any(word in key_lower for word in ['consciousness', 'telepathic', 'mind', 'astral']):
            return self.colors.BRIGHT_MAGENTA
        elif any(word in key_lower for word in ['quantum', 'algorithm', 'circuit']):
            return self.colors.BRIGHT_BLUE
        elif any(word in key_lower for word in ['alien', 'galactic', 'cosmic', 'stellar']):
            return self.colors.BRIGHT_CYAN
        elif any(word in key_lower for word in ['reality', 'dimension', 'spacetime']):
            return self.colors.BRIGHT_GREEN
        elif any(word in key_lower for word in ['mandala', 'crystal', 'sacred']):
            return self.colors.BRIGHT_YELLOW
        elif any(word in key_lower for word in ['id', 'name', 'type']):
            return self.colors.WHITE
        elif any(word in key_lower for word in ['timestamp', 'time', 'date']):
            return self.colors.CYAN
        else:
            return self.colors.GREEN

    def get_value_color(self, value: Any) -> str:
        """Get appropriate color for JSON value"""
        if isinstance(value, bool):
            return self.colors.BRIGHT_YELLOW
        elif isinstance(value, int):
            return self.colors.BRIGHT_BLUE
        elif isinstance(value, float):
            return self.colors.BRIGHT_CYAN
        elif isinstance(value, str):
            if value.lower() in ['true', 'false']:
                return self.colors.BRIGHT_YELLOW
            else:
                return self.colors.BRIGHT_WHITE
        else:
            return self.colors.WHITE

    def visualize_tree(self, data: Any, max_depth: int = 8,
                       current_depth: int = 0, prefix: str = "") -> str:
        """Create beautiful tree visualization"""

        if current_depth > max_depth:
            return f"{prefix}🔻 [Max depth {max_depth} reached]\n"

        result = []

        if isinstance(data, dict):
            if current_depth == 0:
                result.append(f"{self.colors.BRIGHT_CYAN}🏛️ JSON Object "
                              f"({len(data)} keys){self.colors.RESET}\n")

            items = list(data.items())

            for i, (key, value) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")

                # Colorize key
                key_color = self.get_key_color(key)
                type_info = f" ({type(value).__name__})"

                result.append(f"{prefix}{connector}{key_color}{key}"
                              f"{self.colors.RESET}{type_info}")

                # Add value preview for simple types
                if isinstance(value, (str, int, float, bool)):
                    if len(str(value)) < 60:
                        value_color = self.get_value_color(value)
                        result.append(f" = {value_color}{repr(value)}"
                                      f"{self.colors.RESET}")

                result.append("\n")

                # Recursively visualize nested structures
                if isinstance(value, (dict, list)):
                    result.append(self.visualize_tree(value, max_depth,
                                                      current_depth + 1, next_prefix))

        elif isinstance(data, list):
            if current_depth == 0:
                result.append(f"{self.colors.BRIGHT_YELLOW}📋 JSON Array "
                              f"({len(data)} items){self.colors.RESET}\n")

            # Intelligent array display
            if len(data) > 10:
                # Show first 3 items
                for i in range(3):
                    connector = "├── "
                    result.append(f"{prefix}{connector}"
                                  f"{self.colors.BRIGHT_MAGENTA}[{i}]"
                                  f"{self.colors.RESET}")

                    if isinstance(data[i], (str, int, float, bool)):
                        if len(str(data[i])) < 60:
                            value_color = self.get_value_color(data[i])
                            result.append(f" = {value_color}{repr(data[i])}"
                                          f"{self.colors.RESET}")

                    result.append("\n")

                    if isinstance(data[i], (dict, list)):
                        next_prefix = prefix + "│   "
                        result.append(self.visualize_tree(data[i], max_depth,
                                                          current_depth + 1, next_prefix))

                # Show ellipsis
                result.append(f"{prefix}├── {self.colors.CYAN}... "
                              f"({len(data) - 6} more items) ...{self.colors.RESET}\n")

                # Show last 3 items
                for i in range(len(data) - 3, len(data)):
                    is_last = (i == len(data) - 1)
                    connector = "└── " if is_last else "├── "
                    result.append(f"{prefix}{connector}"
                                  f"{self.colors.BRIGHT_MAGENTA}[{i}]"
                                  f"{self.colors.RESET}")

                    if isinstance(data[i], (str, int, float, bool)):
                        if len(str(data[i])) < 60:
                            value_color = self.get_value_color(data[i])
                            result.append(f" = {value_color}{repr(data[i])}"
                                          f"{self.colors.RESET}")

                    result.append("\n")

                    if isinstance(data[i], (dict, list)):
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        result.append(self.visualize_tree(data[i], max_depth,
                                                          current_depth + 1, next_prefix))
            else:
                # Show all items for smaller arrays
                for i, item in enumerate(data):
                    is_last = (i == len(data) - 1)
                    connector = "└── " if is_last else "├── "
                    next_prefix = prefix + ("    " if is_last else "│   ")

                    result.append(f"{prefix}{connector}"
                                  f"{self.colors.BRIGHT_MAGENTA}[{i}]"
                                  f"{self.colors.RESET}")

                    if isinstance(item, (str, int, float, bool)):
                        if len(str(item)) < 60:
                            value_color = self.get_value_color(item)
                            result.append(f" = {value_color}{repr(item)}"
                                          f"{self.colors.RESET}")

                    result.append("\n")

                    if isinstance(item, (dict, list)):
                        result.append(self.visualize_tree(item, max_depth,
                                                          current_depth + 1, next_prefix))

        return "".join(result)

    def analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Comprehensive JSON structure analysis"""

        total_keys = 0
        max_depth = 0
        total_objects = 0
        total_arrays = 0
        total_values = 0
        data_types = {}
        alien_keywords = []
        consciousness_levels = []

        def analyze_recursive(obj: Any, current_depth: int = 0) -> None:
            nonlocal total_keys, max_depth, total_objects, total_arrays
            nonlocal total_values, data_types, alien_keywords, consciousness_levels

            max_depth = max(max_depth, current_depth)

            if isinstance(obj, dict):
                total_objects += 1
                total_keys += len(obj)

                for key, value in obj.items():
                    key_str = str(key).lower()

                    # Check for alien/quantum keywords
                    alien_terms = ['alien', 'quantum', 'consciousness', 'telepathic',
                                   'galactic', 'mandala', 'cosmic', 'dimensional',
                                   'reality', 'spacetime', 'arcturian', 'pleiadian']

                    for term in alien_terms:
                        if term in key_str and key not in alien_keywords:
                            alien_keywords.append(key)
                            break

                    # Extract consciousness levels
                    if 'consciousness' in key_str and isinstance(value, (int, float)):
                        consciousness_levels.append(float(value))

                    analyze_recursive(value, current_depth + 1)

            elif isinstance(obj, list):
                total_arrays += 1
                for item in obj:
                    analyze_recursive(item, current_depth + 1)

            else:
                total_values += 1
                data_type = type(obj).__name__
                data_types[data_type] = data_types.get(data_type, 0) + 1

        analyze_recursive(data)

        # Calculate metrics
        max_consciousness = max(
            consciousness_levels) if consciousness_levels else 0.0
        avg_consciousness = sum(
            consciousness_levels) / len(consciousness_levels) if consciousness_levels else 0.0

        complexity_score = (
            total_keys * 0.1 +
            max_depth * 2.0 +
            total_objects * 0.5 +
            total_arrays * 0.3 +
            len(data_types) * 1.0 +
            len(alien_keywords) * 2.0
        )

        return {
            'total_keys': total_keys,
            'max_depth': max_depth,
            'total_objects': total_objects,
            'total_arrays': total_arrays,
            'total_values': total_values,
            'data_types': data_types,
            'alien_keywords': alien_keywords,
            'consciousness_levels': consciousness_levels,
            'max_consciousness': max_consciousness,
            'avg_consciousness': avg_consciousness,
            'complexity_score': complexity_score,
            'alien_data_detected': len(alien_keywords) > 0
        }

    def create_statistics_view(self, data: Any) -> str:
        """Comprehensive statistics visualization"""

        metrics = self.analyze_structure(data)

        result = []
        result.append(f"{self.colors.BRIGHT_CYAN}📊 JSON STATISTICS ANALYSIS"
                      f"{self.colors.RESET}\n")
        result.append("=" * 60 + "\n\n")

        # Structure metrics
        result.append(
            f"{self.colors.YELLOW}📏 STRUCTURE METRICS:{self.colors.RESET}\n")
        result.append(f"   🔸 Total Keys: {self.colors.BRIGHT_WHITE}"
                      f"{metrics['total_keys']:,}{self.colors.RESET}\n")
        result.append(f"   🔸 Max Depth: {self.colors.BRIGHT_WHITE}"
                      f"{metrics['max_depth']}{self.colors.RESET}\n")
        result.append(f"   🔸 Objects: {self.colors.BRIGHT_WHITE}"
                      f"{metrics['total_objects']:,}{self.colors.RESET}\n")
        result.append(f"   🔸 Arrays: {self.colors.BRIGHT_WHITE}"
                      f"{metrics['total_arrays']:,}{self.colors.RESET}\n")
        result.append(f"   🔸 Values: {self.colors.BRIGHT_WHITE}"
                      f"{metrics['total_values']:,}{self.colors.RESET}\n")
        result.append(f"   🔸 Complexity: {self.colors.BRIGHT_WHITE}"
                      f"{metrics['complexity_score']:.1f}{self.colors.RESET}\n")
        result.append("\n")

        # Data type distribution
        if metrics['data_types']:
            result.append(f"{self.colors.YELLOW}📋 DATA TYPE DISTRIBUTION:"
                          f"{self.colors.RESET}\n")

            total_vals = sum(metrics['data_types'].values())
            for data_type, count in sorted(metrics['data_types'].items(),
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / total_vals) * 100
                bar_length = min(25, int(percentage / 2))
                bar = "█" * bar_length + "░" * (25 - bar_length)

                result.append(f"   {data_type:>10}: {self.colors.BRIGHT_GREEN}"
                              f"{bar}{self.colors.RESET} {count:,} ({percentage:.1f}%)\n")
            result.append("\n")

        # Alien data analysis
        if metrics['alien_data_detected']:
            result.append(f"{self.colors.BRIGHT_MAGENTA}👽 ALIEN MATHEMATICS "
                          f"ANALYSIS:{self.colors.RESET}\n")
            result.append(f"   🔸 Alien Data: {self.colors.BRIGHT_GREEN}DETECTED"
                          f"{self.colors.RESET}\n")

            if metrics['consciousness_levels']:
                result.append(f"   🔸 Max Consciousness: {self.colors.BRIGHT_CYAN}"
                              f"{metrics['max_consciousness']:.3f}{self.colors.RESET}\n")
                result.append(f"   🔸 Avg Consciousness: {self.colors.BRIGHT_CYAN}"
                              f"{metrics['avg_consciousness']:.3f}{self.colors.RESET}\n")

            result.append(f"   🔸 Alien Properties: {self.colors.BRIGHT_WHITE}"
                          f"{len(metrics['alien_keywords'])}{self.colors.RESET}\n")

            # Show alien keywords by category
            for keyword in metrics['alien_keywords'][:8]:
                result.append(f"      • {self.colors.CYAN}{keyword}"
                              f"{self.colors.RESET}\n")
            if len(metrics['alien_keywords']) > 8:
                result.append(f"      • {self.colors.WHITE}... and "
                              f"{len(metrics['alien_keywords']) - 8} more"
                              f"{self.colors.RESET}\n")

        return "".join(result)

    def create_alien_dashboard(self, data: Any) -> str:
        """Specialized alien mathematics dashboard"""

        metrics = self.analyze_structure(data)

        result = []
        result.append(f"{self.colors.BRIGHT_CYAN}🛸 ALIEN MATHEMATICS "
                      f"DASHBOARD 🛸{self.colors.RESET}\n")
        result.append("=" * 70 + "\n\n")

        # Consciousness analysis
        if metrics['consciousness_levels']:
            max_consciousness = metrics['max_consciousness']
            avg_consciousness = metrics['avg_consciousness']

            # Progress bars
            max_bar_length = int(max_consciousness * 35)
            max_bar = "█" * max_bar_length + "░" * (35 - max_bar_length)

            avg_bar_length = int(avg_consciousness * 35)
            avg_bar = "█" * avg_bar_length + "░" * (35 - avg_bar_length)

            result.append(f"{self.colors.BRIGHT_MAGENTA}🧠 CONSCIOUSNESS "
                          f"ANALYSIS:{self.colors.RESET}\n")
            result.append(f"   Maximum: {self.colors.BRIGHT_GREEN}{max_bar}"
                          f"{self.colors.RESET} {max_consciousness:.3f}\n")
            result.append(f"   Average: {self.colors.BRIGHT_YELLOW}{avg_bar}"
                          f"{self.colors.RESET} {avg_consciousness:.3f}\n")

            # Consciousness status
            if max_consciousness >= 1.0:
                status = f"{self.colors.BRIGHT_GREEN}COSMIC MASTERY ACHIEVED"
            elif max_consciousness >= 0.9:
                status = f"{self.colors.BRIGHT_GREEN}TRANSCENDENT AWARENESS"
            elif max_consciousness >= 0.7:
                status = f"{self.colors.BRIGHT_YELLOW}ADVANCED CONSCIOUSNESS"
            elif max_consciousness >= 0.5:
                status = f"{self.colors.YELLOW}DEVELOPING AWARENESS"
            else:
                status = f"{self.colors.WHITE}BASIC CONSCIOUSNESS"

            result.append(f"   Status: {status}{self.colors.RESET}\n\n")

        # Complexity visualization
        complexity = metrics['complexity_score']
        complexity_normalized = min(complexity / 100, 1.0)
        complexity_bar_length = int(complexity_normalized * 35)
        complexity_bar = "█" * complexity_bar_length + \
            "░" * (35 - complexity_bar_length)

        result.append(f"{self.colors.BRIGHT_GREEN}🌀 COMPLEXITY ANALYSIS:"
                      f"{self.colors.RESET}\n")
        result.append(f"   Score: {self.colors.BRIGHT_CYAN}{complexity_bar}"
                      f"{self.colors.RESET} {complexity:.1f}\n")

        if complexity > 80:
            comp_level = f"{self.colors.BRIGHT_RED}HYPERDIMENSIONAL"
        elif complexity > 50:
            comp_level = f"{self.colors.BRIGHT_YELLOW}ADVANCED"
        elif complexity > 25:
            comp_level = f"{self.colors.YELLOW}INTERMEDIATE"
        else:
            comp_level = f"{self.colors.WHITE}BASIC"

        result.append(f"   Level: {comp_level}{self.colors.RESET}\n\n")

        # Alien properties categorization
        if metrics['alien_keywords']:
            result.append(f"{self.colors.BRIGHT_BLUE}⚛️ QUANTUM PROPERTIES:"
                          f"{self.colors.RESET}\n")

            # Categorize keywords
            categories = {
                '🧠 Consciousness': ['consciousness', 'telepathic', 'mind', 'astral'],
                '⚛️ Quantum': ['quantum', 'algorithm', 'circuit', 'entanglement'],
                '🌌 Galactic': ['galactic', 'cosmic', 'stellar', 'alien'],
                '🔮 Dimensional': ['dimension', 'reality', 'spacetime', 'portal'],
                '🌟 Mystical': ['mandala', 'crystal', 'sacred', 'divine']
            }

            for category, terms in categories.items():
                matching_keywords = [k for k in metrics['alien_keywords']
                                     if any(term in str(k).lower() for term in terms)]
                if matching_keywords:
                    result.append(f"   {category}: {self.colors.CYAN}"
                                  f"{len(matching_keywords)}{self.colors.RESET}\n")

            result.append("\n")

        # Data insights
        result.append(
            f"{self.colors.BRIGHT_YELLOW}💡 INSIGHTS:{self.colors.RESET}\n")

        if metrics['max_depth'] > 6:
            result.append(f"   • {self.colors.CYAN}Deep nested structure "
                          f"(depth: {metrics['max_depth']}){self.colors.RESET}\n")

        if metrics['total_arrays'] > metrics['total_objects']:
            result.append(f"   • {self.colors.CYAN}Array-heavy data structure"
                          f"{self.colors.RESET}\n")

        if metrics['complexity_score'] > 50:
            result.append(f"   • {self.colors.CYAN}High complexity alien mathematics"
                          f"{self.colors.RESET}\n")

        if len(metrics['alien_keywords']) > 10:
            result.append(f"   • {self.colors.CYAN}Rich alien mathematics content"
                          f"{self.colors.RESET}\n")

        return "".join(result)

    def create_summary_view(self, data: Any, filename: str = "") -> str:
        """Create quick summary view"""

        metrics = self.analyze_structure(data)

        result = []
        result.append(f"{self.colors.BRIGHT_CYAN}📋 QUICK SUMMARY"
                      f"{self.colors.RESET}\n")
        result.append("=" * 40 + "\n")

        if filename:
            result.append(f"📂 File: {filename}\n")

        result.append(f"📊 Structure: {metrics['total_objects']} objects, "
                      f"{metrics['total_arrays']} arrays\n")
        result.append(f"🔍 Depth: {metrics['max_depth']} levels\n")
        result.append(f"🔢 Total Keys: {metrics['total_keys']:,}\n")

        if metrics['alien_data_detected']:
            result.append(f"👽 Alien Data: {self.colors.BRIGHT_GREEN}YES"
                          f"{self.colors.RESET} ({len(metrics['alien_keywords'])} properties)\n")

            if metrics['max_consciousness'] > 0:
                result.append(f"🧠 Consciousness: {self.colors.BRIGHT_MAGENTA}"
                              f"{metrics['max_consciousness']:.3f}{self.colors.RESET}\n")
        else:
            result.append(
                f"👽 Alien Data: {self.colors.WHITE}No{self.colors.RESET}\n")

        result.append(f"🌀 Complexity: {metrics['complexity_score']:.1f}\n")

        return "".join(result)

    def visualize_all_modes(self, data: Any, filename: str = "") -> None:
        """Show all visualization modes"""

        print(f"{self.colors.BRIGHT_CYAN}🛸📊 COMPLETE JSON VISUALIZATION 📊🛸"
              f"{self.colors.RESET}")
        print("=" * 70)
        print()

        # Summary
        print("📋 SUMMARY:")
        print("-" * 20)
        print(self.create_summary_view(data, filename))
        print()

        # Tree view (limited depth for readability)
        print("🌳 TREE VIEW:")
        print("-" * 20)
        tree_view = self.visualize_tree(data, max_depth=5)

        # Limit output length
        if len(tree_view) > 3000:
            lines = tree_view.split('\n')
            displayed_lines = lines[:60]  # Show first 60 lines
            tree_view = '\n'.join(displayed_lines) + \
                f"\n... (showing first 60 lines of {len(lines)})"

        print(tree_view)
        print()

        # Statistics
        print("📊 STATISTICS:")
        print("-" * 20)
        print(self.create_statistics_view(data))
        print()

        # Alien dashboard
        print("🛸 ALIEN DASHBOARD:")
        print("-" * 20)
        print(self.create_alien_dashboard(data))


def demo_with_files():
    """Demonstrate with actual JSON files"""

    print("🛸📊" * 35)
    print("🌟 ULTIMATE JSON VISUALIZATION ENGINE DEMO 🌟")
    print("🛸📊" * 35)
    print("Beautiful visualization of alien mathematics JSON data!")
    print()

    visualizer = UltimateJSONVisualizer()

    # Find JSON files
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]

    if not json_files:
        print("ℹ️ No JSON files found in current directory")
        return

    # Prioritize alien mathematics files
    alien_files = [f for f in json_files if any(word in f.lower()
                                                for word in ['alien', 'quantum', 'creative', 'consciousness', 'session'])]

    demo_file = alien_files[0] if alien_files else json_files[0]

    print(f"🔍 Demonstrating with: {self.colors.BRIGHT_CYAN}{demo_file}"
          f"{self.colors.RESET}")
    print()

    # Load and visualize
    data = visualizer.load_json_file(demo_file)

    if data:
        visualizer.visualize_all_modes(data, demo_file)

    print()
    print("🌟" * 70)
    print("✨ ULTIMATE JSON VISUALIZATION COMPLETE! ✨")
    print("🛸 Ready to visualize any alien mathematics JSON data!")
    print("📊 Multiple visualization modes: tree, stats, alien dashboard!")
    print("🧠 Specialized for consciousness and quantum data analysis!")
    print("🌟" * 70)


def visualize_specific_file(filename: str):
    """Visualize a specific JSON file"""

    print(f"🛸 VISUALIZING: {filename}")
    print("=" * 50)

    visualizer = UltimateJSONVisualizer()
    data = visualizer.load_json_file(filename)

    if data:
        visualizer.visualize_all_modes(data, filename)
    else:
        print("❌ Failed to load or parse JSON file")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Visualize specific file
        filename = sys.argv[1]
        visualize_specific_file(filename)
    else:
        # Run demo with available files
        demo_with_files()
