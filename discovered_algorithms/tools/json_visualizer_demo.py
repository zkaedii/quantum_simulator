#!/usr/bin/env python3
"""
🛸📊 STANDALONE JSON VISUALIZATION DEMO 📊🛸
============================================
Beautiful JSON visualization for alien mathematics data!
"""

import json
import os
import re
import math
from datetime import datetime
from typing import Dict, List, Any, Optional


class Colors:
    """Console colors for beautiful output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Text colors
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


class JSONVisualizer:
    """Beautiful JSON visualization engine"""

    def __init__(self):
        self.colors = Colors()

    def load_json_file(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded: {filepath} ({os.path.getsize(filepath)} bytes)")
            return data
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return {}

    def get_key_color(self, key: str) -> str:
        """Get color for JSON key based on content"""
        key_lower = str(key).lower()

        if any(word in key_lower for word in ['consciousness', 'telepathic']):
            return self.colors.BRIGHT_MAGENTA
        elif any(word in key_lower for word in ['quantum', 'algorithm']):
            return self.colors.BRIGHT_BLUE
        elif any(word in key_lower for word in ['alien', 'galactic']):
            return self.colors.BRIGHT_CYAN
        elif any(word in key_lower for word in ['reality', 'dimension']):
            return self.colors.BRIGHT_GREEN
        elif any(word in key_lower for word in ['mandala', 'crystal']):
            return self.colors.BRIGHT_YELLOW
        else:
            return self.colors.GREEN

    def get_value_color(self, value: Any) -> str:
        """Get color for JSON value based on type"""
        if isinstance(value, bool):
            return self.colors.BRIGHT_YELLOW
        elif isinstance(value, int):
            return self.colors.BRIGHT_BLUE
        elif isinstance(value, float):
            return self.colors.BRIGHT_CYAN
        elif isinstance(value, str):
            return self.colors.BRIGHT_WHITE
        else:
            return self.colors.WHITE

    def visualize_tree(self, data: Any, max_depth: int = 8,
                       current_depth: int = 0, prefix: str = "") -> str:
        """Create beautiful tree visualization of JSON"""

        if current_depth > max_depth:
            return f"{prefix}🔻 [Max depth reached]\n"

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

                key_color = self.get_key_color(key)
                type_info = f" ({type(value).__name__})"

                result.append(f"{prefix}{connector}{key_color}{key}"
                              f"{self.colors.RESET}{type_info}")

                # Add value preview for simple types
                if isinstance(value, (str, int, float, bool)) and len(str(value)) < 50:
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

            # Show first few and last few items if array is long
            if len(data) > 10:
                # Show first 3 items
                for i in range(3):
                    if i < len(data):
                        connector = "├── "
                        result.append(f"{prefix}{connector}"
                                      f"{self.colors.BRIGHT_MAGENTA}[{i}]"
                                      f"{self.colors.RESET}")
                        if isinstance(data[i], (str, int, float, bool)):
                            value_color = self.get_value_color(data[i])
                            result.append(f" = {value_color}{repr(data[i])}"
                                          f"{self.colors.RESET}")
                        result.append("\n")

                        if isinstance(data[i], (dict, list)):
                            next_prefix = prefix + "│   "
                            result.append(self.visualize_tree(data[i], max_depth,
                                                              current_depth + 1, next_prefix))

                # Show ellipsis
                result.append(
                    f"{prefix}├── ... ({len(data) - 6} more items) ...\n")

                # Show last 3 items
                for i in range(len(data) - 3, len(data)):
                    is_last = (i == len(data) - 1)
                    connector = "└── " if is_last else "├── "
                    result.append(f"{prefix}{connector}"
                                  f"{self.colors.BRIGHT_MAGENTA}[{i}]"
                                  f"{self.colors.RESET}")
                    if isinstance(data[i], (str, int, float, bool)):
                        value_color = self.get_value_color(data[i])
                        result.append(f" = {value_color}{repr(data[i])}"
                                      f"{self.colors.RESET}")
                    result.append("\n")

                    if isinstance(data[i], (dict, list)):
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        result.append(self.visualize_tree(data[i], max_depth,
                                                          current_depth + 1, next_prefix))
            else:
                # Show all items
                for i, item in enumerate(data):
                    is_last = (i == len(data) - 1)
                    connector = "└── " if is_last else "├── "
                    next_prefix = prefix + ("    " if is_last else "│   ")

                    result.append(f"{prefix}{connector}"
                                  f"{self.colors.BRIGHT_MAGENTA}[{i}]"
                                  f"{self.colors.RESET}")

                    if isinstance(item, (str, int, float, bool)):
                        value_color = self.get_value_color(item)
                        result.append(f" = {value_color}{repr(item)}"
                                      f"{self.colors.RESET}")

                    result.append("\n")

                    if isinstance(item, (dict, list)):
                        result.append(self.visualize_tree(item, max_depth,
                                                          current_depth + 1, next_prefix))

        return "".join(result)

    def analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure and return metrics"""

        total_keys = 0
        max_depth = 0
        total_objects = 0
        total_arrays = 0
        total_values = 0
        data_types = {}
        alien_keywords = []
        consciousness_level = 0.0

        def analyze_recursive(obj: Any, current_depth: int = 0) -> int:
            nonlocal total_keys, max_depth, total_objects, total_arrays
            nonlocal total_values, data_types, alien_keywords, consciousness_level

            max_depth = max(max_depth, current_depth)

            if isinstance(obj, dict):
                total_objects += 1
                total_keys += len(obj)

                for key, value in obj.items():
                    # Check for alien/quantum keywords
                    key_str = str(key).lower()
                    for keyword in ['alien', 'quantum', 'consciousness',
                                    'telepathic', 'galactic', 'mandala']:
                        if keyword in key_str:
                            alien_keywords.append(key)

                    # Extract consciousness level
                    if 'consciousness' in key_str and isinstance(value, (int, float)):
                        consciousness_level = max(
                            consciousness_level, float(value))

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

        return {
            'total_keys': total_keys,
            'max_depth': max_depth,
            'total_objects': total_objects,
            'total_arrays': total_arrays,
            'total_values': total_values,
            'data_types': data_types,
            'alien_keywords': alien_keywords,
            'consciousness_level': consciousness_level,
            'alien_data_detected': len(alien_keywords) > 0
        }

    def create_statistics_view(self, data: Any) -> str:
        """Create comprehensive statistics view"""

        metrics = self.analyze_structure(data)

        result = []
        result.append(
            f"{self.colors.BRIGHT_CYAN}📊 JSON STATISTICS{self.colors.RESET}\n")
        result.append("=" * 50 + "\n\n")

        # Basic metrics
        result.append(f"{self.colors.YELLOW}📏 STRUCTURE:{self.colors.RESET}\n")
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
        result.append("\n")

        # Data types
        if metrics['data_types']:
            result.append(
                f"{self.colors.YELLOW}📋 DATA TYPES:{self.colors.RESET}\n")
            for data_type, count in sorted(metrics['data_types'].items(),
                                           key=lambda x: x[1], reverse=True):
                result.append(f"   {data_type:>10}: {self.colors.BRIGHT_GREEN}"
                              f"{count:,}{self.colors.RESET}\n")
            result.append("\n")

        # Alien data analysis
        if metrics['alien_data_detected']:
            result.append(f"{self.colors.BRIGHT_MAGENTA}👽 ALIEN ANALYSIS:"
                          f"{self.colors.RESET}\n")
            result.append(f"   🔸 Alien Data: {self.colors.BRIGHT_GREEN}DETECTED"
                          f"{self.colors.RESET}\n")
            result.append(f"   🔸 Consciousness: {self.colors.BRIGHT_CYAN}"
                          f"{metrics['consciousness_level']:.3f}{self.colors.RESET}\n")

            if metrics['alien_keywords']:
                result.append(f"   🔸 Alien Properties: {self.colors.BRIGHT_WHITE}"
                              f"{len(metrics['alien_keywords'])}{self.colors.RESET}\n")
                for keyword in metrics['alien_keywords'][:5]:
                    result.append(f"      • {self.colors.CYAN}{keyword}"
                                  f"{self.colors.RESET}\n")
                if len(metrics['alien_keywords']) > 5:
                    result.append(f"      • {self.colors.WHITE}... and "
                                  f"{len(metrics['alien_keywords']) - 5} more"
                                  f"{self.colors.RESET}\n")

        return "".join(result)

    def create_alien_dashboard(self, data: Any) -> str:
        """Create alien mathematics dashboard"""

        metrics = self.analyze_structure(data)

        result = []
        result.append(f"{self.colors.BRIGHT_CYAN}🛸 ALIEN MATHEMATICS DASHBOARD 🛸"
                      f"{self.colors.RESET}\n")
        result.append("=" * 60 + "\n\n")

        # Consciousness analysis
        if metrics['consciousness_level'] > 0:
            # Create progress bar
            level = metrics['consciousness_level']
            bar_length = int(level * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)

            result.append(f"{self.colors.BRIGHT_MAGENTA}🧠 CONSCIOUSNESS:"
                          f"{self.colors.RESET}\n")
            result.append(f"   Level: {self.colors.BRIGHT_GREEN}{bar}"
                          f"{self.colors.RESET} {level:.3f}\n")

            if level > 0.9:
                status = f"{self.colors.BRIGHT_GREEN}COSMIC MASTERY"
            elif level > 0.7:
                status = f"{self.colors.BRIGHT_YELLOW}ADVANCED AWARENESS"
            elif level > 0.5:
                status = f"{self.colors.YELLOW}DEVELOPING"
            else:
                status = f"{self.colors.WHITE}BASIC"

            result.append(f"   Status: {status}{self.colors.RESET}\n\n")

        # Complexity analysis
        complexity = (metrics['max_depth'] * 5 +
                      len(metrics['alien_keywords']) * 3 +
                      metrics['total_objects'] * 0.1)

        complexity_bar_length = min(30, int(complexity / 2))
        complexity_bar = "█" * complexity_bar_length + \
            "░" * (30 - complexity_bar_length)

        result.append(
            f"{self.colors.BRIGHT_GREEN}🌀 COMPLEXITY:{self.colors.RESET}\n")
        result.append(f"   Score: {self.colors.BRIGHT_CYAN}{complexity_bar}"
                      f"{self.colors.RESET} {complexity:.1f}\n")

        if complexity > 50:
            comp_level = f"{self.colors.BRIGHT_RED}HYPERDIMENSIONAL"
        elif complexity > 25:
            comp_level = f"{self.colors.BRIGHT_YELLOW}ADVANCED"
        else:
            comp_level = f"{self.colors.WHITE}BASIC"

        result.append(f"   Level: {comp_level}{self.colors.RESET}\n\n")

        # Alien properties
        if metrics['alien_keywords']:
            result.append(f"{self.colors.BRIGHT_BLUE}⚛️ QUANTUM PROPERTIES:"
                          f"{self.colors.RESET}\n")

            # Categorize keywords
            consciousness_keywords = [k for k in metrics['alien_keywords']
                                      if any(word in str(k).lower()
                                             for word in ['consciousness', 'telepathic'])]
            quantum_keywords = [k for k in metrics['alien_keywords']
                                if any(word in str(k).lower()
                                       for word in ['quantum', 'algorithm'])]

            if consciousness_keywords:
                result.append(f"   🧠 Consciousness: {self.colors.CYAN}"
                              f"{len(consciousness_keywords)}{self.colors.RESET}\n")

            if quantum_keywords:
                result.append(f"   ⚛️ Quantum: {self.colors.CYAN}"
                              f"{len(quantum_keywords)}{self.colors.RESET}\n")

            result.append("\n")

        return "".join(result)


def run_json_visualization_demo():
    """Run the main JSON visualization demonstration"""

    print("🛸📊" * 30)
    print("🌟 ALIEN MATHEMATICS JSON VISUALIZATION DEMO 🌟")
    print("🛸📊" * 30)
    print("Beautiful visualization of alien mathematics JSON data!")
    print()

    visualizer = JSONVisualizer()

    # Find JSON files
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]

    if not json_files:
        print("ℹ️ No JSON files found in current directory")
        return

    # Prioritize alien mathematics files
    alien_files = [f for f in json_files if any(word in f.lower()
                                                for word in ['alien', 'quantum', 'creative', 'consciousness'])]

    demo_file = alien_files[0] if alien_files else json_files[0]

    print(f"🔍 VISUALIZING: {demo_file}")
    print("=" * 60)

    # Load and visualize
    data = visualizer.load_json_file(demo_file)

    if data:
        # Tree view
        print("\n🌳 TREE VIEW:")
        print("-" * 30)
        tree_view = visualizer.visualize_tree(data, max_depth=6)
        print(tree_view[:2000])  # Show first 2000 characters
        if len(tree_view) > 2000:
            print(f"... (showing first 2000 of {len(tree_view)} characters)\n")

        # Statistics
        print("📊 STATISTICAL ANALYSIS:")
        print("-" * 30)
        stats = visualizer.create_statistics_view(data)
        print(stats)

        # Alien dashboard
        print("🛸 ALIEN MATHEMATICS DASHBOARD:")
        print("-" * 30)
        dashboard = visualizer.create_alien_dashboard(data)
        print(dashboard)

    print("🌟" * 60)
    print("✨ JSON VISUALIZATION COMPLETE! ✨")
    print("🛸 Beautiful alien mathematics data visualization achieved!")
    print("🌟" * 60)


if __name__ == "__main__":
    run_json_visualization_demo()
