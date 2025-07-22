#!/usr/bin/env python3
"""
🌍👽 ENHANCED ALIEN MATHEMATICS WORLD EXPLORER 👽🌍
================================================
Advanced exploration system for alien mathematics generated worlds!

🎯 FEATURES:
✨ Interactive world exploration with detailed analysis
🌟 Advanced civilization intelligence reports  
💎 Resource distribution mapping and analysis
⚛️ Quantum property visualization and resonance analysis
🌀 Reality coherence and mathematical harmony tracking
🎨 Beautiful ASCII world visualizations with legends
📊 Comparative world statistics and rankings
🔍 Deep-dive exploration of individual worlds
💫 Interdimensional portal detection and mapping
🧠 Consciousness field analysis and evolution tracking
📈 Export capabilities for world data and reports
🎮 Interactive menu system for seamless navigation

Explore your procedural multiverse powered by extraterrestrial mathematics!
"""

import json
import glob
import math
import random
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class ExplorationMode(Enum):
    """Different exploration modes for world analysis"""
    OVERVIEW = "overview_mode"
    DETAILED = "detailed_analysis"
    CIVILIZATION = "civilization_focus"
    RESOURCES = "resource_mapping"
    QUANTUM = "quantum_analysis"
    COMPARISON = "world_comparison"
    INTERACTIVE = "interactive_exploration"


class WorldRanking(Enum):
    """World ranking categories"""
    MOST_ADVANCED = "most_technologically_advanced"
    HIGHEST_CONSCIOUSNESS = "highest_consciousness_level"
    MOST_RESOURCES = "richest_in_resources"
    REALITY_STABLE = "most_reality_stable"
    QUANTUM_RESONANT = "highest_quantum_resonance"
    ALIEN_INFLUENCED = "most_alien_influenced"
    MATHEMATICALLY_HARMONIC = "most_mathematically_harmonic"


@dataclass
class WorldAnalysis:
    """Comprehensive world analysis results"""
    world_name: str
    world_type: str
    technological_rating: float
    consciousness_rating: float
    resource_richness: float
    quantum_potential: float
    reality_stability: float
    alien_influence: float
    mathematical_harmony: float
    civilization_diversity: float
    exploration_difficulty: float
    discovery_potential: float
    overall_score: float


@dataclass
class CivilizationIntelReport:
    """Detailed civilization intelligence report"""
    name: str
    type: str
    population: int
    tech_level: float
    quantum_awareness: float
    dimensional_access: int
    culture_traits: List[str]
    threat_level: str
    cooperation_potential: str
    unique_abilities: List[str]
    communication_methods: List[str]
    trade_opportunities: List[str]


class EnhancedAlienWorldExplorer:
    """Advanced alien mathematics world exploration system"""

    def __init__(self):
        self.worlds = []
        self.world_analyses = []
        self.exploration_history = []
        self.current_mode = ExplorationMode.OVERVIEW
        self.load_all_worlds()
        self.analyze_all_worlds()

    def load_all_worlds(self):
        """Load all generated world files with enhanced error handling"""
        print("🌍 Loading alien mathematics worlds...")

        world_files = glob.glob("world_world_*.json")

        if not world_files:
            print("❌ No world files found! Make sure you're in the correct directory.")
            print("   Expected files like: world_world_20250721_*.json")
            return

        successful_loads = 0
        for file_path in world_files:
            try:
                with open(file_path, 'r') as f:
                    world_data = json.load(f)
                    self.worlds.append(world_data)
                    successful_loads += 1
                    print(f"   ✅ {world_data['world_info']['name']}")
            except Exception as e:
                print(f"   ❌ Failed to load {file_path}: {e}")

        print(
            f"🎯 Successfully loaded {successful_loads}/{len(world_files)} worlds")
        print()

    def analyze_all_worlds(self):
        """Perform comprehensive analysis of all worlds"""
        print("🔬 Analyzing world characteristics...")

        for world in self.worlds:
            analysis = self.analyze_single_world(world)
            self.world_analyses.append(analysis)

        print(f"✅ Analysis complete for {len(self.world_analyses)} worlds")
        print()

    def analyze_single_world(self, world: Dict[str, Any]) -> WorldAnalysis:
        """Perform detailed analysis of a single world"""
        info = world['world_info']
        stats = world['statistics']
        civilizations = world.get('civilizations', [])

        # Calculate technological rating
        avg_tech = sum(civ['technology_level']
                       for civ in civilizations) / max(1, len(civilizations))
        tech_rating = (avg_tech + stats['average_consciousness_level']) / 2

        # Calculate consciousness rating
        consciousness_rating = stats['average_consciousness_level']

        # Calculate resource richness
        # Normalize to 0-1
        resource_richness = stats['resource_deposits'] / 150.0

        # Calculate quantum potential
        quantum_potential = (stats['average_quantum_resonance'] +
                             sum(civ['quantum_awareness'] for civ in civilizations) / max(1, len(civilizations))) / 2

        # Calculate civilization diversity
        civ_types = set(civ['type'] for civ in civilizations)
        diversity = len(civ_types) / max(1, len(civilizations))

        # Calculate exploration difficulty (higher = more challenging/rewarding)
        difficulty = (1 - stats['average_reality_stability'] +
                      info['alien_influence'] +
                      stats.get('interdimensional_portals', 0) / 10) / 3

        # Calculate discovery potential
        discovery_potential = (info['mathematical_harmony'] +
                               quantum_potential +
                               diversity) / 3

        # Calculate overall score
        overall_score = (tech_rating + consciousness_rating + resource_richness +
                         quantum_potential + info['reality_coherence'] +
                         info['mathematical_harmony']) / 6

        return WorldAnalysis(
            world_name=info['name'],
            world_type=info['world_type'],
            technological_rating=tech_rating,
            consciousness_rating=consciousness_rating,
            resource_richness=resource_richness,
            quantum_potential=quantum_potential,
            reality_stability=stats['average_reality_stability'],
            alien_influence=info['alien_influence'],
            mathematical_harmony=info['mathematical_harmony'],
            civilization_diversity=diversity,
            exploration_difficulty=difficulty,
            discovery_potential=discovery_potential,
            overall_score=overall_score
        )

    def create_civilization_intel_report(self, civilization: Dict[str, Any]) -> CivilizationIntelReport:
        """Generate detailed intelligence report for a civilization"""
        # Determine threat level
        if civilization['technology_level'] > 0.8 and civilization['quantum_awareness'] > 0.5:
            threat_level = "🔴 HIGH - Advanced quantum capabilities"
        elif civilization['technology_level'] > 0.6:
            threat_level = "🟡 MODERATE - Technologically capable"
        else:
            threat_level = "🟢 LOW - Peaceful or primitive"

        # Determine cooperation potential
        if 'Transcendent Awareness' in civilization['culture_traits']:
            cooperation = "🌟 EXCELLENT - Enlightened beings, high cooperation potential"
        elif 'Advanced Technology' in civilization['culture_traits']:
            cooperation = "⚡ GOOD - Technologically advanced, trade opportunities"
        elif 'Quantum Consciousness' in civilization['culture_traits']:
            cooperation = "🧠 SPECIALIZED - Quantum-aware, research partnerships possible"
        else:
            cooperation = "🤝 STANDARD - Normal diplomatic relations"

        # Determine unique abilities
        unique_abilities = []
        if civilization['dimensional_access'] > 8:
            unique_abilities.append("🌀 Interdimensional Travel")
        if civilization['quantum_awareness'] > 0.7:
            unique_abilities.append("⚛️ Quantum Manipulation")
        if 'Consciousness Collective' in civilization['type']:
            unique_abilities.append("🧠 Hive Mind Capabilities")
        if civilization['technology_level'] > 0.9:
            unique_abilities.append("🚀 Reality Engineering")

        # Communication methods
        communication = []
        if civilization['quantum_awareness'] > 0.5:
            communication.append("📡 Quantum Entanglement Channels")
        if 'Transcendent Awareness' in civilization['culture_traits']:
            communication.append("🧠 Direct Consciousness Interface")
        if civilization['dimensional_access'] > 5:
            communication.append("🌀 Dimensional Broadcasting")
        communication.append("📻 Standard Electromagnetic Signals")

        # Trade opportunities
        trade_opportunities = []
        if civilization['technology_level'] > 0.7:
            trade_opportunities.append("⚗️ Advanced Technology Exchange")
        if civilization['quantum_awareness'] > 0.6:
            trade_opportunities.append("⚛️ Quantum Knowledge Sharing")
        if 'Interdimensional Travel' in civilization['culture_traits']:
            trade_opportunities.append("🌀 Dimensional Transport Services")
        trade_opportunities.append("💎 Resource Trading")
        trade_opportunities.append("📚 Cultural Knowledge Exchange")

        return CivilizationIntelReport(
            name=civilization['name'],
            type=civilization['type'],
            population=civilization['population'],
            tech_level=civilization['technology_level'],
            quantum_awareness=civilization['quantum_awareness'],
            dimensional_access=civilization['dimensional_access'],
            culture_traits=civilization['culture_traits'],
            threat_level=threat_level,
            cooperation_potential=cooperation,
            unique_abilities=unique_abilities,
            communication_methods=communication,
            trade_opportunities=trade_opportunities
        )

    def create_enhanced_ascii_visualization(self, world: Dict[str, Any]) -> str:
        """Create detailed ASCII visualization of the world"""
        size = world['world_info']['size']
        width, height = size[0], size[1]
        civilizations = world.get('civilizations', [])

        # Create enhanced map grid
        display_width = min(50, width)
        display_height = min(25, height)

        map_grid = []
        elevation_grid = []

        # Generate terrain based on world properties
        for y in range(display_height):
            row = []
            elev_row = []
            for x in range(display_width):
                # Calculate position-based properties using alien math
                pos_x = x / display_width
                pos_y = y / display_height

                # Use alien constants for terrain generation
                arcturian_influence = math.sin(
                    pos_x * 7.7777777) * math.cos(pos_y * 7.7777777)
                pleiadian_influence = math.sin(
                    pos_x * 2.618033989 * math.pi) * math.cos(pos_y * 2.618033989 * math.pi)

                elevation = (arcturian_influence +
                             pleiadian_influence + random.uniform(-0.3, 0.3)) / 2
                elev_row.append(elevation)

                # Determine terrain based on world type and elevation
                world_type = world['world_info']['world_type']

                if world_type == "Terrestrial":
                    if elevation > 0.5:
                        terrain = '🏔️'  # Mountains
                    elif elevation > 0.2:
                        terrain = '🌲'  # Forest
                    elif elevation > -0.2:
                        terrain = '🌱'  # Plains
                    else:
                        terrain = '🌊'  # Water
                elif world_type == "Interdimensional":
                    if elevation > 0.4:
                        terrain = '🌀'  # Portal zones
                    elif elevation > 0.1:
                        terrain = '✨'  # Energy fields
                    elif elevation > -0.1:
                        terrain = '🔮'  # Reality nexus
                    else:
                        terrain = '🌟'  # Stellar regions
                else:  # Alien worlds
                    if elevation > 0.4:
                        terrain = '👽'  # Alien structures
                    elif elevation > 0.1:
                        terrain = '🛸'  # Landing sites
                    elif elevation > -0.1:
                        terrain = '🌌'  # Space
                    else:
                        terrain = '⭐'  # Stars

                row.append(terrain)

            map_grid.append(row)
            elevation_grid.append(elev_row)

        # Place civilizations with enhanced markers
        for civ in civilizations:
            civ_x = int(civ['location'][0] * display_width / size[0])
            civ_y = int(civ['location'][1] * display_height / size[1])

            if 0 <= civ_x < display_width and 0 <= civ_y < display_height:
                if civ['type'] == "Consciousness Collective":
                    if civ['quantum_awareness'] > 0.7:
                        map_grid[civ_y][civ_x] = '🧠'  # Advanced consciousness
                    else:
                        map_grid[civ_y][civ_x] = '💭'  # Basic consciousness
                elif civ['type'] == "Quantum Civilization":
                    map_grid[civ_y][civ_x] = '⚛️'  # Quantum mastery
                elif civ['type'] == "Interdimensional Beings":
                    map_grid[civ_y][civ_x] = '👽'  # Alien beings
                elif civ['technology_level'] > 0.8:
                    map_grid[civ_y][civ_x] = '🏛️'  # Advanced civilization
                else:
                    map_grid[civ_y][civ_x] = '🏘️'  # Standard civilization

        # Add resource indicators
        resource_count = world['statistics']['resource_deposits']
        if resource_count > 100:
            # Add some resource markers
            for _ in range(min(8, resource_count // 20)):
                rx = random.randint(0, display_width - 1)
                ry = random.randint(0, display_height - 1)
                if map_grid[ry][rx] in ['🌱', '🌲', '🌊', '✨', '🌌']:
                    map_grid[ry][rx] = '💎'  # Resource deposit

        # Add portal indicators for interdimensional worlds
        portal_count = world['statistics'].get('interdimensional_portals', 0)
        if portal_count > 0:
            for _ in range(min(portal_count, 5)):
                px = random.randint(0, display_width - 1)
                py = random.randint(0, display_height - 1)
                if map_grid[py][px] not in ['🧠', '⚛️', '👽', '🏛️', '🏘️']:
                    map_grid[py][px] = '🌀'  # Portal

        # Create the visualization string
        visualization = []
        visualization.append("🗺️ ENHANCED WORLD MAP:")
        visualization.append("=" * 60)

        for row in map_grid:
            visualization.append("   " + "".join(row))

        visualization.append("")
        visualization.append("🗺️ ENHANCED MAP LEGEND:")
        visualization.append(
            "   🧠 Advanced Consciousness  💭 Basic Consciousness  ⚛️ Quantum Civilization")
        visualization.append(
            "   👽 Interdimensional Beings  🏛️ Advanced Civilization  🏘️ Standard Civilization")
        visualization.append(
            "   🏔️ Mountains  🌲 Forests  🌱 Plains  🌊 Water  💎 Resources")
        visualization.append(
            "   🌀 Portals  ✨ Energy Fields  🔮 Reality Nexus  🌟 Stellar Regions")
        visualization.append("")

        return "\n".join(visualization)

    def show_world_rankings(self, ranking_type: WorldRanking):
        """Display world rankings by different criteria"""
        print(
            f"🏆 WORLD RANKINGS: {ranking_type.value.replace('_', ' ').title()}")
        print("=" * 60)

        if ranking_type == WorldRanking.MOST_ADVANCED:
            sorted_worlds = sorted(
                self.world_analyses, key=lambda w: w.technological_rating, reverse=True)
            for i, analysis in enumerate(sorted_worlds[:10], 1):
                print(f"   {i:2d}. {analysis.world_name}")
                print(
                    f"       🚀 Tech Rating: {analysis.technological_rating:.3f}")
                print(
                    f"       🧠 Consciousness: {analysis.consciousness_rating:.3f}")
                print()

        elif ranking_type == WorldRanking.HIGHEST_CONSCIOUSNESS:
            sorted_worlds = sorted(
                self.world_analyses, key=lambda w: w.consciousness_rating, reverse=True)
            for i, analysis in enumerate(sorted_worlds[:10], 1):
                print(f"   {i:2d}. {analysis.world_name}")
                print(
                    f"       🧠 Consciousness: {analysis.consciousness_rating:.3f}")
                print(
                    f"       ⚛️ Quantum Potential: {analysis.quantum_potential:.3f}")
                print()

        elif ranking_type == WorldRanking.MOST_RESOURCES:
            sorted_worlds = sorted(
                self.world_analyses, key=lambda w: w.resource_richness, reverse=True)
            for i, analysis in enumerate(sorted_worlds[:10], 1):
                print(f"   {i:2d}. {analysis.world_name}")
                print(
                    f"       💎 Resource Richness: {analysis.resource_richness:.3f}")
                print(
                    f"       🌟 Discovery Potential: {analysis.discovery_potential:.3f}")
                print()

        elif ranking_type == WorldRanking.QUANTUM_RESONANT:
            sorted_worlds = sorted(
                self.world_analyses, key=lambda w: w.quantum_potential, reverse=True)
            for i, analysis in enumerate(sorted_worlds[:10], 1):
                print(f"   {i:2d}. {analysis.world_name}")
                print(
                    f"       ⚛️ Quantum Potential: {analysis.quantum_potential:.3f}")
                print(
                    f"       🔢 Mathematical Harmony: {analysis.mathematical_harmony:.3f}")
                print()

        print()

    def explore_world_detailed(self, world_index: int):
        """Detailed exploration of a specific world"""
        if world_index < 0 or world_index >= len(self.worlds):
            print("❌ Invalid world index!")
            return

        world = self.worlds[world_index]
        analysis = self.world_analyses[world_index]

        print("🌍" * 80)
        print(f"🔍 DETAILED EXPLORATION: {world['world_info']['name']}")
        print("🌍" * 80)

        # Basic info
        info = world['world_info']
        stats = world['statistics']

        print("📊 WORLD OVERVIEW:")
        print(f"   🌐 Type: {info['world_type']}")
        print(f"   📏 Size: {info['size'][0]}x{info['size'][1]}")
        print(f"   🎭 Reality Coherence: {info['reality_coherence']:.1%}")
        print(f"   🔢 Mathematical Harmony: {info['mathematical_harmony']:.1%}")
        print(f"   👽 Alien Influence: {info['alien_influence']:.1%}")
        print(f"   ⭐ Overall Score: {analysis.overall_score:.3f}/1.000")
        print()

        # Environmental data
        print("🌍 ENVIRONMENTAL ANALYSIS:")
        print(f"   🏔️ Elevation: {stats['average_elevation']:.3f}")
        print(f"   🌡️ Temperature: {stats['average_temperature']:.3f}")
        print(f"   💧 Humidity: {stats['average_humidity']:.3f}")
        print(
            f"   ⚛️ Quantum Resonance: {stats['average_quantum_resonance']:.3f}")
        print(
            f"   🌀 Reality Stability: {stats['average_reality_stability']:.3f}")
        print()

        # Resources and special features
        print("💎 RESOURCES & SPECIAL FEATURES:")
        print(f"   💰 Resource Deposits: {stats['resource_deposits']}")
        print(f"   🧠 Consciousness Fields: {stats['consciousness_fields']}")
        if stats.get('interdimensional_portals', 0) > 0:
            print(
                f"   🌀 Interdimensional Portals: {stats['interdimensional_portals']} ⚡")
        print()

        # Civilization intelligence reports
        civilizations = world.get('civilizations', [])
        if civilizations:
            print(
                f"🏛️ CIVILIZATION INTELLIGENCE REPORTS ({len(civilizations)} civilizations):")
            print("=" * 60)

            for i, civ in enumerate(civilizations, 1):
                intel = self.create_civilization_intel_report(civ)

                print(f"📋 CIVILIZATION #{i}: {intel.name}")
                print(f"   🏷️ Type: {intel.type}")
                print(f"   👥 Population: {intel.population:,}")
                print(f"   ⚗️ Technology Level: {intel.tech_level:.1%}")
                print(
                    f"   ⚛️ Quantum Awareness: {intel.quantum_awareness:.1%}")
                print(f"   🌌 Dimensional Access: {intel.dimensional_access}D")
                print(f"   🎭 Culture: {', '.join(intel.culture_traits)}")
                print(f"   ⚠️ Threat Level: {intel.threat_level}")
                print(f"   🤝 Cooperation: {intel.cooperation_potential}")

                if intel.unique_abilities:
                    print(
                        f"   ⭐ Unique Abilities: {', '.join(intel.unique_abilities)}")

                print(
                    f"   📡 Communication: {', '.join(intel.communication_methods[:2])}")
                print(
                    f"   💼 Trade Opportunities: {', '.join(intel.trade_opportunities[:2])}")
                print()

        # ASCII visualization
        visualization = self.create_enhanced_ascii_visualization(world)
        print(visualization)

        # Analysis summary
        print("🔬 EXPLORATION ANALYSIS:")
        print(
            f"   🚀 Technological Rating: {analysis.technological_rating:.3f}")
        print(
            f"   🧠 Consciousness Rating: {analysis.consciousness_rating:.3f}")
        print(f"   💎 Resource Richness: {analysis.resource_richness:.3f}")
        print(f"   ⚛️ Quantum Potential: {analysis.quantum_potential:.3f}")
        print(
            f"   🎯 Exploration Difficulty: {analysis.exploration_difficulty:.3f}")
        print(f"   🌟 Discovery Potential: {analysis.discovery_potential:.3f}")
        print()

        # Recommendations
        print("💡 EXPLORATION RECOMMENDATIONS:")
        if analysis.technological_rating > 0.8:
            print(
                "   ⚡ HIGH PRIORITY: Advanced civilizations detected - establish diplomatic contact")
        if analysis.quantum_potential > 0.7:
            print(
                "   ⚛️ RESEARCH FOCUS: High quantum potential - ideal for quantum experiments")
        if analysis.resource_richness > 0.8:
            print(
                "   💎 MINING OPPORTUNITY: Rich resource deposits - consider extraction missions")
        if analysis.discovery_potential > 0.8:
            print(
                "   🔬 SCIENTIFIC VALUE: High discovery potential - send research expeditions")
        if stats.get('interdimensional_portals', 0) > 0:
            print(
                "   🌀 PORTAL ACCESS: Interdimensional gateways detected - extreme caution advised")

        print()
        print("🌟" * 80)

    def compare_worlds(self, world_indices: List[int]):
        """Compare multiple worlds side by side"""
        if len(world_indices) < 2:
            print("❌ Need at least 2 worlds to compare!")
            return

        valid_indices = [i for i in world_indices if 0 <= i < len(self.worlds)]
        if len(valid_indices) < 2:
            print("❌ Invalid world indices!")
            return

        print("🔄 WORLD COMPARISON ANALYSIS")
        print("=" * 80)

        compared_worlds = [(self.worlds[i], self.world_analyses[i])
                           for i in valid_indices]

        # Comparison table header
        print(f"{'Metric':<25}", end="")
        for world, _ in compared_worlds:
            name = world['world_info']['name'][:15]
            print(f"{name:>15}", end="")
        print()
        print("-" * (25 + 15 * len(compared_worlds)))

        # Comparison metrics
        metrics = [
            ("Technological Rating", lambda w,
             a: f"{a.technological_rating:.3f}"),
            ("Consciousness Level", lambda w,
             a: f"{a.consciousness_rating:.3f}"),
            ("Resource Richness", lambda w, a: f"{a.resource_richness:.3f}"),
            ("Quantum Potential", lambda w, a: f"{a.quantum_potential:.3f}"),
            ("Reality Stability", lambda w, a: f"{a.reality_stability:.3f}"),
            ("Alien Influence", lambda w, a: f"{a.alien_influence:.3f}"),
            ("Math Harmony", lambda w, a: f"{a.mathematical_harmony:.3f}"),
            ("Civilizations", lambda w,
             a: f"{w['statistics']['civilization_count']:>3d}"),
            ("Resources", lambda w,
             a: f"{w['statistics']['resource_deposits']:>3d}"),
            ("Portals", lambda w,
             a: f"{w['statistics'].get('interdimensional_portals', 0):>3d}"),
            ("Overall Score", lambda w, a: f"{a.overall_score:.3f}")
        ]

        for metric_name, metric_func in metrics:
            print(f"{metric_name:<25}", end="")
            for world, analysis in compared_worlds:
                value = metric_func(world, analysis)
                print(f"{value:>15}", end="")
            print()

        print()

        # Winner analysis
        print("🏆 COMPARISON WINNERS:")

        categories = [
            ("Most Technological", lambda a: a.technological_rating),
            ("Highest Consciousness", lambda a: a.consciousness_rating),
            ("Richest Resources", lambda a: a.resource_richness),
            ("Best Quantum Potential", lambda a: a.quantum_potential),
            ("Most Stable Reality", lambda a: a.reality_stability),
            ("Highest Overall Score", lambda a: a.overall_score)
        ]

        for category_name, category_func in categories:
            best_analysis = max(
                compared_worlds, key=lambda x: category_func(x[1]))[1]
            print(f"   {category_name}: {best_analysis.world_name}")

        print()

    def interactive_exploration_menu(self):
        """Interactive menu system for world exploration"""
        while True:
            print("\n🌍👽 ALIEN WORLD EXPLORER - MAIN MENU 👽🌍")
            print("=" * 60)
            print("1. 📊 World Overview & Statistics")
            print("2. 🔍 Detailed World Exploration")
            print("3. 🏆 World Rankings")
            print("4. 🔄 Compare Worlds")
            print("5. 🎯 Find Best Worlds")
            print("6. 📈 Export World Data")
            print("7. 🌟 Generate Exploration Report")
            print("8. ❌ Exit Explorer")
            print()

            choice = input("👽 Select option (1-8): ").strip()

            if choice == "1":
                self.show_world_overview()
            elif choice == "2":
                self.interactive_world_selection()
            elif choice == "3":
                self.interactive_rankings_menu()
            elif choice == "4":
                self.interactive_world_comparison()
            elif choice == "5":
                self.find_best_worlds()
            elif choice == "6":
                self.export_world_data()
            elif choice == "7":
                self.generate_exploration_report()
            elif choice == "8":
                print("🌟 Thank you for exploring the alien mathematics multiverse!")
                break
            else:
                print("❌ Invalid choice! Please select 1-8.")

    def show_world_overview(self):
        """Show overview of all worlds"""
        print("\n📊 ALIEN MATHEMATICS WORLDS OVERVIEW")
        print("=" * 80)

        total_civilizations = sum(
            w['statistics']['civilization_count'] for w in self.worlds)
        total_population = sum(
            sum(civ['population'] for civ in w.get('civilizations', []))
            for w in self.worlds
        )
        total_resources = sum(w['statistics']['resource_deposits']
                              for w in self.worlds)
        total_portals = sum(w['statistics'].get(
            'interdimensional_portals', 0) for w in self.worlds)

        avg_alien_influence = sum(w['world_info']['alien_influence']
                                  for w in self.worlds) / len(self.worlds)
        avg_reality_coherence = sum(
            w['world_info']['reality_coherence'] for w in self.worlds) / len(self.worlds)
        avg_math_harmony = sum(w['world_info']['mathematical_harmony']
                               for w in self.worlds) / len(self.worlds)

        print(f"🌍 Total Worlds: {len(self.worlds)}")
        print(f"🏛️ Total Civilizations: {total_civilizations}")
        print(f"👥 Total Population: {total_population:,}")
        print(f"💎 Total Resources: {total_resources}")
        print(f"🌀 Total Portals: {total_portals}")
        print(f"👽 Avg Alien Influence: {avg_alien_influence:.1%}")
        print(f"🌀 Avg Reality Coherence: {avg_reality_coherence:.1%}")
        print(f"🔢 Avg Mathematical Harmony: {avg_math_harmony:.1%}")
        print()

        print("🌟 WORLD LIST:")
        for i, world in enumerate(self.worlds):
            analysis = self.world_analyses[i]
            info = world['world_info']
            stats = world['statistics']

            print(f"   {i+1:2d}. {info['name']} ({info['world_type']})")
            print(
                f"       ⭐ Score: {analysis.overall_score:.3f} | 🏛️ Civs: {stats['civilization_count']}")
            print(
                f"       👽 Alien: {info['alien_influence']:.1%} | ⚛️ Quantum: {stats['average_quantum_resonance']:.3f}")

            if stats.get('interdimensional_portals', 0) > 0:
                print(
                    f"       🌀 Portals: {stats['interdimensional_portals']} ✨")
            print()

    def interactive_world_selection(self):
        """Interactive world selection for detailed exploration"""
        print("\n🔍 SELECT WORLD FOR DETAILED EXPLORATION")
        print("=" * 50)

        for i, world in enumerate(self.worlds):
            print(f"   {i+1:2d}. {world['world_info']['name']}")

        print()
        try:
            choice = int(input(f"Select world (1-{len(self.worlds)}): ")) - 1
            if 0 <= choice < len(self.worlds):
                self.explore_world_detailed(choice)
                input("\nPress Enter to continue...")
            else:
                print("❌ Invalid world selection!")
        except ValueError:
            print("❌ Please enter a valid number!")

    def interactive_rankings_menu(self):
        """Interactive rankings menu"""
        print("\n🏆 WORLD RANKINGS MENU")
        print("=" * 40)
        print("1. 🚀 Most Technologically Advanced")
        print("2. 🧠 Highest Consciousness Level")
        print("3. 💎 Richest in Resources")
        print("4. 🌀 Most Reality Stable")
        print("5. ⚛️ Highest Quantum Resonance")
        print("6. 👽 Most Alien Influenced")
        print("7. 🔢 Most Mathematically Harmonic")
        print()

        choice = input("Select ranking (1-7): ").strip()
        rankings = {
            "1": WorldRanking.MOST_ADVANCED,
            "2": WorldRanking.HIGHEST_CONSCIOUSNESS,
            "3": WorldRanking.MOST_RESOURCES,
            "4": WorldRanking.REALITY_STABLE,
            "5": WorldRanking.QUANTUM_RESONANT,
            "6": WorldRanking.ALIEN_INFLUENCED,
            "7": WorldRanking.MATHEMATICALLY_HARMONIC
        }

        if choice in rankings:
            print()
            self.show_world_rankings(rankings[choice])
            input("\nPress Enter to continue...")
        else:
            print("❌ Invalid ranking selection!")

    def interactive_world_comparison(self):
        """Interactive world comparison selection"""
        print("\n🔄 WORLD COMPARISON")
        print("=" * 40)

        for i, world in enumerate(self.worlds):
            print(f"   {i+1:2d}. {world['world_info']['name']}")

        print()
        try:
            indices_input = input(
                "Enter world numbers to compare (e.g., 1,3,5): ").strip()
            indices = [int(x.strip()) - 1 for x in indices_input.split(',')]

            if len(indices) >= 2:
                print()
                self.compare_worlds(indices)
                input("\nPress Enter to continue...")
            else:
                print("❌ Need at least 2 worlds to compare!")
        except ValueError:
            print("❌ Please enter valid numbers separated by commas!")

    def find_best_worlds(self):
        """Find and display the best worlds in various categories"""
        print("\n🎯 BEST WORLDS ANALYSIS")
        print("=" * 60)

        # Best overall
        best_overall = max(self.world_analyses, key=lambda w: w.overall_score)
        print(f"🏆 BEST OVERALL WORLD: {best_overall.world_name}")
        print(f"   ⭐ Overall Score: {best_overall.overall_score:.3f}")
        print(f"   🌐 Type: {best_overall.world_type}")
        print()

        # Most promising for different activities
        best_tech = max(self.world_analyses,
                        key=lambda w: w.technological_rating)
        best_resources = max(self.world_analyses,
                             key=lambda w: w.resource_richness)
        best_quantum = max(self.world_analyses,
                           key=lambda w: w.quantum_potential)
        best_discovery = max(self.world_analyses,
                             key=lambda w: w.discovery_potential)

        print("🌟 SPECIALIZED RECOMMENDATIONS:")
        print(f"   🚀 Best for Technology Research: {best_tech.world_name}")
        print(f"   💎 Best for Resource Mining: {best_resources.world_name}")
        print(f"   ⚛️ Best for Quantum Experiments: {best_quantum.world_name}")
        print(
            f"   🔬 Best for Scientific Discovery: {best_discovery.world_name}")
        print()

        # Most dangerous/challenging
        most_dangerous = max(self.world_analyses,
                             key=lambda w: w.exploration_difficulty)
        print(f"⚠️ MOST CHALLENGING WORLD: {most_dangerous.world_name}")
        print(f"   🎯 Difficulty: {most_dangerous.exploration_difficulty:.3f}")
        print(f"   💀 High risk, high reward exploration target")
        print()

        input("Press Enter to continue...")

    def export_world_data(self):
        """Export world data to files"""
        print("\n📈 EXPORT WORLD DATA")
        print("=" * 40)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export summary
        summary_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_worlds": len(self.worlds),
                "export_type": "alien_mathematics_world_summary"
            },
            "world_analyses": [
                {
                    "world_name": analysis.world_name,
                    "world_type": analysis.world_type,
                    "technological_rating": analysis.technological_rating,
                    "consciousness_rating": analysis.consciousness_rating,
                    "resource_richness": analysis.resource_richness,
                    "quantum_potential": analysis.quantum_potential,
                    "overall_score": analysis.overall_score
                }
                for analysis in self.world_analyses
            ]
        }

        summary_filename = f"alien_world_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"✅ World summary exported to: {summary_filename}")

        # Export detailed report
        report_filename = f"alien_world_report_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("🌍👽 ALIEN MATHEMATICS WORLDS EXPLORATION REPORT 👽🌍\n")
            f.write("=" * 80 + "\n\n")

            for i, world in enumerate(self.worlds):
                analysis = self.world_analyses[i]
                f.write(f"WORLD #{i+1}: {world['world_info']['name']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Type: {world['world_info']['world_type']}\n")
                f.write(f"Overall Score: {analysis.overall_score:.3f}\n")
                f.write(
                    f"Civilizations: {world['statistics']['civilization_count']}\n")
                f.write(
                    f"Resources: {world['statistics']['resource_deposits']}\n")
                f.write(
                    f"Quantum Potential: {analysis.quantum_potential:.3f}\n")
                f.write("\n")

        print(f"✅ Detailed report exported to: {report_filename}")
        print()
        input("Press Enter to continue...")

    def generate_exploration_report(self):
        """Generate comprehensive exploration report"""
        print("\n🌟 GENERATING COMPREHENSIVE EXPLORATION REPORT...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"alien_multiverse_exploration_report_{timestamp}.txt"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("🌍👽 ALIEN MATHEMATICS MULTIVERSE EXPLORATION REPORT 👽🌍\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Explorer Version: Enhanced Alien World Explorer v2.0\n\n")

            # Executive summary
            f.write("📋 EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Worlds Discovered: {len(self.worlds)}\n")
            f.write(
                f"Average Overall Score: {sum(a.overall_score for a in self.world_analyses) / len(self.world_analyses):.3f}\n")
            f.write(
                f"Worlds with Interdimensional Access: {sum(1 for w in self.worlds if w['statistics'].get('interdimensional_portals', 0) > 0)}\n")
            f.write(
                f"Total Alien Civilizations: {sum(w['statistics']['civilization_count'] for w in self.worlds)}\n\n")

            # Top worlds
            best_worlds = sorted(
                self.world_analyses, key=lambda w: w.overall_score, reverse=True)[:5]
            f.write("🏆 TOP 5 WORLDS\n")
            f.write("-" * 20 + "\n")
            for i, world in enumerate(best_worlds, 1):
                f.write(
                    f"{i}. {world.world_name} (Score: {world.overall_score:.3f})\n")
            f.write("\n")

            # Detailed world profiles
            for i, world in enumerate(self.worlds):
                analysis = self.world_analyses[i]
                f.write(
                    f"WORLD PROFILE #{i+1}: {world['world_info']['name']}\n")
                f.write("=" * 60 + "\n")

                info = world['world_info']
                stats = world['statistics']

                f.write(f"World Type: {info['world_type']}\n")
                f.write(f"Size: {info['size'][0]}x{info['size'][1]}\n")
                f.write(f"Overall Score: {analysis.overall_score:.3f}\n")
                f.write(
                    f"Technological Rating: {analysis.technological_rating:.3f}\n")
                f.write(
                    f"Consciousness Rating: {analysis.consciousness_rating:.3f}\n")
                f.write(
                    f"Resource Richness: {analysis.resource_richness:.3f}\n")
                f.write(
                    f"Quantum Potential: {analysis.quantum_potential:.3f}\n")
                f.write(
                    f"Reality Stability: {analysis.reality_stability:.3f}\n")
                f.write(f"Alien Influence: {analysis.alien_influence:.3f}\n")
                f.write(
                    f"Mathematical Harmony: {analysis.mathematical_harmony:.3f}\n\n")

                f.write(f"Civilizations: {stats['civilization_count']}\n")
                f.write(f"Resource Deposits: {stats['resource_deposits']}\n")
                f.write(
                    f"Consciousness Fields: {stats['consciousness_fields']}\n")

                if stats.get('interdimensional_portals', 0) > 0:
                    f.write(
                        f"Interdimensional Portals: {stats['interdimensional_portals']}\n")

                f.write("\nCivilization Details:\n")
                for civ in world.get('civilizations', []):
                    f.write(f"  • {civ['name']} ({civ['type']})\n")
                    f.write(f"    Population: {civ['population']:,}\n")
                    f.write(f"    Technology: {civ['technology_level']:.1%}\n")
                    f.write(
                        f"    Quantum Awareness: {civ['quantum_awareness']:.1%}\n")
                    f.write(
                        f"    Dimensional Access: {civ['dimensional_access']}D\n")
                    f.write(
                        f"    Culture: {', '.join(civ['culture_traits'])}\n\n")

                f.write("\n" + "=" * 60 + "\n\n")

        print(f"✅ Comprehensive report generated: {report_filename}")
        print("📋 Report includes detailed profiles of all worlds and civilizations!")
        print()
        input("Press Enter to continue...")

    def run_enhanced_exploration(self):
        """Run the enhanced alien world exploration system"""
        print("🌍👽" * 30)
        print("🌌 ENHANCED ALIEN MATHEMATICS WORLD EXPLORER 🌌")
        print("🌍👽" * 30)
        print()

        if not self.worlds:
            print("❌ No alien worlds found to explore!")
            print(
                "   Make sure you have generated worlds using the alien math world generator.")
            return

        print(
            f"✅ Successfully loaded {len(self.worlds)} alien mathematics worlds!")
        print("🔬 Advanced analysis and exploration capabilities initialized!")
        print("🎯 Ready for comprehensive multiverse exploration!")
        print()

        # Show quick overview
        print("🌟 QUICK MULTIVERSE OVERVIEW:")
        best_world = max(self.world_analyses, key=lambda w: w.overall_score)
        most_advanced = max(self.world_analyses,
                            key=lambda w: w.technological_rating)
        most_resources = max(self.world_analyses,
                             key=lambda w: w.resource_richness)

        print(
            f"   🏆 Best Overall: {best_world.world_name} (Score: {best_world.overall_score:.3f})")
        print(
            f"   🚀 Most Advanced: {most_advanced.world_name} (Tech: {most_advanced.technological_rating:.3f})")
        print(
            f"   💎 Richest: {most_resources.world_name} (Resources: {most_resources.resource_richness:.3f})")

        worlds_with_portals = sum(1 for w in self.worlds if w['statistics'].get(
            'interdimensional_portals', 0) > 0)
        if worlds_with_portals > 0:
            print(
                f"   🌀 Interdimensional Access: {worlds_with_portals} worlds have active portals!")

        print()

        # Start interactive exploration
        self.interactive_exploration_menu()


def main():
    """Run the enhanced alien world explorer"""
    explorer = EnhancedAlienWorldExplorer()
    explorer.run_enhanced_exploration()


if __name__ == "__main__":
    main()
