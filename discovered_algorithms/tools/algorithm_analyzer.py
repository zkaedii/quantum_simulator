#!/usr/bin/env python3
"""
📊 QUANTUM ALGORITHM DEEP ANALYSIS SYSTEM
========================================

Comprehensive analysis tools for discovered quantum algorithms:
- Circuit structure analysis
- Performance benchmarking
- Quantum advantage validation
- Gate optimization analysis
- Application potential assessment
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger("AlgorithmAnalyzer")


@dataclass
class AnalysisResult:
    """Results from quantum algorithm analysis."""
    algorithm_name: str
    analysis_type: str
    metrics: Dict[str, float]
    recommendations: List[str]
    analysis_time: float
    detailed_results: Dict[str, Any]


class QuantumAlgorithmAnalyzer:
    """Deep analysis system for quantum algorithms."""

    def __init__(self):
        self.analysis_results = []

    def analyze_algorithm_from_file(self, file_path: str) -> AnalysisResult:
        """Analyze algorithm from JSON file."""
        try:
            with open(file_path, 'r') as f:
                algorithm_data = json.load(f)

            return self.analyze_algorithm(algorithm_data)

        except Exception as e:
            logger.error(f"Failed to analyze algorithm from {file_path}: {e}")
            return None

    def analyze_algorithm(self, algorithm_data: Dict) -> AnalysisResult:
        """Comprehensive algorithm analysis."""
        start_time = datetime.now()

        algorithm_name = algorithm_data.get(
            'algorithm_info', {}).get('id', 'Unknown')
        logger.info(f"🔍 DEEP ANALYSIS: {algorithm_name}")

        # Perform comprehensive analysis
        circuit_analysis = self._analyze_circuit_structure(algorithm_data)
        performance_analysis = self._analyze_performance(algorithm_data)
        complexity_analysis = self._analyze_complexity(algorithm_data)
        optimization_analysis = self._analyze_optimization_potential(
            algorithm_data)
        quantum_advantage_analysis = self._analyze_quantum_advantage(
            algorithm_data)
        application_analysis = self._analyze_application_potential(
            algorithm_data)

        # Compile results
        metrics = {
            **circuit_analysis['metrics'],
            **performance_analysis['metrics'],
            **complexity_analysis['metrics'],
            **optimization_analysis['metrics'],
            **quantum_advantage_analysis['metrics'],
            **application_analysis['metrics']
        }

        recommendations = (
            circuit_analysis['recommendations'] +
            performance_analysis['recommendations'] +
            complexity_analysis['recommendations'] +
            optimization_analysis['recommendations'] +
            quantum_advantage_analysis['recommendations'] +
            application_analysis['recommendations']
        )

        detailed_results = {
            'circuit_analysis': circuit_analysis,
            'performance_analysis': performance_analysis,
            'complexity_analysis': complexity_analysis,
            'optimization_analysis': optimization_analysis,
            'quantum_advantage_analysis': quantum_advantage_analysis,
            'application_analysis': application_analysis
        }

        analysis_time = (datetime.now() - start_time).total_seconds()

        result = AnalysisResult(
            algorithm_name=algorithm_name,
            analysis_type="comprehensive_deep_analysis",
            metrics=metrics,
            recommendations=recommendations,
            analysis_time=analysis_time,
            detailed_results=detailed_results
        )

        self.analysis_results.append(result)
        return result

    def _analyze_circuit_structure(self, algorithm_data: Dict) -> Dict:
        """Analyze quantum circuit structure and patterns."""
        logger.info("   🔧 Analyzing circuit structure...")

        circuit = algorithm_data.get(
            'quantum_circuit', {}).get('gate_sequence', [])
        gates_used = algorithm_data.get(
            'gate_statistics', {}).get('gate_distribution', {})

        # Structure analysis
        total_gates = len(circuit)
        unique_gates = len(gates_used)

        # Gate type analysis
        single_qubit_gates = sum(
            1 for gate in circuit if len(gate.get('qubits', [])) == 1)
        two_qubit_gates = sum(1 for gate in circuit if len(
            gate.get('qubits', [])) == 2)
        multi_qubit_gates = sum(
            1 for gate in circuit if len(gate.get('qubits', [])) >= 3)

        # Parameterized gate analysis
        parameterized_gates = sum(
            1 for gate in circuit if gate.get('parameters', []))
        param_ratio = parameterized_gates / max(1, total_gates)

        # Circuit depth and width analysis
        circuit_depth = total_gates
        qubit_count = algorithm_data.get(
            'quantum_circuit', {}).get('qubit_count', 4)

        # Pattern analysis
        gate_diversity = unique_gates / max(1, total_gates)
        entangling_ratio = two_qubit_gates / max(1, total_gates)
        complexity_score = (multi_qubit_gates * 3 + two_qubit_gates *
                            2 + single_qubit_gates) / max(1, total_gates)

        # Advanced patterns
        has_toffoli = any(gate.get('gate') == 'ccx' for gate in circuit)
        has_controlled_rotations = any(
            gate.get('gate', '').startswith('cr') for gate in circuit)
        has_advanced_gates = has_toffoli or has_controlled_rotations

        metrics = {
            'total_gates': total_gates,
            'unique_gates': unique_gates,
            'gate_diversity': gate_diversity,
            'parameterized_ratio': param_ratio,
            'entangling_ratio': entangling_ratio,
            'complexity_score': complexity_score,
            'single_qubit_gates': single_qubit_gates,
            'two_qubit_gates': two_qubit_gates,
            'multi_qubit_gates': multi_qubit_gates,
            'circuit_depth': circuit_depth,
            'qubit_count': qubit_count
        }

        recommendations = []

        if param_ratio < 0.3:
            recommendations.append(
                "Consider adding more parameterized gates for variational optimization")
        if entangling_ratio < 0.2:
            recommendations.append(
                "Low entanglement - consider adding more two-qubit gates")
        if gate_diversity < 0.5:
            recommendations.append(
                "Limited gate diversity - explore broader gate sets")
        if has_advanced_gates:
            recommendations.append("Excellent use of advanced quantum gates")
        if complexity_score > 2.0:
            recommendations.append(
                "High complexity circuit - suitable for challenging problems")

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'details': {
                'has_toffoli': has_toffoli,
                'has_controlled_rotations': has_controlled_rotations,
                'gate_sequence_length': len(circuit)
            }
        }

    def _analyze_performance(self, algorithm_data: Dict) -> Dict:
        """Analyze algorithm performance metrics."""
        logger.info("   📊 Analyzing performance metrics...")

        performance = algorithm_data.get('performance_metrics', {})

        fidelity = performance.get('fidelity', 0.0)
        quantum_advantage = performance.get('quantum_advantage', 1.0)
        discovery_time = performance.get('discovery_time_seconds', 0.0)
        convergence_gen = performance.get('convergence_generation', 0)
        speedup_class = performance.get('speedup_class', 'classical')

        # Performance classification
        fidelity_class = "excellent" if fidelity >= 0.95 else "good" if fidelity >= 0.8 else "fair" if fidelity >= 0.6 else "poor"
        advantage_class = "revolutionary" if quantum_advantage >= 10 else "breakthrough" if quantum_advantage >= 5 else "significant" if quantum_advantage >= 3 else "moderate"

        # Discovery efficiency
        discovery_efficiency = 1.0 / \
            max(0.1, discovery_time)  # Inverse of time
        convergence_efficiency = 1.0 / \
            max(1, convergence_gen)  # Inverse of generations

        # Overall performance score
        performance_score = (fidelity * 0.4 +
                             min(quantum_advantage/10, 1.0) * 0.4 +
                             discovery_efficiency * 0.1 +
                             convergence_efficiency * 0.1)

        metrics = {
            'fidelity': fidelity,
            'quantum_advantage': quantum_advantage,
            'discovery_time': discovery_time,
            'convergence_generation': convergence_gen,
            'discovery_efficiency': discovery_efficiency,
            'convergence_efficiency': convergence_efficiency,
            'performance_score': performance_score
        }

        recommendations = []

        if fidelity < 0.8:
            recommendations.append(
                f"Fidelity ({fidelity:.3f}) could be improved - consider circuit optimization")
        if quantum_advantage < 3.0:
            recommendations.append(
                f"Quantum advantage ({quantum_advantage:.2f}x) is modest - explore enhancement strategies")
        if discovery_time > 5.0:
            recommendations.append(
                "Discovery took significant time - algorithm may be complex to optimize")
        if convergence_gen == 0:
            recommendations.append(
                "Immediate convergence - algorithm may be in a local optimum")
        if performance_score > 0.8:
            recommendations.append(
                "Excellent overall performance - ready for practical applications")

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'details': {
                'fidelity_class': fidelity_class,
                'advantage_class': advantage_class,
                'speedup_class': speedup_class
            }
        }

    def _analyze_complexity(self, algorithm_data: Dict) -> Dict:
        """Analyze algorithmic complexity and scalability."""
        logger.info("   🧮 Analyzing complexity and scalability...")

        circuit = algorithm_data.get(
            'quantum_circuit', {}).get('gate_sequence', [])
        performance = algorithm_data.get('performance_metrics', {})
        gates_used = algorithm_data.get(
            'gate_statistics', {}).get('gate_distribution', {})

        qubit_count = algorithm_data.get(
            'quantum_circuit', {}).get('qubit_count', 4)
        circuit_depth = len(circuit)
        entanglement_measure = performance.get('entanglement_measure', 0.0)

        # Complexity metrics
        gate_complexity = sum(count * self._gate_complexity_factor(gate)
                              for gate, count in gates_used.items())
        entanglement_complexity = entanglement_measure * qubit_count
        depth_complexity = circuit_depth / qubit_count

        # Scalability analysis
        classical_complexity = 2**qubit_count  # Exponential classical simulation cost
        quantum_gate_count = sum(gates_used.values())
        quantum_complexity_estimate = quantum_gate_count * qubit_count

        scalability_ratio = classical_complexity / \
            max(1, quantum_complexity_estimate)

        # Resource requirements
        gate_variety = len(gates_used)
        advanced_gate_count = sum(gates_used.get(gate, 0)
                                  for gate in ['ccx', 'crx', 'cry', 'crz'])

        metrics = {
            'gate_complexity': gate_complexity,
            'entanglement_complexity': entanglement_complexity,
            'depth_complexity': depth_complexity,
            'scalability_ratio': scalability_ratio,
            'classical_complexity': classical_complexity,
            'quantum_complexity': quantum_complexity_estimate,
            'gate_variety': gate_variety,
            'advanced_gate_count': advanced_gate_count
        }

        recommendations = []

        if scalability_ratio > 1000:
            recommendations.append(
                "Excellent scalability potential - significant classical vs quantum complexity gap")
        if gate_complexity > 50:
            recommendations.append(
                "High gate complexity - may require error correction for reliable execution")
        if depth_complexity > 3:
            recommendations.append(
                "Deep circuit relative to qubit count - consider parallelization opportunities")
        if advanced_gate_count == 0:
            recommendations.append(
                "No advanced gates used - potential for more sophisticated operations")
        if entanglement_complexity < 1.0:
            recommendations.append(
                "Low entanglement complexity - may limit quantum advantage")

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'details': {
                'requires_error_correction': gate_complexity > 100,
                'suitable_for_nisq': gate_complexity < 50,
                'high_entanglement': entanglement_complexity > 2.0
            }
        }

    def _analyze_optimization_potential(self, algorithm_data: Dict) -> Dict:
        """Analyze potential for algorithm optimization and improvement."""
        logger.info("   ⚙️ Analyzing optimization potential...")

        circuit = algorithm_data.get(
            'quantum_circuit', {}).get('gate_sequence', [])
        gates_used = algorithm_data.get(
            'gate_statistics', {}).get('gate_distribution', {})
        performance = algorithm_data.get('performance_metrics', {})

        # Gate optimization potential
        redundant_gates = self._find_potential_redundancies(circuit)
        optimization_opportunities = self._find_optimization_opportunities(
            gates_used)

        # Parameter optimization potential
        parameterized_gates = sum(
            1 for gate in circuit if gate.get('parameters', []))
        param_optimization_potential = parameterized_gates / \
            max(1, len(circuit))

        # Circuit simplification potential
        simplification_score = self._calculate_simplification_potential(
            circuit, gates_used)

        # Performance improvement potential
        current_fidelity = performance.get('fidelity', 0.0)
        improvement_headroom = 1.0 - current_fidelity

        metrics = {
            'redundant_gates': redundant_gates,
            'optimization_opportunities': len(optimization_opportunities),
            'param_optimization_potential': param_optimization_potential,
            'simplification_score': simplification_score,
            'improvement_headroom': improvement_headroom
        }

        recommendations = []

        if redundant_gates > 0:
            recommendations.append(
                f"Found {redundant_gates} potentially redundant gates for removal")
        if param_optimization_potential > 0.3:
            recommendations.append(
                "High parameter optimization potential - consider variational methods")
        if simplification_score > 0.5:
            recommendations.append(
                "Circuit may benefit from gate sequence optimization")
        if improvement_headroom > 0.2:
            recommendations.append(
                f"Significant improvement potential ({improvement_headroom:.2f}) available")
        if len(optimization_opportunities) > 0:
            recommendations.extend(
                [f"Optimization opportunity: {opp}" for opp in optimization_opportunities[:3]])

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'details': {
                'optimization_opportunities': optimization_opportunities,
                'current_fidelity': current_fidelity
            }
        }

    def _analyze_quantum_advantage(self, algorithm_data: Dict) -> Dict:
        """Analyze quantum advantage characteristics and validation."""
        logger.info("   ⚡ Analyzing quantum advantage...")

        performance = algorithm_data.get('performance_metrics', {})
        properties = algorithm_data.get('algorithm_properties', {})

        quantum_advantage = performance.get('quantum_advantage', 1.0)
        speedup_class = performance.get('speedup_class', 'classical')
        creates_superposition = properties.get('creates_superposition', False)
        generates_entanglement = properties.get(
            'generates_entanglement', False)

        # Quantum resource analysis
        entanglement_measure = performance.get('entanglement_measure', 0.0)

        # Advantage validation
        advantage_sources = []
        if creates_superposition:
            advantage_sources.append("superposition")
        if generates_entanglement:
            advantage_sources.append("entanglement")
        if entanglement_measure > 0.3:
            advantage_sources.append("high_entanglement")

        advantage_score = len(advantage_sources) * quantum_advantage / 4.0

        # Classical hardness analysis
        classical_hardness = self._estimate_classical_hardness(algorithm_data)
        quantum_efficiency = quantum_advantage / max(1, classical_hardness)

        metrics = {
            'quantum_advantage': quantum_advantage,
            'advantage_score': advantage_score,
            'classical_hardness': classical_hardness,
            'quantum_efficiency': quantum_efficiency,
            'entanglement_measure': entanglement_measure,
            'advantage_sources_count': len(advantage_sources)
        }

        recommendations = []

        if quantum_advantage < 2.0:
            recommendations.append(
                "Limited quantum advantage - consider algorithm modifications")
        if not creates_superposition:
            recommendations.append(
                "No superposition detected - missing key quantum resource")
        if not generates_entanglement:
            recommendations.append(
                "No entanglement detected - may limit quantum advantage")
        if advantage_score > 5.0:
            recommendations.append(
                "Strong quantum advantage detected - excellent for demonstration")
        if speedup_class == "exponential":
            recommendations.append(
                "Exponential speedup class - high potential for quantum supremacy")

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'details': {
                'speedup_class': speedup_class,
                'advantage_sources': advantage_sources,
                'creates_superposition': creates_superposition,
                'generates_entanglement': generates_entanglement
            }
        }

    def _analyze_application_potential(self, algorithm_data: Dict) -> Dict:
        """Analyze potential applications and practical utility."""
        logger.info("   🎯 Analyzing application potential...")

        domain = algorithm_data.get(
            'algorithm_info', {}).get('domain', 'unknown')
        performance = algorithm_data.get('performance_metrics', {})
        applications = algorithm_data.get('applications', [])

        fidelity = performance.get('fidelity', 0.0)
        quantum_advantage = performance.get('quantum_advantage', 1.0)

        # Application scoring
        application_count = len(applications)
        practical_readiness = min(1.0, fidelity * quantum_advantage / 3.0)

        # Domain-specific analysis
        domain_maturity = self._assess_domain_maturity(domain)
        commercial_potential = self._assess_commercial_potential(
            domain, quantum_advantage)
        research_value = self._assess_research_value(algorithm_data)

        # Implementation readiness
        implementation_complexity = self._assess_implementation_complexity(
            algorithm_data)
        hardware_requirements = self._assess_hardware_requirements(
            algorithm_data)

        metrics = {
            'application_count': application_count,
            'practical_readiness': practical_readiness,
            'domain_maturity': domain_maturity,
            'commercial_potential': commercial_potential,
            'research_value': research_value,
            'implementation_complexity': implementation_complexity,
            'hardware_requirements': hardware_requirements
        }

        recommendations = []

        if practical_readiness > 0.7:
            recommendations.append(
                "High practical readiness - suitable for near-term applications")
        if commercial_potential > 0.6:
            recommendations.append(
                "Strong commercial potential - consider industry partnerships")
        if research_value > 0.8:
            recommendations.append(
                "High research value - excellent for academic publication")
        if implementation_complexity < 0.3:
            recommendations.append(
                "Low implementation complexity - suitable for educational demonstrations")
        if hardware_requirements < 0.5:
            recommendations.append(
                "Modest hardware requirements - can run on current quantum computers")

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'details': {
                'domain': domain,
                'applications': applications,
                'target_industries': self._identify_target_industries(domain)
            }
        }

    def _gate_complexity_factor(self, gate: str) -> float:
        """Return complexity factor for different gates."""
        complexity_map = {
            'h': 1.0, 'x': 1.0, 'y': 1.0, 'z': 1.0,
            'rx': 1.2, 'ry': 1.2, 'rz': 1.2,
            'cx': 2.0, 'cz': 2.0,
            'ccx': 5.0, 'cswap': 4.0,
            'crx': 3.0, 'cry': 3.0, 'crz': 3.0,
            'swap': 3.0, 'iswap': 3.5
        }
        return complexity_map.get(gate, 2.0)

    def _find_potential_redundancies(self, circuit: List[Dict]) -> int:
        """Find potentially redundant gates in circuit."""
        # Simplified redundancy detection
        redundancies = 0
        for i in range(len(circuit) - 1):
            if circuit[i].get('gate') == circuit[i+1].get('gate'):
                if circuit[i].get('qubits') == circuit[i+1].get('qubits'):
                    redundancies += 1
        return redundancies

    def _find_optimization_opportunities(self, gates_used: Dict[str, int]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []

        if gates_used.get('h', 0) > gates_used.get('cx', 0) * 2:
            opportunities.append("Consider balancing Hadamard and CNOT gates")

        if gates_used.get('x', 0) > 3:
            opportunities.append("Multiple X gates may be optimizable")

        if not any(gate.startswith('r') for gate in gates_used.keys()):
            opportunities.append(
                "Consider adding parameterized gates for optimization")

        return opportunities

    def _calculate_simplification_potential(self, circuit: List[Dict], gates_used: Dict[str, int]) -> float:
        """Calculate potential for circuit simplification."""
        total_gates = len(circuit)
        unique_gates = len(gates_used)

        # More gates relative to unique types suggests potential simplification
        if total_gates == 0:
            return 0.0

        simplification_potential = (total_gates - unique_gates) / total_gates
        return max(0.0, min(1.0, simplification_potential))

    def _estimate_classical_hardness(self, algorithm_data: Dict) -> float:
        """Estimate classical computational hardness."""
        qubit_count = algorithm_data.get(
            'quantum_circuit', {}).get('qubit_count', 4)
        entanglement_measure = algorithm_data.get(
            'performance_metrics', {}).get('entanglement_measure', 0.0)

        # Exponential scaling with qubits and entanglement
        base_hardness = 2**min(qubit_count, 10)  # Cap for numerical stability
        entanglement_multiplier = 1 + entanglement_measure * 5

        return base_hardness * entanglement_multiplier

    def _assess_domain_maturity(self, domain: str) -> float:
        """Assess maturity of the quantum algorithm domain."""
        maturity_map = {
            'quantum_search': 0.9,
            'quantum_optimization': 0.7,
            'quantum_simulation': 0.8,
            'quantum_cryptography': 0.6,
            'quantum_ml': 0.5,
            'quantum_communication': 0.7,
            'quantum_error_correction': 0.8,
            'quantum_chemistry': 0.6
        }
        return maturity_map.get(domain, 0.5)

    def _assess_commercial_potential(self, domain: str, quantum_advantage: float) -> float:
        """Assess commercial potential of algorithm."""
        commercial_domains = {
            'quantum_optimization': 0.9,
            'quantum_simulation': 0.8,
            'quantum_ml': 0.8,
            'quantum_cryptography': 0.7,
            'quantum_chemistry': 0.7,
            'quantum_search': 0.6,
            'quantum_communication': 0.6,
            'quantum_error_correction': 0.5
        }

        base_potential = commercial_domains.get(domain, 0.4)
        advantage_multiplier = min(2.0, quantum_advantage / 3.0)

        return min(1.0, base_potential * advantage_multiplier)

    def _assess_research_value(self, algorithm_data: Dict) -> float:
        """Assess research value of algorithm."""
        quantum_advantage = algorithm_data.get(
            'performance_metrics', {}).get('quantum_advantage', 1.0)
        speedup_class = algorithm_data.get(
            'performance_metrics', {}).get('speedup_class', 'classical')

        base_value = 0.5

        if speedup_class == 'exponential':
            base_value += 0.3
        elif speedup_class == 'polynomial':
            base_value += 0.2

        if quantum_advantage > 5.0:
            base_value += 0.2

        return min(1.0, base_value)

    def _assess_implementation_complexity(self, algorithm_data: Dict) -> float:
        """Assess implementation complexity."""
        circuit_depth = algorithm_data.get(
            'performance_metrics', {}).get('circuit_depth', 0)
        gates_used = algorithm_data.get(
            'gate_statistics', {}).get('gate_distribution', {})

        advanced_gates = sum(gates_used.get(gate, 0)
                             for gate in ['ccx', 'crx', 'cry', 'crz'])
        total_gates = sum(gates_used.values())

        depth_complexity = min(1.0, circuit_depth / 20.0)
        gate_complexity = min(1.0, advanced_gates / max(1, total_gates))

        return (depth_complexity + gate_complexity) / 2

    def _assess_hardware_requirements(self, algorithm_data: Dict) -> float:
        """Assess hardware requirements."""
        qubit_count = algorithm_data.get(
            'quantum_circuit', {}).get('qubit_count', 4)
        circuit_depth = algorithm_data.get(
            'performance_metrics', {}).get('circuit_depth', 0)

        # Normalize to 20 qubits
        qubit_requirement = min(1.0, qubit_count / 20.0)
        # Normalize to 100 gates
        depth_requirement = min(1.0, circuit_depth / 100.0)

        return (qubit_requirement + depth_requirement) / 2

    def _identify_target_industries(self, domain: str) -> List[str]:
        """Identify target industries for domain."""
        industry_map = {
            'quantum_optimization': ['Finance', 'Logistics', 'Manufacturing', 'Energy'],
            'quantum_simulation': ['Pharmaceuticals', 'Materials Science', 'Chemistry'],
            'quantum_ml': ['Technology', 'Finance', 'Healthcare', 'Automotive'],
            'quantum_cryptography': ['Cybersecurity', 'Banking', 'Government', 'Telecommunications'],
            'quantum_chemistry': ['Pharmaceuticals', 'Chemical Industry', 'Materials Science'],
            'quantum_search': ['Technology', 'Data Analytics', 'Research'],
            'quantum_communication': ['Telecommunications', 'Cybersecurity', 'Government'],
            'quantum_error_correction': ['Quantum Computing', 'Research', 'Technology']
        }
        return industry_map.get(domain, ['Research', 'Technology'])

    def generate_analysis_report(self, analysis_result: AnalysisResult) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append(f"📊 COMPREHENSIVE ANALYSIS REPORT")
        report.append(f"{'='*50}")
        report.append(f"Algorithm: {analysis_result.algorithm_name}")
        report.append(f"Analysis Type: {analysis_result.analysis_type}")
        report.append(f"Analysis Time: {analysis_result.analysis_time:.2f}s")
        report.append("")

        report.append("🎯 KEY METRICS:")
        for metric, value in analysis_result.metrics.items():
            if isinstance(value, float):
                report.append(f"   {metric}: {value:.3f}")
            else:
                report.append(f"   {metric}: {value}")
        report.append("")

        report.append("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis_result.recommendations, 1):
            report.append(f"   {i}. {rec}")
        report.append("")

        report.append("🔍 DETAILED ANALYSIS:")
        for analysis_type, details in analysis_result.detailed_results.items():
            report.append(f"   {analysis_type.upper()}:")
            for key, value in details.get('metrics', {}).items():
                if isinstance(value, float):
                    report.append(f"     {key}: {value:.3f}")
                else:
                    report.append(f"     {key}: {value}")
            report.append("")

        return "\n".join(report)

    def save_analysis_results(self, filename: str = None):
        """Save analysis results to JSON file."""
        if filename is None:
            filename = f"algorithm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_analyses': len(self.analysis_results),
            'results': [
                {
                    'algorithm_name': result.algorithm_name,
                    'analysis_type': result.analysis_type,
                    'metrics': result.metrics,
                    'recommendations': result.recommendations,
                    'analysis_time': result.analysis_time
                }
                for result in self.analysis_results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analysis results saved to {filename}")


if __name__ == "__main__":
    print("📊 Quantum Algorithm Deep Analysis System")
    print("Analyzing discovered algorithms for comprehensive insights...")
    print()

    analyzer = QuantumAlgorithmAnalyzer()

    # Analyze algorithms from discovery sessions
    algorithm_files = [
        "../search/QAlgo-Search-2.json",
        "../optimization/QAlgo-Optimization-1.json"
    ]

    for file_path in algorithm_files:
        try:
            result = analyzer.analyze_algorithm_from_file(file_path)
            if result:
                print(analyzer.generate_analysis_report(result))
                print("\n" + "="*70 + "\n")
        except Exception as e:
            print(f"Failed to analyze {file_path}: {e}")

    # Save results
    analyzer.save_analysis_results()
    print("🎯 Deep analysis completed and saved!")
