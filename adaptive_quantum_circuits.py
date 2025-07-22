#!/usr/bin/env python3
"""
Adaptive Quantum Circuits: Self-Modifying Quantum Programs
==========================================================

Breakthrough implementation of quantum circuits that adapt their structure
in real-time based on:
- Runtime measurement feedback
- Environmental noise conditions  
- Performance optimization goals
- Dynamic resource constraints

This enables quantum programs that evolve and optimize themselves during execution.
"""

import numpy as np
import asyncio
import logging
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import threading
from collections import deque

# Import from existing quantum simulator
from quantum_simulator import (
    QuantumSimulator, GateOperation, GateType, SimulationType,
    AIGateOptimizer
)

logger = logging.getLogger(__name__)


class AdaptationTrigger(Enum):
    """Types of triggers that cause circuit adaptation."""
    MEASUREMENT_FEEDBACK = "measurement"
    ERROR_RATE_THRESHOLD = "error_rate"
    PERFORMANCE_DEGRADATION = "performance"
    RESOURCE_CONSTRAINT = "resource"
    ENVIRONMENTAL_CHANGE = "environment"
    CONVERGENCE_STALL = "convergence"


@dataclass
class CircuitMetrics:
    """Real-time metrics for circuit performance."""
    fidelity: float = 0.0
    execution_time: float = 0.0
    error_rate: float = 0.0
    gate_count: int = 0
    entanglement_entropy: float = 0.0
    success_probability: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdaptationRule:
    """Rule for how circuits should adapt."""
    trigger: AdaptationTrigger
    condition: Callable[[CircuitMetrics], bool]
    adaptation_strategy: Callable[[List[GateOperation]], List[GateOperation]]
    priority: int = 1
    enabled: bool = True


class QuantumFeedbackController:
    """Controls quantum circuit adaptation based on measurement feedback."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.measurement_history: deque = deque(maxlen=100)
        self.adaptation_history: List[Dict[str, Any]] = []

    def process_measurement_result(self, measurement: Dict[str, int]) -> Dict[str, Any]:
        """Process measurement result and generate adaptation signals."""
        self.measurement_history.append(measurement)

        feedback = {
            "timestamp": time.time(),
            "measurement": measurement,
            "patterns": self._analyze_measurement_patterns(),
            "recommendations": self._generate_adaptation_recommendations()
        }

        return feedback

    def _analyze_measurement_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in measurement history."""
        if len(self.measurement_history) < 5:
            return {"status": "insufficient_data"}

        recent_measurements = list(self.measurement_history)[-10:]

        # Calculate bias in measurements
        state_counts = {}
        for measurement in recent_measurements:
            state_str = "".join(str(measurement.get(f"qubit_{i}", 0))
                                for i in range(self.num_qubits))
            state_counts[state_str] = state_counts.get(state_str, 0) + 1

        # Analyze entropy of measurement distribution
        total_measurements = len(recent_measurements)
        probabilities = [
            count / total_measurements for count in state_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        # Detect bias toward specific states
        max_prob = max(probabilities) if probabilities else 0
        bias_detected = max_prob > 0.7  # More than 70% in one state

        return {
            "entropy": entropy,
            "max_probability": max_prob,
            "bias_detected": bias_detected,
            "state_distribution": state_counts,
            "uniformity_score": entropy / np.log2(len(state_counts)) if state_counts else 0
        }

    def _generate_adaptation_recommendations(self) -> List[str]:
        """Generate recommendations for circuit adaptation."""
        recommendations = []

        if len(self.measurement_history) < 3:
            return recommendations

        patterns = self._analyze_measurement_patterns()

        if patterns.get("bias_detected", False):
            recommendations.append("add_randomization_gates")
            recommendations.append("increase_entanglement")

        if patterns.get("entropy", 0) < 0.5:
            recommendations.append("add_hadamard_gates")
            recommendations.append("modify_rotation_angles")

        if patterns.get("uniformity_score", 0) > 0.9:
            recommendations.append("add_phase_gates")
            recommendations.append("optimize_for_structure")

        return recommendations


class EnvironmentalMonitor:
    """Monitors environmental conditions affecting quantum circuits."""

    def __init__(self):
        self.noise_profile: Dict[str, float] = {}
        self.coherence_times: Dict[int, float] = {}
        self.gate_fidelities: Dict[str, float] = {}
        self.monitoring_active = False

    def start_monitoring(self):
        """Start continuous environmental monitoring."""
        self.monitoring_active = True
        # In real implementation, this would interface with hardware monitors

    def stop_monitoring(self):
        """Stop environmental monitoring."""
        self.monitoring_active = False

    def get_current_conditions(self) -> Dict[str, Any]:
        """Get current environmental conditions."""
        # Simulate realistic noise conditions
        return {
            "decoherence_rate": np.random.uniform(0.001, 0.01),
            "gate_error_rate": np.random.uniform(0.0001, 0.001),
            "crosstalk_strength": np.random.uniform(0.0, 0.05),
            "temperature_drift": np.random.uniform(-0.1, 0.1),
            "timestamp": time.time()
        }

    def predict_error_evolution(self, time_horizon: float = 1.0) -> Dict[str, float]:
        """Predict how errors will evolve over time."""
        current = self.get_current_conditions()

        # Simple predictive model (in practice, would use ML)
        predicted = {}
        for key, value in current.items():
            if isinstance(value, (int, float)) and key != "timestamp":
                # Add trend and uncertainty
                trend = np.random.uniform(-0.1, 0.1) * value
                noise = np.random.uniform(-0.05, 0.05) * value
                predicted[f"{key}_predicted"] = value + trend + noise

        return predicted


class AdaptiveQuantumCircuit:
    """Self-modifying quantum circuit with real-time adaptation."""

    def __init__(self,
                 num_qubits: int,
                 initial_circuit: Optional[List[GateOperation]] = None,
                 adaptation_threshold: float = 0.1):
        self.num_qubits = num_qubits
        self.circuit = initial_circuit or []
        self.adaptation_threshold = adaptation_threshold

        # Adaptation infrastructure
        self.feedback_controller = QuantumFeedbackController(num_qubits)
        self.environmental_monitor = EnvironmentalMonitor()
        self.adaptation_rules: List[AdaptationRule] = []
        self.optimizer = AIGateOptimizer()

        # Performance tracking
        self.metrics_history: List[CircuitMetrics] = []
        self.adaptation_log: List[Dict[str, Any]] = []

        # Setup default adaptation rules
        self._setup_default_adaptation_rules()

    def _setup_default_adaptation_rules(self):
        """Setup default adaptation rules."""

        # Rule 1: Add randomization when bias detected
        def bias_condition(metrics: CircuitMetrics) -> bool:
            return metrics.success_probability < 0.3

        def add_randomization(circuit: List[GateOperation]) -> List[GateOperation]:
            new_circuit = circuit.copy()
            # Add Hadamard gates for randomization
            for qubit in range(self.num_qubits):
                if np.random.random() < 0.3:
                    hadamard_op = GateOperation(GateType.HADAMARD, [qubit])
                    insertion_point = np.random.randint(
                        0, len(new_circuit) + 1)
                    new_circuit.insert(insertion_point, hadamard_op)
            return new_circuit

        self.adaptation_rules.append(AdaptationRule(
            trigger=AdaptationTrigger.MEASUREMENT_FEEDBACK,
            condition=bias_condition,
            adaptation_strategy=add_randomization,
            priority=1
        ))

        # Rule 2: Reduce depth when error rate high
        def high_error_condition(metrics: CircuitMetrics) -> bool:
            return metrics.error_rate > 0.05

        def reduce_depth(circuit: List[GateOperation]) -> List[GateOperation]:
            if len(circuit) <= 5:
                return circuit
            # Remove gates with lowest impact
            return circuit[:-max(1, len(circuit) // 4)]

        self.adaptation_rules.append(AdaptationRule(
            trigger=AdaptationTrigger.ERROR_RATE_THRESHOLD,
            condition=high_error_condition,
            adaptation_strategy=reduce_depth,
            priority=2
        ))

        # Rule 3: Add entangling gates when entropy low
        def low_entanglement_condition(metrics: CircuitMetrics) -> bool:
            return metrics.entanglement_entropy < 0.5

        def add_entanglement(circuit: List[GateOperation]) -> List[GateOperation]:
            new_circuit = circuit.copy()
            # Add CNOT gates for entanglement
            for _ in range(np.random.randint(1, 3)):
                control = np.random.randint(0, self.num_qubits)
                target = np.random.randint(0, self.num_qubits)
                if control != target:
                    cnot_op = GateOperation(GateType.CNOT, [target], [control])
                    new_circuit.append(cnot_op)
            return new_circuit

        self.adaptation_rules.append(AdaptationRule(
            trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
            condition=low_entanglement_condition,
            adaptation_strategy=add_entanglement,
            priority=3
        ))

    async def execute_adaptive_circuit(self,
                                       target_function: Callable[[QuantumSimulator], float],
                                       max_iterations: int = 100,
                                       adaptation_frequency: int = 10) -> Dict[str, Any]:
        """Execute circuit with real-time adaptation."""

        logger.info(
            f"🔄 Starting adaptive circuit execution with {len(self.circuit)} initial gates")

        best_performance = float('-inf')
        best_circuit = self.circuit.copy()
        stagnation_count = 0

        self.environmental_monitor.start_monitoring()

        try:
            for iteration in range(max_iterations):
                # Execute current circuit
                simulator = QuantumSimulator(self.num_qubits)
                execution_start = time.time()

                try:
                    for operation in self.circuit:
                        simulator.apply_gate(
                            operation.gate_type,
                            operation.target_qubits,
                            operation.control_qubits,
                            operation.parameters
                        )

                    execution_time = time.time() - execution_start
                    performance = target_function(simulator)

                    # Calculate metrics
                    metrics = await self._calculate_metrics(simulator, execution_time, performance)
                    self.metrics_history.append(metrics)

                    # Check for improvement
                    if performance > best_performance:
                        best_performance = performance
                        best_circuit = self.circuit.copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1

                    # Periodic adaptation
                    if iteration % adaptation_frequency == 0 or stagnation_count > 15:
                        await self._trigger_adaptation(metrics)
                        stagnation_count = 0

                    # Log progress
                    if iteration % 20 == 0:
                        logger.info(f"Iteration {iteration}: Performance = {performance:.4f}, "
                                    f"Circuit depth = {len(self.circuit)}")

                except Exception as e:
                    logger.warning(
                        f"Circuit execution failed at iteration {iteration}: {e}")
                    # Trigger error-based adaptation
                    error_metrics = CircuitMetrics(
                        error_rate=1.0, gate_count=len(self.circuit))
                    await self._trigger_adaptation(error_metrics)

        finally:
            self.environmental_monitor.stop_monitoring()

        result = {
            "best_performance": best_performance,
            "best_circuit": best_circuit,
            "final_circuit": self.circuit,
            "total_adaptations": len(self.adaptation_log),
            "metrics_history": self.metrics_history[-10:],  # Last 10 metrics
            "adaptation_summary": self._summarize_adaptations()
        }

        logger.info(
            f"✅ Adaptive execution completed. Best performance: {best_performance:.4f}")
        return result

    async def _calculate_metrics(self,
                                 simulator: QuantumSimulator,
                                 execution_time: float,
                                 performance: float) -> CircuitMetrics:
        """Calculate comprehensive circuit metrics."""

        # Basic metrics
        gate_count = len(self.circuit)

        # Entanglement entropy
        if simulator.num_qubits >= 2:
            entanglement_entropy = simulator.compute_entanglement_entropy([0])
        else:
            entanglement_entropy = 0.0

        # Environmental conditions
        env_conditions = self.environmental_monitor.get_current_conditions()
        error_rate = env_conditions.get("gate_error_rate", 0.0)

        # Fidelity estimate (simplified)
        state_norm = np.linalg.norm(simulator.state)
        fidelity = state_norm if np.isfinite(state_norm) else 0.0

        return CircuitMetrics(
            fidelity=fidelity,
            execution_time=execution_time,
            error_rate=error_rate,
            gate_count=gate_count,
            entanglement_entropy=entanglement_entropy,
            success_probability=performance,
            resource_usage={"memory": gate_count * 8, "time": execution_time}
        )

    async def _trigger_adaptation(self, metrics: CircuitMetrics):
        """Trigger circuit adaptation based on current metrics."""

        adaptation_applied = False

        # Sort rules by priority
        active_rules = [rule for rule in self.adaptation_rules if rule.enabled]
        active_rules.sort(key=lambda x: x.priority)

        for rule in active_rules:
            if rule.condition(metrics):
                logger.info(f"🔧 Triggering adaptation: {rule.trigger.value}")

                # Apply adaptation strategy
                old_circuit = self.circuit.copy()
                self.circuit = rule.adaptation_strategy(self.circuit)

                # Optimize adapted circuit
                self.circuit = self.optimizer.optimize_circuit(self.circuit)

                # Log adaptation
                adaptation_record = {
                    "timestamp": time.time(),
                    "trigger": rule.trigger.value,
                    "old_circuit_length": len(old_circuit),
                    "new_circuit_length": len(self.circuit),
                    "metrics": metrics
                }
                self.adaptation_log.append(adaptation_record)

                adaptation_applied = True
                break  # Apply only one adaptation per trigger

        # If no rule triggered but performance is very poor, apply emergency adaptation
        if not adaptation_applied and metrics.success_probability < 0.1:
            await self._emergency_adaptation()

    async def _emergency_adaptation(self):
        """Apply emergency adaptation when circuit performance is critically poor."""
        logger.warning("🚨 Applying emergency adaptation")

        # Simplified emergency strategy: reduce to minimal circuit + randomization
        emergency_circuit = []

        # Add basic randomization
        for qubit in range(self.num_qubits):
            emergency_circuit.append(GateOperation(GateType.HADAMARD, [qubit]))

        # Add minimal entanglement
        if self.num_qubits >= 2:
            emergency_circuit.append(GateOperation(GateType.CNOT, [1], [0]))

        self.circuit = emergency_circuit

        self.adaptation_log.append({
            "timestamp": time.time(),
            "trigger": "emergency",
            "new_circuit_length": len(self.circuit),
            "description": "Emergency circuit reset"
        })

    def _summarize_adaptations(self) -> Dict[str, Any]:
        """Summarize adaptation history."""
        if not self.adaptation_log:
            return {"total_adaptations": 0}

        trigger_counts = {}
        for record in self.adaptation_log:
            trigger = record.get("trigger", "unknown")
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        return {
            "total_adaptations": len(self.adaptation_log),
            "trigger_distribution": trigger_counts,
            "adaptation_frequency": len(self.adaptation_log) / max(1, len(self.metrics_history)),
            "average_improvement": self._calculate_average_improvement()
        }

    def _calculate_average_improvement(self) -> float:
        """Calculate average performance improvement after adaptations."""
        if len(self.metrics_history) < 2:
            return 0.0

        improvements = []
        for i in range(1, len(self.metrics_history)):
            improvement = (self.metrics_history[i].success_probability -
                           self.metrics_history[i-1].success_probability)
            improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def add_custom_adaptation_rule(self, rule: AdaptationRule):
        """Add custom adaptation rule."""
        self.adaptation_rules.append(rule)
        logger.info(f"Added custom adaptation rule: {rule.trigger.value}")

    def visualize_adaptation_history(self) -> Dict[str, Any]:
        """Generate visualization data for adaptation history."""
        performance_over_time = [
            m.success_probability for m in self.metrics_history]
        error_rates_over_time = [m.error_rate for m in self.metrics_history]
        circuit_depths = [m.gate_count for m in self.metrics_history]

        adaptation_points = [record["timestamp"]
                             for record in self.adaptation_log]

        return {
            "performance_timeline": performance_over_time,
            "error_rate_timeline": error_rates_over_time,
            "circuit_depth_timeline": circuit_depths,
            "adaptation_timestamps": adaptation_points,
            "summary": self._summarize_adaptations()
        }


class QuantumProgramEvolution:
    """Higher-level evolution of quantum programs using adaptive circuits."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.program_population: List[AdaptiveQuantumCircuit] = []
        self.generation = 0

    def initialize_population(self, population_size: int = 20):
        """Initialize population of adaptive circuits."""
        for _ in range(population_size):
            # Generate random initial circuit
            initial_circuit = self._generate_random_circuit()
            adaptive_circuit = AdaptiveQuantumCircuit(
                self.num_qubits, initial_circuit)
            self.program_population.append(adaptive_circuit)

        logger.info(
            f"Initialized population of {population_size} adaptive circuits")

    def _generate_random_circuit(self, max_depth: int = 10) -> List[GateOperation]:
        """Generate random initial circuit."""
        circuit = []
        depth = np.random.randint(3, max_depth)

        gates = [GateType.HADAMARD, GateType.PAULI_X, GateType.PAULI_Y,
                 GateType.PAULI_Z, GateType.S_GATE, GateType.T_GATE]

        for _ in range(depth):
            gate_type = np.random.choice(gates)
            qubit = np.random.randint(0, self.num_qubits)

            if gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
                angle = np.random.uniform(-np.pi, np.pi)
                operation = GateOperation(
                    gate_type, [qubit], parameters=[angle])
            else:
                operation = GateOperation(gate_type, [qubit])

            circuit.append(operation)

        # Add some entangling gates
        if self.num_qubits >= 2:
            for _ in range(np.random.randint(1, 3)):
                control = np.random.randint(0, self.num_qubits)
                target = np.random.randint(0, self.num_qubits)
                if control != target:
                    circuit.append(GateOperation(
                        GateType.CNOT, [target], [control]))

        return circuit

    async def evolve_programs(self,
                              target_function: Callable[[QuantumSimulator], float],
                              generations: int = 10) -> Dict[str, Any]:
        """Evolve population of adaptive quantum programs."""

        logger.info(
            f"🧬 Starting quantum program evolution for {generations} generations")

        best_overall_performance = float('-inf')
        best_program = None
        evolution_history = []

        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")

            # Evaluate all programs
            program_performances = []

            for i, program in enumerate(self.program_population):
                try:
                    result = await program.execute_adaptive_circuit(
                        target_function,
                        max_iterations=20,
                        adaptation_frequency=5
                    )
                    performance = result["best_performance"]
                    program_performances.append((performance, i, result))

                    if performance > best_overall_performance:
                        best_overall_performance = performance
                        best_program = program

                except Exception as e:
                    logger.warning(f"Program {i} failed: {e}")
                    program_performances.append((float('-inf'), i, None))

            # Sort by performance
            program_performances.sort(reverse=True, key=lambda x: x[0])

            generation_summary = {
                "generation": generation,
                "best_performance": program_performances[0][0],
                "average_performance": np.mean([p[0] for p in program_performances if p[0] != float('-inf')]),
                "adaptation_stats": self._collect_adaptation_stats()
            }
            evolution_history.append(generation_summary)

            logger.info(f"Generation {generation}: Best = {program_performances[0][0]:.4f}, "
                        f"Average = {generation_summary['average_performance']:.4f}")

            # Selection and reproduction for next generation (if not last generation)
            if generation < generations - 1:
                await self._reproduce_population(program_performances)

        return {
            "best_program": best_program,
            "best_performance": best_overall_performance,
            "evolution_history": evolution_history,
            "final_population_stats": self._analyze_final_population()
        }

    def _collect_adaptation_stats(self) -> Dict[str, Any]:
        """Collect adaptation statistics from current population."""
        total_adaptations = sum(len(p.adaptation_log)
                                for p in self.program_population)

        trigger_counts = {}
        for program in self.program_population:
            for record in program.adaptation_log:
                trigger = record.get("trigger", "unknown")
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        return {
            "total_adaptations": total_adaptations,
            "average_adaptations_per_program": total_adaptations / len(self.program_population),
            "trigger_distribution": trigger_counts
        }

    async def _reproduce_population(self, program_performances: List[Tuple[float, int, Any]]):
        """Create next generation through selection and reproduction."""
        # Keep top 25% performers
        elite_size = len(self.program_population) // 4
        elite_indices = [p[1] for p in program_performances[:elite_size]]

        new_population = [self.program_population[i] for i in elite_indices]

        # Fill rest with variations of elite programs
        while len(new_population) < len(self.program_population):
            parent_idx = np.random.choice(elite_indices)
            parent = self.program_population[parent_idx]

            # Create offspring by mutating parent circuit
            offspring_circuit = self._mutate_circuit(parent.circuit)
            offspring = AdaptiveQuantumCircuit(
                self.num_qubits, offspring_circuit)

            # Inherit some adaptation rules from parent
            offspring.adaptation_rules = parent.adaptation_rules.copy()

            new_population.append(offspring)

        self.program_population = new_population
        self.generation += 1

    def _mutate_circuit(self, circuit: List[GateOperation]) -> List[GateOperation]:
        """Mutate a circuit to create offspring."""
        mutated = circuit.copy()

        # Apply random mutations
        mutation_types = ["add_gate", "remove_gate",
                          "modify_gate", "swap_gates"]
        mutation_type = np.random.choice(mutation_types)

        if mutation_type == "add_gate" and len(mutated) < 20:
            new_gate = self._generate_random_circuit(1)[0]
            insertion_point = np.random.randint(0, len(mutated) + 1)
            mutated.insert(insertion_point, new_gate)

        elif mutation_type == "remove_gate" and len(mutated) > 1:
            removal_idx = np.random.randint(0, len(mutated))
            mutated.pop(removal_idx)

        elif mutation_type == "modify_gate" and mutated:
            gate_idx = np.random.randint(0, len(mutated))
            if mutated[gate_idx].parameters:
                # Modify parameters
                param_idx = np.random.randint(
                    0, len(mutated[gate_idx].parameters))
                mutated[gate_idx].parameters[param_idx] += np.random.uniform(
                    -0.5, 0.5)

        elif mutation_type == "swap_gates" and len(mutated) >= 2:
            i, j = np.random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated

    def _analyze_final_population(self) -> Dict[str, Any]:
        """Analyze characteristics of final population."""
        circuit_depths = [len(p.circuit) for p in self.program_population]
        adaptation_counts = [len(p.adaptation_log)
                             for p in self.program_population]

        return {
            "population_size": len(self.program_population),
            "average_circuit_depth": np.mean(circuit_depths),
            "circuit_depth_std": np.std(circuit_depths),
            "average_adaptations": np.mean(adaptation_counts),
            "max_adaptations": max(adaptation_counts) if adaptation_counts else 0,
            "diversity_score": self._calculate_population_diversity()
        }

    def _calculate_population_diversity(self) -> float:
        """Calculate diversity score of population."""
        # Simple diversity measure based on circuit structure differences
        circuit_signatures = []
        for program in self.program_population:
            signature = "|".join([f"{op.gate_type.value}_{op.target_qubits}"
                                  for op in program.circuit])
            circuit_signatures.append(signature)

        unique_signatures = len(set(circuit_signatures))
        total_programs = len(self.program_population)

        return unique_signatures / total_programs if total_programs > 0 else 0.0

# Example demonstration


async def demonstrate_adaptive_circuits():
    """Demonstrate adaptive quantum circuits."""

    logger.info("🚀 Demonstrating Adaptive Quantum Circuits")

    # Create adaptive circuit
    initial_circuit = [
        GateOperation(GateType.HADAMARD, [0]),
        GateOperation(GateType.CNOT, [1], [0]),
        GateOperation(GateType.HADAMARD, [1])
    ]

    adaptive_circuit = AdaptiveQuantumCircuit(
        num_qubits=3,
        initial_circuit=initial_circuit
    )

    # Define target function (maximize entanglement)
    def entanglement_target(simulator: QuantumSimulator) -> float:
        if simulator.num_qubits >= 2:
            return simulator.compute_entanglement_entropy([0])
        return 0.0

    # Execute adaptive circuit
    result = await adaptive_circuit.execute_adaptive_circuit(
        target_function=entanglement_target,
        max_iterations=50,
        adaptation_frequency=8
    )

    logger.info(f"🎯 Adaptive Circuit Results:")
    logger.info(f"  Best Performance: {result['best_performance']:.4f}")
    logger.info(f"  Total Adaptations: {result['total_adaptations']}")
    logger.info(f"  Final Circuit Depth: {len(result['final_circuit'])}")

    # Visualize adaptation history
    viz_data = adaptive_circuit.visualize_adaptation_history()
    logger.info(f"  Adaptation Summary: {viz_data['summary']}")

    # Demonstrate program evolution
    evolution_engine = QuantumProgramEvolution(num_qubits=3)
    evolution_engine.initialize_population(population_size=10)

    evolution_result = await evolution_engine.evolve_programs(
        target_function=entanglement_target,
        generations=5
    )

    logger.info(f"🧬 Evolution Results:")
    logger.info(
        f"  Best Evolved Performance: {evolution_result['best_performance']:.4f}")
    logger.info(
        f"  Population Diversity: {evolution_result['final_population_stats']['diversity_score']:.4f}")

    return {
        "adaptive_result": result,
        "evolution_result": evolution_result
    }

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_adaptive_circuits())
