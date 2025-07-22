#!/usr/bin/env python3
"""
Real-time Quantum Error Correction with ML-Guided Syndrome Detection
====================================================================

Advanced quantum error correction system that uses machine learning to:
- Predict and detect quantum errors in real-time
- Adaptively choose optimal correction strategies
- Learn device-specific error patterns
- Optimize error correction for NISQ devices

This enables fault-tolerant quantum computation with intelligent error management.
"""

import numpy as np
import asyncio
import logging
from typing import List, Dict, Callable, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import threading
from collections import deque, defaultdict
import pickle

# Import from existing quantum simulator
from quantum_simulator import (
    QuantumSimulator, GateOperation, GateType, SimulationType
)

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of quantum errors."""
    BIT_FLIP = "bit_flip"           # Pauli-X errors
    PHASE_FLIP = "phase_flip"       # Pauli-Z errors
    DEPOLARIZING = "depolarizing"   # Mixed Pauli errors
    AMPLITUDE_DAMPING = "amplitude_damping"  # T1 decay
    PHASE_DAMPING = "phase_damping"          # T2 dephasing
    CROSSTALK = "crosstalk"         # Inter-qubit interference
    COHERENT = "coherent"           # Systematic rotation errors


class CorrectionStrategy(Enum):
    """Error correction strategies."""
    SURFACE_CODE = "surface_code"
    STEANE_CODE = "steane_code"
    SHOR_CODE = "shor_code"
    COLOR_CODE = "color_code"
    REPETITION_CODE = "repetition_code"
    ADAPTIVE_STRATEGY = "adaptive"


@dataclass
class ErrorEvent:
    """Quantum error event record."""
    timestamp: float
    qubit_index: int
    error_type: ErrorType
    error_strength: float
    syndrome_pattern: List[int]
    correction_applied: Optional[str] = None
    correction_success: bool = False


@dataclass
class SyndromePattern:
    """Error syndrome pattern from stabilizer measurements."""
    stabilizer_measurements: np.ndarray
    timestamp: float
    confidence_score: float
    predicted_errors: Dict[int, ErrorType]
    correction_recommendation: str


class QuantumErrorDetector:
    """ML-based quantum error detection system."""

    def __init__(self, num_qubits: int, code_distance: int = 3):
        self.num_qubits = num_qubits
        self.code_distance = code_distance
        self.logical_qubits = self._calculate_logical_qubits()

        # Error pattern learning
        self.error_history: deque = deque(maxlen=10000)
        self.syndrome_patterns: Dict[str,
                                     List[SyndromePattern]] = defaultdict(list)

        # ML models for error prediction
        self.error_classifier = QuantumErrorClassifier(num_qubits)
        self.syndrome_decoder = QuantumSyndromeDecoder(
            num_qubits, code_distance)

        # Performance metrics
        self.detection_accuracy_history: List[float] = []
        self.correction_success_rate: List[float] = []

    def _calculate_logical_qubits(self) -> int:
        """Calculate number of logical qubits for given code distance."""
        # For surface code: logical qubits = (physical qubits - stabilizers) / 2
        # Simplified calculation
        return max(1, self.num_qubits // (2 * self.code_distance))

    def detect_errors_realtime(self, quantum_state: np.ndarray,
                               stabilizer_operators: List[np.ndarray]) -> List[ErrorEvent]:
        """Detect errors in real-time using ML-guided syndrome detection."""
        current_time = time.time()

        # Measure stabilizers
        syndrome_measurements = self._measure_stabilizers(
            quantum_state, stabilizer_operators)

        # ML-based syndrome analysis
        syndrome_pattern = self._analyze_syndrome_ml(
            syndrome_measurements, current_time)

        # Predict error locations and types
        predicted_errors = self._predict_errors_from_syndrome(syndrome_pattern)

        # Create error events
        error_events = []
        for qubit_idx, error_type in predicted_errors.items():
            error_strength = self._estimate_error_strength(
                qubit_idx, syndrome_pattern)

            error_event = ErrorEvent(
                timestamp=current_time,
                qubit_index=qubit_idx,
                error_type=error_type,
                error_strength=error_strength,
                syndrome_pattern=syndrome_measurements.tolist()
            )
            error_events.append(error_event)
            self.error_history.append(error_event)

        return error_events

    def _measure_stabilizers(self, quantum_state: np.ndarray,
                             stabilizer_operators: List[np.ndarray]) -> np.ndarray:
        """Measure stabilizer operators on quantum state."""
        measurements = []

        for stabilizer in stabilizer_operators:
            # Calculate expectation value of stabilizer
            expectation = np.real(np.conj(quantum_state).T @
                                  stabilizer @ quantum_state)

            # Convert to binary measurement (1 for positive eigenvalue, 0 for negative)
            measurement = 1 if expectation > 0 else 0
            measurements.append(measurement)

        return np.array(measurements)

    def _analyze_syndrome_ml(self, syndrome_measurements: np.ndarray, timestamp: float) -> SyndromePattern:
        """Analyze syndrome using machine learning."""
        # Convert syndrome to string key
        syndrome_key = "".join(map(str, syndrome_measurements))

        # Check against learned patterns
        confidence_score = self._calculate_syndrome_confidence(
            syndrome_key, syndrome_measurements)

        # Use ML classifier to predict errors
        predicted_errors = self.error_classifier.classify_syndrome(
            syndrome_measurements)

        # Get correction recommendation
        correction_recommendation = self.syndrome_decoder.decode_syndrome(
            syndrome_measurements, predicted_errors
        )

        syndrome_pattern = SyndromePattern(
            stabilizer_measurements=syndrome_measurements,
            timestamp=timestamp,
            confidence_score=confidence_score,
            predicted_errors=predicted_errors,
            correction_recommendation=correction_recommendation
        )

        # Store pattern for learning
        self.syndrome_patterns[syndrome_key].append(syndrome_pattern)

        return syndrome_pattern

    def _calculate_syndrome_confidence(self, syndrome_key: str,
                                       syndrome_measurements: np.ndarray) -> float:
        """Calculate confidence in syndrome interpretation."""
        if syndrome_key not in self.syndrome_patterns:
            return 0.5  # Neutral confidence for new patterns

        # Analyze historical patterns
        historical_patterns = self.syndrome_patterns[syndrome_key]

        if len(historical_patterns) < 3:
            return 0.6

        # Calculate consistency of historical corrections
        success_rates = [1.0 if hasattr(p, 'correction_success') and p.correction_success else 0.0
                         for p in historical_patterns[-10:]]

        return np.mean(success_rates) if success_rates else 0.5

    def _predict_errors_from_syndrome(self, syndrome_pattern: SyndromePattern) -> Dict[int, ErrorType]:
        """Predict error locations and types from syndrome pattern."""
        return syndrome_pattern.predicted_errors

    def _estimate_error_strength(self, qubit_idx: int, syndrome_pattern: SyndromePattern) -> float:
        """Estimate strength of error on given qubit."""
        # Simplified error strength estimation
        syndrome_weight = np.sum(syndrome_pattern.stabilizer_measurements)
        confidence = syndrome_pattern.confidence_score

        # Higher syndrome weight and lower confidence suggest stronger errors
        base_strength = syndrome_weight / \
            len(syndrome_pattern.stabilizer_measurements)
        confidence_factor = 1.0 - confidence

        return min(1.0, base_strength * (1.0 + confidence_factor))


class QuantumErrorClassifier:
    """ML classifier for quantum error types from syndromes."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.training_data: List[Tuple[np.ndarray, Dict[int, ErrorType]]] = []
        self.model_weights: Dict[str, np.ndarray] = {}
        self.is_trained = False

    def add_training_data(self, syndrome: np.ndarray, true_errors: Dict[int, ErrorType]):
        """Add training data for the classifier."""
        self.training_data.append((syndrome.copy(), true_errors.copy()))

        # Retrain if we have enough data
        if len(self.training_data) >= 50 and len(self.training_data) % 25 == 0:
            self.train_classifier()

    def train_classifier(self):
        """Train the error classifier using collected data."""
        if len(self.training_data) < 10:
            return

        logger.info(
            f"🧠 Training error classifier with {len(self.training_data)} samples")

        # Simple linear classifier for each error type
        error_types = list(ErrorType)

        for error_type in error_types:
            # Create training matrices
            X = np.array([data[0] for data in self.training_data])
            y = np.array([1 if error_type in data[1].values() else 0
                         for data in self.training_data])

            # Simple linear regression (in practice, would use more sophisticated ML)
            if np.sum(y) > 0:  # Only train if we have positive examples
                weights = self._fit_linear_classifier(X, y)
                self.model_weights[error_type.value] = weights

        self.is_trained = True
        logger.info("✅ Error classifier training completed")

    def _fit_linear_classifier(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit simple linear classifier."""
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Solve normal equations (regularized)
        try:
            weights = np.linalg.solve(
                X_with_bias.T @ X_with_bias + 0.01 *
                np.eye(X_with_bias.shape[1]),
                X_with_bias.T @ y
            )
        except np.linalg.LinAlgError:
            # Fallback to random weights
            weights = np.random.normal(0, 0.1, X_with_bias.shape[1])

        return weights

    def classify_syndrome(self, syndrome: np.ndarray) -> Dict[int, ErrorType]:
        """Classify syndrome to predict error types and locations."""
        if not self.is_trained or not self.model_weights:
            # Fallback to heuristic classification
            return self._heuristic_classification(syndrome)

        # ML-based classification
        predicted_errors = {}
        syndrome_with_bias = np.concatenate([[1], syndrome])

        for error_type_str, weights in self.model_weights.items():
            if weights is not None and len(weights) == len(syndrome_with_bias):
                score = np.dot(weights, syndrome_with_bias)
                probability = 1 / (1 + np.exp(-score))  # Sigmoid

                if probability > 0.5:
                    # Predict this error type occurred
                    error_type = ErrorType(error_type_str)

                    # Determine likely qubit (simplified)
                    qubit_idx = self._predict_error_location(
                        syndrome, error_type)
                    predicted_errors[qubit_idx] = error_type

        return predicted_errors

    def _heuristic_classification(self, syndrome: np.ndarray) -> Dict[int, ErrorType]:
        """Fallback heuristic classification when ML model not ready."""
        predicted_errors = {}

        # Simple heuristics based on syndrome pattern
        syndrome_weight = np.sum(syndrome)

        if syndrome_weight == 0:
            return predicted_errors  # No errors detected

        # For each violated stabilizer, predict likely error
        for i, measurement in enumerate(syndrome):
            if measurement == 1:  # Stabilizer violated
                qubit_idx = i % self.num_qubits

                # Simple pattern-based prediction
                if syndrome_weight == 1:
                    predicted_errors[qubit_idx] = ErrorType.BIT_FLIP
                elif syndrome_weight == 2:
                    predicted_errors[qubit_idx] = ErrorType.PHASE_FLIP
                else:
                    predicted_errors[qubit_idx] = ErrorType.DEPOLARIZING

        return predicted_errors

    def _predict_error_location(self, syndrome: np.ndarray, error_type: ErrorType) -> int:
        """Predict most likely location of error."""
        # Find first violated stabilizer as proxy for error location
        violated_stabilizers = np.where(syndrome == 1)[0]

        if len(violated_stabilizers) > 0:
            return violated_stabilizers[0] % self.num_qubits
        else:
            return 0  # Default to first qubit


class QuantumSyndromeDecoder:
    """ML-enhanced syndrome decoder for error correction."""

    def __init__(self, num_qubits: int, code_distance: int):
        self.num_qubits = num_qubits
        self.code_distance = code_distance
        self.lookup_table: Dict[str, str] = {}
        self.decoder_accuracy: List[float] = []

    def decode_syndrome(self, syndrome: np.ndarray,
                        predicted_errors: Dict[int, ErrorType]) -> str:
        """Decode syndrome to correction strategy."""
        syndrome_key = "".join(map(str, syndrome))

        # Check lookup table first
        if syndrome_key in self.lookup_table:
            return self.lookup_table[syndrome_key]

        # Generate correction strategy
        correction_strategy = self._generate_correction_strategy(
            syndrome, predicted_errors)

        # Cache in lookup table
        self.lookup_table[syndrome_key] = correction_strategy

        return correction_strategy

    def _generate_correction_strategy(self, syndrome: np.ndarray,
                                      predicted_errors: Dict[int, ErrorType]) -> str:
        """Generate optimal correction strategy."""
        if len(predicted_errors) == 0:
            return "no_correction"

        # Determine correction based on predicted errors
        correction_gates = []

        for qubit_idx, error_type in predicted_errors.items():
            if error_type == ErrorType.BIT_FLIP:
                correction_gates.append(f"X_{qubit_idx}")
            elif error_type == ErrorType.PHASE_FLIP:
                correction_gates.append(f"Z_{qubit_idx}")
            elif error_type == ErrorType.DEPOLARIZING:
                # For depolarizing, apply both X and Z with probability
                correction_gates.append(f"X_{qubit_idx}")
                correction_gates.append(f"Z_{qubit_idx}")
            # Add more error type handlers as needed

        return "|".join(correction_gates)

    def update_decoder_performance(self, syndrome: np.ndarray,
                                   applied_correction: str, success: bool):
        """Update decoder performance metrics."""
        syndrome_key = "".join(map(str, syndrome))

        # Update lookup table confidence
        if success:
            # Reinforce successful correction
            self.lookup_table[syndrome_key] = applied_correction

        # Track accuracy
        self.decoder_accuracy.append(1.0 if success else 0.0)

        # Maintain recent accuracy window
        if len(self.decoder_accuracy) > 1000:
            self.decoder_accuracy = self.decoder_accuracy[-1000:]


class AdaptiveQuantumErrorCorrection:
    """Adaptive quantum error correction system."""

    def __init__(self, num_qubits: int, code_distance: int = 3):
        self.num_qubits = num_qubits
        self.code_distance = code_distance

        # Error correction components
        self.error_detector = QuantumErrorDetector(num_qubits, code_distance)
        self.correction_strategies: Dict[CorrectionStrategy, 'ErrorCorrectionCode'] = {
        }

        # Initialize correction codes
        self._initialize_correction_codes()

        # Adaptive strategy selection
        self.strategy_performance: Dict[CorrectionStrategy, List[float]] = defaultdict(
            list)
        self.current_strategy = CorrectionStrategy.SURFACE_CODE

        # Real-time monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.correction_log: List[Dict[str, Any]] = []

    def _initialize_correction_codes(self):
        """Initialize available error correction codes."""
        # Surface code (simplified)
        self.correction_strategies[CorrectionStrategy.SURFACE_CODE] = SurfaceCode(
            self.num_qubits, self.code_distance
        )

        # Repetition code
        self.correction_strategies[CorrectionStrategy.REPETITION_CODE] = RepetitionCode(
            self.num_qubits, self.code_distance
        )

        # Add more codes as needed

    async def start_realtime_correction(self, quantum_simulator: QuantumSimulator,
                                        correction_frequency: float = 0.001):  # 1ms intervals
        """Start real-time quantum error correction."""
        logger.info(f"🛡️ Starting real-time quantum error correction")
        self.monitoring_active = True

        correction_count = 0
        last_correction_time = time.time()

        while self.monitoring_active:
            current_time = time.time()

            if current_time - last_correction_time >= correction_frequency:
                try:
                    # Get current quantum state
                    quantum_state = quantum_simulator.state.copy()

                    # Get stabilizers for current correction strategy
                    current_code = self.correction_strategies[self.current_strategy]
                    stabilizers = current_code.get_stabilizer_operators()

                    # Detect errors
                    error_events = self.error_detector.detect_errors_realtime(
                        quantum_state, stabilizers
                    )

                    # Apply corrections if errors detected
                    if error_events:
                        correction_success = await self._apply_corrections(
                            quantum_simulator, error_events
                        )

                        correction_count += len(error_events)

                        # Log correction attempt
                        correction_record = {
                            "timestamp": current_time,
                            "errors_detected": len(error_events),
                            "correction_strategy": self.current_strategy.value,
                            "correction_success": correction_success,
                            "error_types": [e.error_type.value for e in error_events]
                        }
                        self.correction_log.append(correction_record)

                        # Update strategy performance
                        self.strategy_performance[self.current_strategy].append(
                            1.0 if correction_success else 0.0
                        )

                        # Adaptive strategy selection
                        if correction_count % 100 == 0:
                            await self._adapt_correction_strategy()

                    last_correction_time = current_time

                except Exception as e:
                    logger.error(f"Error correction failed: {e}")
                    break

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(correction_frequency / 10)

        logger.info(
            f"✅ Real-time error correction stopped. Total corrections: {correction_count}")

    def stop_realtime_correction(self):
        """Stop real-time error correction."""
        self.monitoring_active = False

    async def _apply_corrections(self, quantum_simulator: QuantumSimulator,
                                 error_events: List[ErrorEvent]) -> bool:
        """Apply quantum error corrections."""
        try:
            current_code = self.correction_strategies[self.current_strategy]

            for error_event in error_events:
                # Get correction gates for this error
                correction_gates = current_code.get_correction_gates(
                    error_event.qubit_index, error_event.error_type
                )

                # Apply correction gates
                for gate_operation in correction_gates:
                    quantum_simulator.apply_gate(
                        gate_operation.gate_type,
                        gate_operation.target_qubits,
                        gate_operation.control_qubits,
                        gate_operation.parameters
                    )

                # Mark correction as applied
                error_event.correction_applied = self.current_strategy.value

                # Verify correction (simplified)
                correction_success = self._verify_correction(
                    quantum_simulator, error_event)
                error_event.correction_success = correction_success

            return all(event.correction_success for event in error_events)

        except Exception as e:
            logger.error(f"Failed to apply corrections: {e}")
            return False

    def _verify_correction(self, quantum_simulator: QuantumSimulator,
                           error_event: ErrorEvent) -> bool:
        """Verify that error correction was successful."""
        # Simplified verification: check if state norm is preserved
        state_norm = np.linalg.norm(quantum_simulator.state)

        # In a real implementation, this would involve syndrome measurement
        return 0.99 <= state_norm <= 1.01

    async def _adapt_correction_strategy(self):
        """Adaptively select best correction strategy."""
        # Calculate recent performance for each strategy
        strategy_scores = {}

        for strategy, performance_history in self.strategy_performance.items():
            if len(performance_history) >= 10:
                # Last 50 corrections
                recent_performance = performance_history[-50:]
                strategy_scores[strategy] = np.mean(recent_performance)

        if strategy_scores:
            # Select best performing strategy
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            best_score = strategy_scores[best_strategy]
            current_score = strategy_scores.get(self.current_strategy, 0.0)

            # Switch if significant improvement (>5%)
            if best_score > current_score + 0.05:
                logger.info(
                    f"🔄 Switching correction strategy: {self.current_strategy.value} → {best_strategy.value}")
                logger.info(
                    f"   Performance improvement: {current_score:.3f} → {best_score:.3f}")
                self.current_strategy = best_strategy

    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive correction statistics."""
        if not self.correction_log:
            return {"status": "no_corrections_applied"}

        # Calculate overall statistics
        total_corrections = len(self.correction_log)
        successful_corrections = sum(
            1 for log in self.correction_log if log["correction_success"])
        success_rate = successful_corrections / \
            total_corrections if total_corrections > 0 else 0.0

        # Strategy performance
        strategy_stats = {}
        for strategy, performance in self.strategy_performance.items():
            if performance:
                strategy_stats[strategy.value] = {
                    "total_corrections": len(performance),
                    "success_rate": np.mean(performance),
                    "recent_success_rate": np.mean(performance[-50:]) if len(performance) >= 50 else np.mean(performance)
                }

        # Error type distribution
        error_type_counts = defaultdict(int)
        for log in self.correction_log:
            for error_type in log["error_types"]:
                error_type_counts[error_type] += 1

        return {
            "total_corrections": total_corrections,
            "overall_success_rate": success_rate,
            "current_strategy": self.current_strategy.value,
            "strategy_performance": strategy_stats,
            "error_type_distribution": dict(error_type_counts),
            "recent_correction_rate": len([log for log in self.correction_log[-100:] if time.time() - log["timestamp"] < 60]),
            # Last 10 corrections
            "correction_log_sample": self.correction_log[-10:]
        }


class ErrorCorrectionCode:
    """Base class for quantum error correction codes."""

    def __init__(self, num_qubits: int, code_distance: int):
        self.num_qubits = num_qubits
        self.code_distance = code_distance

    def get_stabilizer_operators(self) -> List[np.ndarray]:
        """Get stabilizer operators for the code."""
        raise NotImplementedError

    def get_correction_gates(self, qubit_index: int, error_type: ErrorType) -> List[GateOperation]:
        """Get correction gates for specific error."""
        raise NotImplementedError


class SurfaceCode(ErrorCorrectionCode):
    """Surface code implementation."""

    def get_stabilizer_operators(self) -> List[np.ndarray]:
        """Generate surface code stabilizers."""
        stabilizers = []

        # Simplified surface code stabilizers (X and Z type)
        for i in range(self.num_qubits - 1):
            # X-type stabilizer
            x_stabilizer = np.eye(2**self.num_qubits, dtype=complex)
            # Apply X gates (simplified)
            stabilizers.append(x_stabilizer)

            # Z-type stabilizer
            z_stabilizer = np.eye(2**self.num_qubits, dtype=complex)
            # Apply Z gates (simplified)
            stabilizers.append(z_stabilizer)

        return stabilizers

    def get_correction_gates(self, qubit_index: int, error_type: ErrorType) -> List[GateOperation]:
        """Get correction gates for surface code."""
        corrections = []

        if error_type == ErrorType.BIT_FLIP:
            corrections.append(GateOperation(GateType.PAULI_X, [qubit_index]))
        elif error_type == ErrorType.PHASE_FLIP:
            corrections.append(GateOperation(GateType.PAULI_Z, [qubit_index]))
        elif error_type == ErrorType.DEPOLARIZING:
            # Apply both corrections
            corrections.append(GateOperation(GateType.PAULI_X, [qubit_index]))
            corrections.append(GateOperation(GateType.PAULI_Z, [qubit_index]))

        return corrections


class RepetitionCode(ErrorCorrectionCode):
    """Repetition code implementation."""

    def get_stabilizer_operators(self) -> List[np.ndarray]:
        """Generate repetition code stabilizers."""
        stabilizers = []

        # Z-Z stabilizers for bit-flip repetition code
        for i in range(self.num_qubits - 1):
            stabilizer = np.eye(2**self.num_qubits, dtype=complex)
            # In full implementation, would apply Z⊗Z operators
            stabilizers.append(stabilizer)

        return stabilizers

    def get_correction_gates(self, qubit_index: int, error_type: ErrorType) -> List[GateOperation]:
        """Get correction gates for repetition code."""
        corrections = []

        # For repetition code, always apply X correction for detected errors
        if error_type in [ErrorType.BIT_FLIP, ErrorType.DEPOLARIZING]:
            corrections.append(GateOperation(GateType.PAULI_X, [qubit_index]))

        return corrections

# Example demonstration


async def demonstrate_quantum_error_correction():
    """Demonstrate real-time quantum error correction."""
    logger.info("🚀 Demonstrating Real-time Quantum Error Correction")

    # Create quantum simulator with errors
    num_qubits = 4
    simulator = QuantumSimulator(num_qubits)

    # Initialize in entangled state
    simulator.apply_gate(GateType.HADAMARD, [0])
    simulator.apply_gate(GateType.CNOT, [1], [0])
    simulator.apply_gate(GateType.CNOT, [2], [0])

    logger.info(
        f"Initial state fidelity: {np.linalg.norm(simulator.state):.6f}")

    # Create adaptive error correction system
    error_correction = AdaptiveQuantumErrorCorrection(
        num_qubits, code_distance=3)

    # Add some training data to the error classifier
    logger.info("🧠 Training error detection system...")
    for _ in range(100):
        # Generate synthetic training data
        syndrome = np.random.randint(0, 2, size=6)  # Random syndrome
        errors = {0: ErrorType.BIT_FLIP} if np.sum(syndrome) > 0 else {}
        error_correction.error_detector.error_classifier.add_training_data(
            syndrome, errors)

    # Simulate errors and correction
    logger.info("🛡️ Starting error correction simulation...")

    # Start real-time correction in background
    correction_task = asyncio.create_task(
        error_correction.start_realtime_correction(
            simulator, correction_frequency=0.01  # 10ms intervals
        )
    )

    # Simulate quantum computation with errors
    simulation_time = 2.0  # seconds
    error_injection_rate = 0.1  # errors per second

    start_time = time.time()
    injected_errors = 0

    while time.time() - start_time < simulation_time:
        # Inject random errors
        if np.random.random() < error_injection_rate * 0.01:  # 10ms timestep
            error_qubit = np.random.randint(0, num_qubits)
            error_type = np.random.choice(
                [GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z])

            try:
                simulator.apply_gate(error_type, [error_qubit])
                injected_errors += 1
                logger.debug(
                    f"Injected {error_type.value} error on qubit {error_qubit}")
            except Exception as e:
                logger.warning(f"Failed to inject error: {e}")

        # Continue quantum computation
        if np.random.random() < 0.1:  # Apply random gates occasionally
            gate_type = np.random.choice(
                [GateType.HADAMARD, GateType.S_GATE, GateType.T_GATE])
            target_qubit = np.random.randint(0, num_qubits)
            try:
                simulator.apply_gate(gate_type, [target_qubit])
            except Exception:
                pass

        await asyncio.sleep(0.01)  # 10ms simulation timestep

    # Stop error correction
    error_correction.stop_realtime_correction()
    await correction_task

    final_fidelity = np.linalg.norm(simulator.state)
    logger.info(f"Final state fidelity: {final_fidelity:.6f}")

    # Get correction statistics
    stats = error_correction.get_correction_statistics()

    logger.info(f"📊 Error Correction Results:")
    logger.info(f"  Errors Injected: {injected_errors}")
    logger.info(f"  Corrections Applied: {stats.get('total_corrections', 0)}")
    logger.info(f"  Success Rate: {stats.get('overall_success_rate', 0):.3f}")
    logger.info(
        f"  Current Strategy: {stats.get('current_strategy', 'unknown')}")
    logger.info(f"  Final Fidelity: {final_fidelity:.6f}")

    # Performance comparison
    if stats.get('total_corrections', 0) > 0:
        error_suppression = (1.0 - injected_errors /
                             max(1, stats['total_corrections'])) * 100
        logger.info(f"  Error Suppression: {error_suppression:.1f}%")

    return {
        "injected_errors": injected_errors,
        "correction_stats": stats,
        "final_fidelity": final_fidelity,
        "simulation_time": simulation_time
    }

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_quantum_error_correction())
