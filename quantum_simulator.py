#!/usr/bin/env python3
"""
High-Fidelity NumPy-based Quantum Computing Simulator with AI-Augmented Features
==============================================================================

A comprehensive quantum computing simulator that combines classical numerical simulation
with modern AI tools for enhanced gate synthesis, state tracking, and circuit introspection.

Features:
- Complete standard gate library with arbitrary tensoring
- State vector and density matrix simulations
- Measurement collapse, decoherence models, and error injection
- Entanglement metrics computation
- AI-augmented circuit optimization
- Symbolic reasoning capabilities
- Comprehensive testing framework
- JSON schema-based output

Author: Claude (Anthropic)
Version: qsim-v1.3
Date: 2025-07-12
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import logm, sqrtm, expm
from typing import Union, Tuple, List, Dict, Any, Optional, Callable
import json
import datetime
import logging
import unittest
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import time
from functools import wraps
from cachetools import TTLCache
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Enhanced Exception Handling & Error Management
# ============================================================================

class QuantumGateError(Exception):
    """Base quantum gate exception with enhanced context and timestamping."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.datetime.now()
        self.error_id = f"QGE-{int(time.time() * 1000000) % 1000000:06d}"  # Unique error ID
        
        # Enhanced context with system information
        self.context.update({
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "python_version": sys.version_info[:3],
            "numpy_version": np.__version__
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    def __str__(self) -> str:
        return f"[{self.error_id}] {super().__str__()}"

class ConfigError(QuantumGateError):
    """Configuration loading errors."""
    pass

class StateError(QuantumGateError):
    """Quantum state errors."""
    pass

class CircuitError(QuantumGateError):
    """Quantum circuit errors."""
    pass

class ValidationError(QuantumGateError):
    """Validation errors."""
    pass

# ============================================================================
# Advanced Error Management & Condition Guards
# ============================================================================

class ConditionGuard:
    """
    Validates conditions before execution with custom predicates.
    """
    def __init__(self, predicate: Callable[[Any], bool], message: str = "Condition failed"):
        self.predicate = predicate
        self.message = message
        self.validation_count = 0
    
    def validate(self, value: Any) -> None:
        """Validate value against predicate."""
        self.validation_count += 1
        if not self.predicate(value):
            raise ValidationError(f"{self.message}: {value}", 
                                context={"validation_count": self.validation_count})
    
    def __call__(self, value: Any) -> Any:
        """Allow guard to be used as callable."""
        self.validate(value)
        return value

class BaseErrorHandler:
    """
    Centralized error handling with context, recovery strategies, and enhanced monitoring.
    """
    def __init__(self):
        self.error_count: Dict[str, int] = {}
        self.recovery_strategies: Dict[type, Callable] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.error_rate_window = 100  # Track error rate over last N operations
        self.critical_error_threshold = 10  # Critical if >10 errors in window
    
    def register_recovery(self, error_type: type, strategy: Callable) -> None:
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(self, error: Exception, context: str = "") -> Optional[Any]:
        """Handle error with registered recovery strategy and enhanced logging."""
        error_name = error.__class__.__name__
        self.error_count[error_name] = self.error_count.get(error_name, 0) + 1
        
        # Create enhanced error record
        error_record = {
            "error_type": error_name,
            "message": str(error),
            "context": context,
            "timestamp": datetime.datetime.now().isoformat(),
            "recovery_attempted": False,
            "recovery_successful": False
        }
        
        # Add error ID if available
        if hasattr(error, 'error_id'):
            error_record["error_id"] = error.error_id
        
        # Add to history
        self.error_history.append(error_record)
        
        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        # Enhanced logging with context
        logger.error(f"[{error_name}] {context}: {error}")
        
        # Check for critical error rate
        recent_errors = self.error_history[-self.error_rate_window:]
        if len(recent_errors) >= self.critical_error_threshold:
            logger.critical(f"Critical error rate detected: {len(recent_errors)} errors in last {self.error_rate_window} operations")
        
        # Try recovery strategy
        if type(error) in self.recovery_strategies:
            try:
                error_record["recovery_attempted"] = True
                result = self.recovery_strategies[type(error)](error)
                error_record["recovery_successful"] = True
                logger.info(f"Successfully recovered from {error_name}")
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_name}: {recovery_error}")
                error_record["recovery_error"] = str(recovery_error)
        
        # Re-raise if no recovery or recovery failed
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        recent_errors = self.error_history[-self.error_rate_window:]
        error_rate = len(recent_errors) / self.error_rate_window if self.error_rate_window > 0 else 0
        
        # Calculate error trends
        if len(self.error_history) >= 2:
            half_point = len(self.error_history) // 2
            first_half_errors = len(self.error_history[:half_point])
            second_half_errors = len(self.error_history[half_point:])
            trend = "increasing" if second_half_errors > first_half_errors else "decreasing" if second_half_errors < first_half_errors else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "total_errors": sum(self.error_count.values()),
            "error_breakdown": self.error_count.copy(),
            "registered_recoveries": len(self.recovery_strategies),
            "error_rate": error_rate,
            "recent_error_count": len(recent_errors),
            "error_trend": trend,
            "critical_threshold": self.critical_error_threshold,
            "is_critical": error_rate > (self.critical_error_threshold / self.error_rate_window),
            "history_size": len(self.error_history)
        }
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent errors."""
        return self.error_history[-count:] if count <= len(self.error_history) else self.error_history.copy()
    
    def clear_error_history(self) -> None:
        """Clear error history (use with caution in production)."""
        self.error_history.clear()
        self.error_count.clear()
        logger.info("Error history cleared")

# ============================================================================
# Advanced Decorators and Utilities
# ============================================================================

def retry(max_attempts: int = 3, backoff_factor: float = 0.5):
    """
    Decorator to retry a function up to max_attempts with exponential backoff.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = backoff_factor
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.warning(f"{fn.__name__} failed (attempt {attempts}): {e}")
                    if attempts >= max_attempts:
                        logger.error(f"{fn.__name__} exceeded max attempts")
                        raise
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return decorator

def enforce_max_qubits(max_qubits: int):
    """Decorator to enforce maximum qubit count."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'num_qubits') and self.num_qubits > max_qubits:
                raise QuantumGateError(f"Exceeds max qubits: {self.num_qubits} > {max_qubits}")
            return fn(self, *args, **kwargs)
        return wrapper
    return decorator

def validate_normalized(fn):
    """Decorator to validate state normalization before and after operations."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Pre-validation
        if hasattr(self, 'state'):
            norm = np.linalg.norm(self.state)
            if not np.isclose(norm, 1.0, atol=1e-6):
                raise StateError(f"State not normalized before operation: norm = {norm}")
        
        # Execute operation
        result = fn(self, *args, **kwargs)
        
        # Post-validation
        if hasattr(self, 'state'):
            norm = np.linalg.norm(self.state)
            if not np.isclose(norm, 1.0, atol=1e-6):
                raise StateError(f"State not normalized after operation: norm = {norm}")
        
        return result
    return wrapper

def validate_unitary(fn):
    """Decorator to validate that gate matrices are unitary."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Get gate matrix from arguments or return value
        result = fn(*args, **kwargs)
        
        if isinstance(result, np.ndarray) and result.ndim == 2:
            # Check if result is unitary
            n = result.shape[0]
            identity = np.eye(n, dtype=complex)
            product = result @ result.conj().T
            if not np.allclose(product, identity, atol=1e-10):
                raise ValidationError(f"Matrix is not unitary in {fn.__name__}")
        
        return result
    return wrapper

def with_error_handler(handler: BaseErrorHandler):
    """Decorator to wrap functions with error handling."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                return handler.handle_error(e, f"in {fn.__name__}")
        return wrapper
    return decorator

# ============================================================================
# Advanced Caching System
# ============================================================================

class GateType(Enum):
    """Enumeration of supported quantum gate types."""
    PAULI_X = "X"
    PAULI_Y = "Y" 
    PAULI_Z = "Z"
    HADAMARD = "H"
    S_GATE = "S"
    T_GATE = "T"
    ROTATION_X = "RX"
    ROTATION_Y = "RY"
    ROTATION_Z = "RZ"
    U3_GATE = "U3"
    CNOT = "CX"
    CZ = "CZ"
    SWAP = "SWAP"
    IDENTITY = "I"

class AdvancedCacheManager:
    """
    Dual-layer caching system with LRU and TTL for unitaries.
    """
    
    def __init__(self, maxsize: int = 512, ttl: int = 600):
        self._lru_cache = {}
        self._ttl_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached unitary matrix."""
        # Check LRU first
        if key in self._lru_cache:
            return self
    
    @retry(max_attempts=3, backoff_factor=0.2)
    async def apply_async(self, gate_type: GateType, target_qubits: List[int], 
                         control_qubits: Optional[List[int]] = None, 
                         parameters: Optional[List[float]] = None) -> 'QuantumSimulator':
        """
        Asynchronous version of apply_gate with enhanced error handling.
        
        Args:
            gate_type: Type of gate to apply
            target_qubits: Target qubit indices
            control_qubits: Control qubit indices (for controlled gates)
            parameters: Gate parameters (for parameterized gates)
            
        Returns:
            Self for method chaining
        """
        return self.apply_gate(gate_type, target_qubits, control_qubits, parameters)
    
    async def apply_batch_async(self, operations: List[Tuple[GateType, List[int], Optional[List[int]], Optional[List[float]]]]) -> 'QuantumSimulator':
        """
        Apply multiple gates asynchronously with concurrent execution.
        
        Args:
            operations: List of (gate_type, target_qubits, control_qubits, parameters) tuples
            
        Returns:
            Self for method chaining
        """
        try:
            # Apply operations sequentially (for now - can be made truly concurrent)
            for gate_type, target_qubits, control_qubits, parameters in operations:
                await self.apply_async(gate_type, target_qubits, control_qubits, parameters)
            
            logger.info(f"Successfully applied batch of {len(operations)} operations")
            return self
            
        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            raise CircuitError(f"Batch application failed: {e}")
    
    @retry(max_attempts=3)
    async def integrate_hamiltonian(self, hamiltonian: Callable[[float], np.ndarray],
                                   t0: float, t1: float, steps: int = 100) -> 'QuantumSimulator':
        """
        Numerically integrate state evolution under a time-dependent Hamiltonian H(t).
        Uses first-order Trotter decomposition: U = exp(-i H dt).
        
        Args:
            hamiltonian: Function returning Hamiltonian matrix at time t
            t0: Start time
            t1: End time  
            steps: Number of integration steps
            
        Returns:
            Self for method chaining
            
        Time Complexity: O(steps * 8^n)
        """
        # Input validation
        if t1 <= t0:
            raise ValueError("End time must be greater than start time")
        if steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        dt = (t1 - t0) / steps
        
        try:
            for i in range(steps):
                t = t0 + i * dt
                H = hamiltonian(t)
                
                # Validate Hamiltonian is Hermitian
                if not np.allclose(H, H.conj().T, atol=1e-10):
                    logger.warning(f"Hamiltonian at t={t} is not Hermitian")
                
                # Time evolution operator
                U = expm(-1j * H * dt)
                
                # Apply to state
                if self.simulation_type == SimulationType.STATE_VECTOR:
                    self.state = U @ self.state
                    # Ensure normalization
                    norm = np.linalg.norm(self.state)
                    if norm > 1e-15:
                        self.state = self.state / norm
                else:
                    # For density matrix: ρ' = UρU†
                    self.state = U @ self.state @ U.conj().T
            
            # Record operation
            operation = GateOperation(
                gate_type=GateType.IDENTITY,  # Placeholder for custom operation
                target_qubits=list(range(self.num_qubits)),
                parameters=[t0, t1, float(steps)]
            )
            self.circuit_history.append(operation)
            
            logger.info(f"Hamiltonian integration completed: t={t0} to {t1}, steps={steps}")
            return self
            
        except Exception as e:
            raise CircuitError(f"Hamiltonian integration failed: {e}")
    
    @retry(max_attempts=3) 
    def induce_operator(self, operator: np.ndarray, target_qubits: List[int]) -> 'QuantumSimulator':
        """
        Apply an induced unitary generated by the given operator: U = exp(-i * operator).
        
        Args:
            operator: Operator matrix to induce unitary from
            target_qubits: Target qubit indices
            
        Returns:
            Self for method chaining
            
        Time Complexity: O(8^n) for matrix exponentiation and application
        """
        try:
            # Validate operator is Hermitian (should be for physical operators)
            if not np.allclose(operator, operator.conj().T, atol=1e-10):
                logger.warning("Operator is not Hermitian - may not represent physical observable")
            
            # Generate unitary from operator
            U = expm(-1j * operator)
            
            # Validate resulting matrix is unitary
            if not self._validate_unitary(U):
                raise ValidationError("Induced operator does not generate unitary matrix")
            
            # Tensor to full system
            full_gate = self._tensor_gate_to_system(U, target_qubits, None)
            
            # Apply to quantum state
            if self.simulation_type == SimulationType.STATE_VECTOR:
                self.state = full_gate @ self.state
                # Ensure normalization
                norm = np.linalg.norm(self.state)
                if norm > 1e-15:
                    self.state = self.state / norm
            else:
                # For density matrix: ρ' = UρU†
                self.state = full_gate @ self.state @ full_gate.conj().T
            
            # Record operation
            operation = GateOperation(
                gate_type=GateType.IDENTITY,  # Placeholder for induced operation
                target_qubits=target_qubits,
                matrix=U
            )
            self.circuit_history.append(operation)
            
            logger.info(f"Operator induction applied to qubits {target_qubits}")
            return self
            
        except Exception as e:
            raise CircuitError(f"Operator induction failed: {e}")
        # Check TTL cache
        if key in self._ttl_cache:
            matrix = self._ttl_cache[key]
            self._lru_cache[key] = matrix  # Promote to LRU
            return matrix
        return None
    
    def put(self, key: str, matrix: np.ndarray):
        """Cache unitary matrix."""
        self._lru_cache[key] = matrix
        self._ttl_cache[key] = matrix
    
    def clear(self):
        """Clear all caches."""
        self._lru_cache.clear()
        self._ttl_cache.clear()

# Global cache manager
cache_manager = AdvancedCacheManager()

class GateType(Enum):
    """Enumeration of supported quantum gate types."""
    PAULI_X = "X"
    PAULI_Y = "Y" 
    PAULI_Z = "Z"
    HADAMARD = "H"
    S_GATE = "S"
    T_GATE = "T"
    ROTATION_X = "RX"
    ROTATION_Y = "RY"
    ROTATION_Z = "RZ"
    U3_GATE = "U3"
    CNOT = "CX"
    CZ = "CZ"
    SWAP = "SWAP"
    IDENTITY = "I"

class SimulationType(Enum):
    """Type of quantum simulation."""
    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"

@dataclass
class QuantumState:
    """Container for quantum state information."""
    amplitudes: np.ndarray
    num_qubits: int
    simulation_type: SimulationType
    is_normalized: bool = True
    
@dataclass
class GateOperation:
    """Container for gate operation metadata."""
    gate_type: GateType
    target_qubits: List[int]
    control_qubits: Optional[List[int]] = None
    parameters: Optional[List[float]] = None
    matrix: Optional[np.ndarray] = None

@dataclass
class SimulationResult:
    """Container for simulation results with JSON schema envelope."""
    timestamp: str
    version: str
    status: str
    payload: Dict[str, Any]
    debug: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# Core Quantum Gate Library
# ============================================================================

class QuantumGates:
    """
    High-fidelity quantum gate library with complete standard gate set.
    All gates are implemented as unitary matrices with validation.
    
    Time Complexity: O(1) for single gates, O(2^n) for n-qubit tensoring
    """
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """
        Pauli-X gate (NOT gate).
        
        Returns:
            2x2 unitary matrix representing X gate
            
        Time Complexity: O(1)
        """
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod  
    def pauli_y() -> np.ndarray:
        """
        Pauli-Y gate.
        
        Returns:
            2x2 unitary matrix representing Y gate
            
        Time Complexity: O(1)
        """
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """
        Pauli-Z gate.
        
        Returns:
            2x2 unitary matrix representing Z gate
            
        Time Complexity: O(1)
        """
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """
        Hadamard gate (creates superposition).
        
        Returns:
            2x2 unitary matrix representing H gate
            
        Time Complexity: O(1)
        """
        return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    
    @staticmethod
    def s_gate() -> np.ndarray:
        """
        S gate (phase gate, sqrt(Z)).
        
        Returns:
            2x2 unitary matrix representing S gate
            
        Time Complexity: O(1)
        """
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    
    @staticmethod
    def t_gate() -> np.ndarray:
        """
        T gate (π/8 gate, sqrt(S)).
        
        Returns:
            2x2 unitary matrix representing T gate
            
        Time Complexity: O(1)
        """
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """
        Rotation around X-axis by angle theta using matrix exponential.
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            2x2 unitary matrix representing RX(theta)
            
        Time Complexity: O(1)
        """
        return expm(-1j * theta/2 * np.array([[0, 1], [1, 0]], dtype=complex))
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """
        Rotation around Y-axis by angle theta.
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            2x2 unitary matrix representing RY(theta)
            
        Time Complexity: O(1)
        """
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """
        Rotation around Z-axis by angle theta using matrix exponential.
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            2x2 unitary matrix representing RZ(theta)
            
        Time Complexity: O(1)
        """
        return expm(-1j * theta/2 * np.array([[1, 0], [0, -1]], dtype=complex))
    
    @staticmethod
    def u3_gate(theta: float, phi: float, lam: float) -> np.ndarray:
        """
        Universal single-qubit gate with 3 parameters.
        
        Args:
            theta: Rotation angle
            phi: Phase angle 1
            lam: Phase angle 2
            
        Returns:
            2x2 unitary matrix representing U3(theta, phi, lambda)
            
        Time Complexity: O(1)
        """
        return np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def cnot() -> np.ndarray:
        """
        Controlled-NOT (CNOT) gate.
        
        Returns:
            4x4 unitary matrix representing CNOT gate
            
        Time Complexity: O(1)
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def cz() -> np.ndarray:
        """
        Controlled-Z gate.
        
        Returns:
            4x4 unitary matrix representing CZ gate
            
        Time Complexity: O(1)
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    
    @staticmethod
    def swap() -> np.ndarray:
        """
        SWAP gate.
        
        Returns:
            4x4 unitary matrix representing SWAP gate
            
        Time Complexity: O(1)
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def identity(n: int = 1) -> np.ndarray:
        """
        Identity gate for n qubits.
        
        Args:
            n: Number of qubits
            
        Returns:
            2^n x 2^n identity matrix
            
        Time Complexity: O(4^n)
        """
        return np.eye(2**n, dtype=complex)

# ============================================================================
# Quantum State Management
# ============================================================================

class QuantumSimulator:
    """
    High-fidelity quantum simulator supporting both state vector and density matrix representations.
    
    Features:
    - Multi-qubit state evolution
    - Arbitrary gate tensoring via Kronecker products
    - Measurement simulation with collapse
    - Decoherence and error modeling
    - Entanglement analysis
    """
    
    def __init__(self, num_qubits: int, simulation_type: SimulationType = SimulationType.STATE_VECTOR):
        """
        Initialize quantum simulator.
        
        Args:
            num_qubits: Number of qubits in the system
            simulation_type: Type of simulation (state vector or density matrix)
            
        Time Complexity: O(2^num_qubits) for state vector, O(4^num_qubits) for density matrix
        """
        self.num_qubits = num_qubits
        self.simulation_type = simulation_type
        self.gates = QuantumGates()
        
        # Initialize quantum state
        if simulation_type == SimulationType.STATE_VECTOR:
            self.state = self._initialize_state_vector()
        else:
            self.state = self._initialize_density_matrix()
            
        self.circuit_history: List[GateOperation] = []
        self._validate_state()
        
    def _initialize_state_vector(self) -> np.ndarray:
        """Initialize state vector in |00...0⟩ state."""
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0  # |00...0⟩ state
        return state
    
    def _initialize_density_matrix(self) -> np.ndarray:
        """Initialize density matrix in |00...0⟩⟨00...0| state."""
        dim = 2**self.num_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0  # |00...0⟩⟨00...0|
        return rho
    
    def _validate_state(self) -> bool:
        """
        Validate quantum state normalization and properties.
        
        Returns:
            True if state is valid
            
        Time Complexity: O(2^n) for state vector, O(4^n) for density matrix
        """
        if self.simulation_type == SimulationType.STATE_VECTOR:
            norm = np.linalg.norm(self.state)
            is_normalized = np.isclose(norm, 1.0, atol=1e-10)
            if not is_normalized:
                warnings.warn(f"State vector not normalized: norm = {norm}")
            return is_normalized
        else:
            # Check trace = 1 and hermiticity for density matrix
            trace = np.trace(self.state)
            is_trace_one = np.isclose(trace, 1.0, atol=1e-10)
            is_hermitian = np.allclose(self.state, self.state.conj().T, atol=1e-10)
            
            if not is_trace_one:
                warnings.warn(f"Density matrix trace not 1: trace = {trace}")
            if not is_hermitian:
                warnings.warn("Density matrix not Hermitian")
                
            return is_trace_one and is_hermitian
    
    def _validate_unitary(self, matrix: np.ndarray) -> bool:
        """
        Validate that a matrix is unitary.
        
        Args:
            matrix: Matrix to validate
            
        Returns:
            True if matrix is unitary
            
        Time Complexity: O(n^3) where n is matrix dimension
        """
        n = matrix.shape[0]
        identity = np.eye(n, dtype=complex)
        product = matrix @ matrix.conj().T
        return np.allclose(product, identity, atol=1e-10)
    
    def _tensor_gate_to_system(self, gate: np.ndarray, target_qubits: List[int], 
                              control_qubits: Optional[List[int]] = None) -> np.ndarray:
        """
        Tensor a gate to the full quantum system using Kronecker products.
        
        Args:
            gate: Gate matrix to apply
            target_qubits: List of target qubit indices
            control_qubits: List of control qubit indices (for controlled gates)
            
        Returns:
            Full system gate matrix
            
        Time Complexity: O(8^n) where n is num_qubits
        """
        if control_qubits is None:
            return self._tensor_single_gate(gate, target_qubits)
        else:
            return self._tensor_controlled_gate(gate, target_qubits, control_qubits)
    
    def _tensor_single_gate(self, gate: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Tensor single gate to system."""
        dim = 2**self.num_qubits
        full_gate = np.eye(dim, dtype=complex)
        
        # For multi-qubit gates, use direct placement
        if len(target_qubits) > 1:
            # Handle multi-qubit gates like CNOT, SWAP
            return self._place_multi_qubit_gate(gate, target_qubits)
        
        # Single qubit gate tensoring
        target = target_qubits[0]
        gates_list = []
        
        for i in range(self.num_qubits):
            if i == target:
                gates_list.append(gate)
            else:
                gates_list.append(self.gates.identity())
        
        # Tensor product of all gates
        result = gates_list[0]
        for g in gates_list[1:]:
            result = np.kron(result, g)
            
        return result
    
    def _place_multi_qubit_gate(self, gate: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Place multi-qubit gate in correct position in full system."""
        dim = 2**self.num_qubits
        full_gate = np.eye(dim, dtype=complex)
        
        # For 2-qubit gates like CNOT, SWAP, CZ
        if len(target_qubits) == 2 and gate.shape == (4, 4):
            control_qubit, target_qubit = target_qubits[0], target_qubits[1]
            
            # Apply gate based on control and target positions
            for i in range(dim):
                binary_i = format(i, f'0{self.num_qubits}b')
                
                for j in range(dim):
                    binary_j = format(j, f'0{self.num_qubits}b')
                    
                    # Extract control and target bits
                    control_bit_i = int(binary_i[control_qubit])
                    target_bit_i = int(binary_i[target_qubit])
                    control_bit_j = int(binary_j[control_qubit])
                    target_bit_j = int(binary_j[target_qubit])
                    
                    # Check if states differ only on control/target qubits
                    other_bits_same = all(
                        binary_i[k] == binary_j[k] 
                        for k in range(self.num_qubits) 
                        if k not in target_qubits
                    )
                    
                    if other_bits_same:
                        # Map to 2-qubit gate indices
                        gate_i = control_bit_i * 2 + target_bit_i
                        gate_j = control_bit_j * 2 + target_bit_j
                        full_gate[i, j] = gate[gate_i, gate_j]
        
        return full_gate
    
    def _create_qubit_mapping(self, target_qubits: List[int]) -> Dict[int, int]:
        """Create mapping from global qubit indices to gate qubit indices."""
        return {target_qubits[i]: i for i in range(len(target_qubits))}
    
    def _map_to_gate_index(self, global_index: int, qubit_map: Dict[int, int]) -> Optional[int]:
        """Map global state index to gate-specific index."""
        # Convert global index to binary representation
        binary = format(global_index, f'0{self.num_qubits}b')
        
        # Extract bits for target qubits
        gate_bits = []
        for qubit in sorted(qubit_map.keys()):
            gate_bits.append(binary[qubit])
        
        # Convert back to integer
        if gate_bits:
            return int(''.join(gate_bits), 2)
        return None
    
    def _tensor_controlled_gate(self, gate: np.ndarray, target_qubits: List[int], 
                               control_qubits: List[int]) -> np.ndarray:
        """Tensor controlled gate to system."""
        dim = 2**self.num_qubits
        full_gate = np.eye(dim, dtype=complex)
        
        # For each computational basis state
        for i in range(dim):
            binary = format(i, f'0{self.num_qubits}b')
            
            # Check if all control qubits are in |1⟩ state
            controls_active = all(binary[c] == '1' for c in control_qubits)
            
            if controls_active:
                # Apply gate to target qubits
                for j in range(dim):
                    binary_j = format(j, f'0{self.num_qubits}b')
                    
                    # Check if states differ only on target qubits
                    if self._states_differ_only_on_targets(binary, binary_j, target_qubits):
                        # Map to gate indices and apply gate
                        gate_i = self._extract_target_bits(binary, target_qubits)
                        gate_j = self._extract_target_bits(binary_j, target_qubits)
                        full_gate[i, j] = gate[gate_i, gate_j]
        
        return full_gate
    
    def _states_differ_only_on_targets(self, state1: str, state2: str, 
                                      target_qubits: List[int]) -> bool:
        """Check if two states differ only on target qubits."""
        for i in range(len(state1)):
            if i not in target_qubits and state1[i] != state2[i]:
                return False
        return True
    
    def _extract_target_bits(self, binary_state: str, target_qubits: List[int]) -> int:
        """Extract bits corresponding to target qubits."""
        bits = [binary_state[q] for q in sorted(target_qubits)]
        return int(''.join(bits), 2)
    
    def apply_gate(self, gate_type: GateType, target_qubits: List[int], 
                   control_qubits: Optional[List[int]] = None, 
                   parameters: Optional[List[float]] = None) -> 'QuantumSimulator':
        """
        Apply quantum gate to the system.
        
        Args:
            gate_type: Type of gate to apply
            target_qubits: Target qubit indices
            control_qubits: Control qubit indices (for controlled gates)
            parameters: Gate parameters (for parameterized gates)
            
        Returns:
            Self for method chaining
            
        Time Complexity: O(8^n) for gate construction + O(8^n) for application
        """
        # Validate qubit indices
        all_qubits = target_qubits + (control_qubits or [])
        for qubit in all_qubits:
            if not (0 <= qubit < self.num_qubits):
                raise IndexError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
        
        # Get gate matrix
        gate_matrix = self._get_gate_matrix(gate_type, parameters)
        
        # Validate gate is unitary
        if not self._validate_unitary(gate_matrix):
            raise ValueError(f"Gate {gate_type.value} is not unitary")
        
        # Tensor to full system
        full_gate = self._tensor_gate_to_system(gate_matrix, target_qubits, control_qubits)
        
        # Apply to quantum state
        if self.simulation_type == SimulationType.STATE_VECTOR:
            self.state = full_gate @ self.state
            # Ensure normalization (for numerical stability)
            norm = np.linalg.norm(self.state)
            if norm > 1e-15:
                self.state = self.state / norm
        else:
            # For density matrix: ρ' = UρU†
            self.state = full_gate @ self.state @ full_gate.conj().T
        
        # Record operation
        operation = GateOperation(
            gate_type=gate_type,
            target_qubits=target_qubits,
            control_qubits=control_qubits,
            parameters=parameters,
            matrix=gate_matrix
        )
        self.circuit_history.append(operation)
        
        # Validate state after operation
        self._validate_state()
        
        return self
    
    def _get_gate_matrix(self, gate_type: GateType, 
                        parameters: Optional[List[float]] = None) -> np.ndarray:
        """Get gate matrix for specified gate type."""
        if gate_type == GateType.PAULI_X:
            return self.gates.pauli_x()
        elif gate_type == GateType.PAULI_Y:
            return self.gates.pauli_y()
        elif gate_type == GateType.PAULI_Z:
            return self.gates.pauli_z()
        elif gate_type == GateType.HADAMARD:
            return self.gates.hadamard()
        elif gate_type == GateType.S_GATE:
            return self.gates.s_gate()
        elif gate_type == GateType.T_GATE:
            return self.gates.t_gate()
        elif gate_type == GateType.ROTATION_X:
            if parameters is None or len(parameters) != 1:
                raise ValueError("RX gate requires exactly one parameter (theta)")
            return self.gates.rotation_x(parameters[0])
        elif gate_type == GateType.ROTATION_Y:
            if parameters is None or len(parameters) != 1:
                raise ValueError("RY gate requires exactly one parameter (theta)")
            return self.gates.rotation_y(parameters[0])
        elif gate_type == GateType.ROTATION_Z:
            if parameters is None or len(parameters) != 1:
                raise ValueError("RZ gate requires exactly one parameter (theta)")
            return self.gates.rotation_z(parameters[0])
        elif gate_type == GateType.U3_GATE:
            if parameters is None or len(parameters) != 3:
                raise ValueError("U3 gate requires exactly three parameters (theta, phi, lambda)")
            return self.gates.u3_gate(parameters[0], parameters[1], parameters[2])
        elif gate_type == GateType.CNOT:
            return self.gates.cnot()
        elif gate_type == GateType.CZ:
            return self.gates.cz()
        elif gate_type == GateType.SWAP:
            return self.gates.swap()
        elif gate_type == GateType.IDENTITY:
            return self.gates.identity()
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")
    
    def measure(self, qubit_indices: Optional[List[int]] = None, 
               basis: str = 'computational') -> Tuple[List[int], 'QuantumSimulator']:
        """
        Perform quantum measurement with state collapse.
        
        Args:
            qubit_indices: Qubits to measure (None for all qubits)
            basis: Measurement basis ('computational', 'x', 'y')
            
        Returns:
            Tuple of (measurement_results, collapsed_simulator)
            
        Time Complexity: O(2^n) for probability calculation + O(2^n) for collapse
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # Transform to measurement basis if needed
        if basis != 'computational':
            self._transform_to_measurement_basis(qubit_indices, basis)
        
        # Calculate measurement probabilities
        if self.simulation_type == SimulationType.STATE_VECTOR:
            probabilities = self._calculate_sv_probabilities()
        else:
            probabilities = self._calculate_dm_probabilities()
        
        # Sample measurement outcome
        outcome_index = np.random.choice(len(probabilities), p=probabilities)
        outcome_bits = self._index_to_binary(outcome_index, self.num_qubits)
        
        # Extract measured bits
        measured_bits = [int(outcome_bits[i]) for i in qubit_indices]
        
        # Collapse state
        collapsed_sim = self._collapse_state(outcome_index, probabilities[outcome_index])
        
        return measured_bits, collapsed_sim
    
    def _transform_to_measurement_basis(self, qubit_indices: List[int], basis: str):
        """Transform qubits to specified measurement basis."""
        for qubit in qubit_indices:
            if basis == 'x':
                self.apply_gate(GateType.HADAMARD, [qubit])
            elif basis == 'y':
                # Y basis: apply S† then H
                self.apply_gate(GateType.S_GATE, [qubit])  # S
                # S† = S^3 for S gate
                self.apply_gate(GateType.S_GATE, [qubit])  # S²
                self.apply_gate(GateType.S_GATE, [qubit])  # S³ = S†
                self.apply_gate(GateType.HADAMARD, [qubit])
    
    def _calculate_sv_probabilities(self) -> np.ndarray:
        """Calculate measurement probabilities for state vector."""
        return np.abs(self.state)**2
    
    def _calculate_dm_probabilities(self) -> np.ndarray:
        """Calculate measurement probabilities for density matrix."""
        return np.real(np.diag(self.state))
    
    def _index_to_binary(self, index: int, num_bits: int) -> str:
        """Convert index to binary string."""
        return format(index, f'0{num_bits}b')
    
    def _collapse_state(self, outcome_index: int, probability: float) -> 'QuantumSimulator':
        """Collapse quantum state to measured outcome."""
        collapsed_sim = QuantumSimulator(self.num_qubits, self.simulation_type)
        
        if self.simulation_type == SimulationType.STATE_VECTOR:
            # Project to measured state and renormalize
            collapsed_sim.state = np.zeros_like(self.state)
            collapsed_sim.state[outcome_index] = 1.0
        else:
            # Project density matrix
            collapsed_sim.state = np.zeros_like(self.state)
            collapsed_sim.state[outcome_index, outcome_index] = 1.0
        
        return collapsed_sim
    
    def get_eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get eigendecomposition of the current state.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors)
            
        Time Complexity: O(8^n) for eigenvalue decomposition
        """
        if self.simulation_type == SimulationType.STATE_VECTOR:
            # For state vector, create density matrix first
            rho = np.outer(self.state, self.state.conj())
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self.state)
        
        return eigenvalues, eigenvectors
    
    def compute_entanglement_entropy(self, subsystem_qubits: List[int]) -> float:
        """
        Compute von Neumann entropy of subsystem (entanglement measure).
        
        Args:
            subsystem_qubits: Qubits in the subsystem
            
        Returns:
            Von Neumann entropy S = -Tr(ρ_A log ρ_A)
            
        Time Complexity: O(8^n) for partial trace + O(4^k) for log computation
        """
        # Get reduced density matrix for subsystem
        rho_sub = self._partial_trace(subsystem_qubits)
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(rho_sub)
        
        # Filter out zero eigenvalues to avoid log(0)
        eigenvals = eigenvals[eigenvals > 1e-15]
        
        # Compute von Neumann entropy
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        
        return float(np.real(entropy))  # Take real part to handle numerical precision
    
    def _partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """
        Compute partial trace over specified qubits.
        
        Args:
            keep_qubits: Qubits to keep in the reduced system
            
        Returns:
            Reduced density matrix
            
        Time Complexity: O(8^n)
        """
        if self.simulation_type == SimulationType.STATE_VECTOR:
            # Convert to density matrix first
            full_rho = np.outer(self.state, self.state.conj())
        else:
            full_rho = self.state
        
        # Get qubits to trace out
        trace_qubits = [i for i in range(self.num_qubits) if i not in keep_qubits]
        trace_qubits.sort(reverse=True)  # Trace out from highest index first
        
        # Perform partial trace iteratively
        rho_reduced = full_rho.copy()
        current_num_qubits = self.num_qubits
        
        for qubit in trace_qubits:
            rho_reduced = self._trace_out_qubit(rho_reduced, qubit, current_num_qubits)
            current_num_qubits -= 1
        
        return rho_reduced
    
    def _trace_out_qubit(self, rho: np.ndarray, qubit_index: int, total_qubits: int) -> np.ndarray:
        """Trace out a single qubit from density matrix."""
        dim = 2**total_qubits
        reduced_dim = dim // 2
        
        if rho.shape[0] != dim:
            raise ValueError(f"Matrix dimension {rho.shape[0]} doesn't match expected {dim}")
        
        rho_reduced = np.zeros((reduced_dim, reduced_dim), dtype=complex)
        
        # Create mapping for indices after removing one qubit
        for i in range(reduced_dim):
            for j in range(reduced_dim):
                # Sum over traced qubit states |0⟩ and |1⟩
                i0 = self._insert_bit_at_position(i, 0, qubit_index, total_qubits)
                i1 = self._insert_bit_at_position(i, 1, qubit_index, total_qubits)
                j0 = self._insert_bit_at_position(j, 0, qubit_index, total_qubits)
                j1 = self._insert_bit_at_position(j, 1, qubit_index, total_qubits)
                
                rho_reduced[i, j] = rho[i0, j0] + rho[i1, j1]
        
        return rho_reduced
    
    def _insert_bit_at_position(self, index: int, bit: int, position: int, total_bits: int) -> int:
        """Insert a bit at specified position in binary representation."""
        if total_bits == 1:
            return bit
        
        # Convert reduced index to binary string
        binary_str = format(index, f'0{total_bits-1}b')
        
        # Insert the bit at the specified position (from left, 0-indexed)
        if position == 0:
            new_binary = str(bit) + binary_str
        elif position >= len(binary_str):
            new_binary = binary_str + str(bit)
        else:
            new_binary = binary_str[:position] + str(bit) + binary_str[position:]
        
        return int(new_binary, 2)
    
    def compute_schmidt_coefficients(self, subsystem_a: List[int]) -> np.ndarray:
        """
        Compute Schmidt coefficients for bipartite entanglement.
        
        Args:
            subsystem_a: Qubits in subsystem A
            
        Returns:
            Schmidt coefficients (square roots of eigenvalues of reduced density matrix)
            
        Time Complexity: O(8^n) for partial trace + O(4^k) for eigendecomposition
        """
        rho_a = self._partial_trace(subsystem_a)
        eigenvals = np.linalg.eigvals(rho_a)
        eigenvals = np.real(eigenvals)  # Take real part for numerical stability
        schmidt_coeffs = np.sqrt(np.maximum(eigenvals, 0))  # Ensure non-negative
        return np.sort(schmidt_coeffs)[::-1]  # Sort in descending order
    
    def get_state_json(self) -> Dict[str, Any]:
        """
        Get current quantum state in JSON-serializable format.
        
        Returns:
            Dictionary containing state information
        """
        if self.simulation_type == SimulationType.STATE_VECTOR:
            probabilities = np.abs(self.state)**2
            state_data = {
                'amplitudes_real': self.state.real.tolist(),
                'amplitudes_imag': self.state.imag.tolist(),
                'probabilities': probabilities.tolist()
            }

# ============================================================================
# Advanced Circuit Management Features
# ============================================================================

class CircuitSnapshot:
    """
    Immutable snapshot of quantum circuit state for rollback functionality.
    """
    
    def __init__(self, state: np.ndarray, history: List[GateOperation], 
                 timestamp: datetime.datetime):
        self.state = state.copy()
        self.history = history.copy()
        self.timestamp = timestamp
        self._hash = hash(state.tobytes())
    
    def __hash__(self) -> int:
        return self._hash

class RollbackContext:
    """
    Context manager for transaction-like quantum operations with automatic rollback.
    """
    
    def __init__(self, simulator: 'QuantumSimulator'):
        self.simulator = simulator
        self._snapshot: Optional[CircuitSnapshot] = None
    
    def __enter__(self) -> 'QuantumSimulator':
        """Create snapshot on enter."""
        self._snapshot = CircuitSnapshot(
            self.simulator.state,
            self.simulator.circuit_history,
            datetime.datetime.now()
        )
        logger.info("Circuit snapshot created for transaction")
        return self.simulator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rollback on exception, commit on success."""
        if exc_type is not None:
            # Exception occurred, rollback
            if self._snapshot:
                self.simulator.state = self._snapshot.state.copy()
                self.simulator.circuit_history = self._snapshot.history.copy()
                logger.warning(f"Circuit rolled back due to {exc_type.__name__}: {exc_val}")
        else:
            # Success, commit
            logger.info("Circuit transaction committed successfully")

class CircuitProfiler:
    """
    Advanced circuit profiler for performance analysis.
    """
    
    def __init__(self):
        self.gate_timings: Dict[str, List[float]] = {}
        self.operation_count: Dict[str, int] = {}
    
    def profile_gate(self, gate_type: GateType, operation_func: Callable[[], Any]) -> Any:
        """Profile a gate operation and record timing."""
        start_time = time.time()
        try:
            result = operation_func()
            elapsed = time.time() - start_time
            
            gate_name = gate_type.value
            if gate_name not in self.gate_timings:
                self.gate_timings[gate_name] = []
                self.operation_count[gate_name] = 0
            
            self.gate_timings[gate_name].append(elapsed)
            self.operation_count[gate_name] += 1
            
            return result
        except Exception as e:
            logger.error(f"Gate {gate_type.value} failed during profiling: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all gates."""
        stats = {}
        for gate_name, timings in self.gate_timings.items():
            if timings:
                stats[gate_name] = {
                    'count': self.operation_count[gate_name],
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'avg_time': sum(timings) / len(timings),
                    'total_time': sum(timings)
                }
        return stats

class AutoParameterOptimizer:
    """
    Automatic parameter optimization for quantum gates.
    """
    
    def __init__(self, simulator: 'QuantumSimulator', cost_function: Callable[[np.ndarray], float]):
        self.simulator = simulator
        self.cost_function = cost_function
    
    @retry(max_attempts=3)
    def optimize_parameter(self, gate_type: GateType, target_qubits: List[int], 
                          initial_param: float, bounds: Tuple[float, float] = (-np.pi, np.pi),
                          tolerance: float = 1e-6) -> float:
        """
        Optimize a single parameter using gradient-free optimization.
        
        Args:
            gate_type: Gate type to optimize
            target_qubits: Target qubits for the gate
            initial_param: Initial parameter value
            bounds: Parameter bounds (min, max)
            tolerance: Optimization tolerance
            
        Returns:
            Optimized parameter value
        """
        def objective(param: float) -> float:
            """Objective function for optimization."""
            with RollbackContext(self.simulator):
                try:
                    self.simulator.apply_gate(gate_type, target_qubits, parameters=[param])
                    return self.cost_function(self.simulator.state)
                except Exception as e:
                    logger.warning(f"Parameter optimization failed for {param}: {e}")
                    return float('inf')
        
        # Simple grid search optimization (can be replaced with scipy.optimize)
        best_param = initial_param
        best_cost = objective(initial_param)
        
        # Grid search over parameter space
        min_bound, max_bound = bounds
        step_size = (max_bound - min_bound) / 100
        
        for param in np.arange(min_bound, max_bound, step_size):
            cost = objective(param)
            if cost < best_cost:
                best_cost = cost
                best_param = param
        
        logger.info(f"Parameter optimization complete: {initial_param} -> {best_param} (cost: {best_cost})")
        return best_param

class AdvancedNoiseModel:
    """
    Advanced noise modeling with multiple channels and adaptive injection.
    """
    
    def __init__(self, base_error_rate: float = 0.001):
        self.base_error_rate = base_error_rate
        self.error_history: List[Tuple[str, float, datetime.datetime]] = []
    
    def inject_adaptive_noise(self, simulator: 'QuantumSimulator', 
                             gate_count_factor: float = 0.1) -> None:
        """
        Inject noise that scales with circuit depth and complexity.
        
        Args:
            simulator: Quantum simulator instance
            gate_count_factor: Factor to scale noise with gate count
        """
        gate_count = len(simulator.circuit_history)
        effective_error_rate = self.base_error_rate * (1 + gate_count * gate_count_factor)
        
        # Apply noise to each qubit with some probability
        for qubit in range(simulator.num_qubits):
            if np.random.random() < effective_error_rate:
                # Choose random Pauli error
                error_type = np.random.choice([GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z])
                
                try:
                    simulator.apply_gate(error_type, [qubit])
                    self.error_history.append((error_type.value, effective_error_rate, datetime.datetime.now()))
                    logger.debug(f"Injected {error_type.value} error on qubit {qubit}")
                except Exception as e:
                    logger.warning(f"Failed to inject noise: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about injected errors."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        for error_type, _, _ in self.error_history:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "latest_error": self.error_history[-1] if self.error_history else None
        }
        diagonal_probs = np.real(np.diag(self.state))
        state_data = {
            'density_matrix_real': self.state.real.tolist(),
            'density_matrix_imag': self.state.imag.tolist(),
            'diagonal_probabilities': diagonal_probs.tolist()
        }
        
        return {
            'num_qubits': self.num_qubits,
            'simulation_type': self.simulation_type.value,
            'is_normalized': self._validate_state(),
            'state_data': state_data
        }

# ============================================================================
# Error Models and Decoherence
# ============================================================================

class NoiseModel:
    """
    Quantum noise and decoherence models.
    
    Supports various noise channels:
    - Depolarizing noise
    - Amplitude damping
    - Phase damping
    - Bit flip
    - Phase flip
    """
    
    @staticmethod
    def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
        """
        Apply depolarizing noise channel.
        
        Args:
            rho: Density matrix
            p: Depolarizing probability
            
        Returns:
            Noisy density matrix
            
        Time Complexity: O(n²) where n is matrix dimension
        """
        if not (0 <= p <= 1):
            raise ValueError("Depolarizing probability must be between 0 and 1")
        
        dim = rho.shape[0]
        identity = np.eye(dim, dtype=complex) / dim
        
        return (1 - p) * rho + p * identity
    
    @staticmethod
    def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply amplitude damping noise channel (models energy loss).
        
        Args:
            rho: Density matrix (must be 2x2 for single qubit)
            gamma: Damping probability
            
        Returns:
            Noisy density matrix
            
        Time Complexity: O(1) for single qubit
        """
        if not (0 <= gamma <= 1):
            raise ValueError("Damping probability must be between 0 and 1")
        
        if rho.shape != (2, 2):
            raise ValueError("Amplitude damping currently only supports single qubit")
        
        # Kraus operators for amplitude damping
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        
        # Apply channel: sum_i E_i ρ E_i†
        rho_new = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        return rho_new
    
    @staticmethod
    def phase_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply phase damping noise channel (models dephasing).
        
        Args:
            rho: Density matrix (must be 2x2 for single qubit)
            gamma: Dephasing probability
            
        Returns:
            Noisy density matrix
            
        Time Complexity: O(1) for single qubit
        """
        if not (0 <= gamma <= 1):
            raise ValueError("Dephasing probability must be between 0 and 1")
        
        if rho.shape != (2, 2):
            raise ValueError("Phase damping currently only supports single qubit")
        
        # Kraus operators for phase damping
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
        
        # Apply channel
        rho_new = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        return rho_new

# ============================================================================
# AI-Augmented Features
# ============================================================================

class AIGateOptimizer:
    """
    AI-augmented gate optimization and synthesis.
    
    Features:
    - Circuit depth reduction
    - Gate sequence optimization
    - Symbolic gate decomposition
    """
    
    def __init__(self):
        self.optimization_cache: Dict[str, List[GateOperation]] = {}
    
    def optimize_circuit(self, operations: List[GateOperation]) -> List[GateOperation]:
        """
        Optimize quantum circuit using AI-augmented techniques.
        
        Args:
            operations: List of gate operations to optimize
            
        Returns:
            Optimized list of gate operations
            
        Time Complexity: O(n²) where n is number of gates
        """
        # Create circuit signature for caching
        circuit_sig = self._create_circuit_signature(operations)
        
        if circuit_sig in self.optimization_cache:
            return self.optimization_cache[circuit_sig]
        
        optimized = self._apply_optimization_rules(operations)
        self.optimization_cache[circuit_sig] = optimized
        
        return optimized
    
    def _create_circuit_signature(self, operations: List[GateOperation]) -> str:
        """Create unique signature for circuit caching."""
        sig_parts = []
        for op in operations:
            sig_parts.append(f"{op.gate_type.value}-{op.target_qubits}-{op.control_qubits}")
        return "|".join(sig_parts)
    
    def _apply_optimization_rules(self, operations: List[GateOperation]) -> List[GateOperation]:
        """Apply circuit optimization rules."""
        optimized = operations.copy()
        
        # Rule 1: Cancel adjacent inverse gates
        optimized = self._cancel_inverse_gates(optimized)
        
        # Rule 2: Merge rotation gates
        optimized = self._merge_rotation_gates(optimized)
        
        # Rule 3: Commute gates through controls
        optimized = self._commute_through_controls(optimized)
        
        return optimized
    
    def _cancel_inverse_gates(self, operations: List[GateOperation]) -> List[GateOperation]:
        """Cancel adjacent gates that are inverses of each other."""
        result = []
        i = 0
        
        while i < len(operations):
            current = operations[i]
            
            # Check if next gate cancels current gate
            if i + 1 < len(operations):
                next_gate = operations[i + 1]
                if self._are_inverse_gates(current, next_gate):
                    i += 2  # Skip both gates
                    continue
            
            result.append(current)
            i += 1
        
        return result
    
    def _are_inverse_gates(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check if two gates are inverses of each other."""
        # Same target qubits
        if gate1.target_qubits != gate2.target_qubits:
            return False
        
        # Check for specific inverse pairs
        inverse_pairs = [
            (GateType.PAULI_X, GateType.PAULI_X),
            (GateType.PAULI_Y, GateType.PAULI_Y),
            (GateType.PAULI_Z, GateType.PAULI_Z),
            (GateType.HADAMARD, GateType.HADAMARD),
            (GateType.CNOT, GateType.CNOT),
            (GateType.SWAP, GateType.SWAP)
        ]
        
        gate_pair = (gate1.gate_type, gate2.gate_type)
        return gate_pair in inverse_pairs
    
    def _merge_rotation_gates(self, operations: List[GateOperation]) -> List[GateOperation]:
        """Merge consecutive rotation gates on same qubit."""
        result = []
        i = 0
        
        while i < len(operations):
            current = operations[i]
            
            # Look for consecutive rotation gates on same axis and qubit
            if current.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
                merged_angle = current.parameters[0] if current.parameters else 0
                j = i + 1
                
                while j < len(operations):
                    next_gate = operations[j]
                    if (next_gate.gate_type == current.gate_type and 
                        next_gate.target_qubits == current.target_qubits):
                        merged_angle += next_gate.parameters[0] if next_gate.parameters else 0
                        j += 1
                    else:
                        break
                
                # Create merged gate
                if j > i + 1:  # Found gates to merge
                    merged_gate = GateOperation(
                        gate_type=current.gate_type,
                        target_qubits=current.target_qubits,
                        control_qubits=current.control_qubits,
                        parameters=[merged_angle % (2 * np.pi)]  # Normalize angle
                    )
                    result.append(merged_gate)
                    i = j
                else:
                    result.append(current)
                    i += 1
            else:
                result.append(current)
                i += 1
        
        return result
    
    def _commute_through_controls(self, operations: List[GateOperation]) -> List[GateOperation]:
        """Optimize by commuting gates through control qubits."""
        # Placeholder for more advanced commutation rules
        return operations
    
    def synthesize_unitary(self, target_unitary: np.ndarray, max_depth: int = 100) -> List[GateOperation]:
        """
        Synthesize gate sequence to approximate target unitary.
        
        Args:
            target_unitary: Target unitary matrix to synthesize
            max_depth: Maximum allowed circuit depth
            
        Returns:
            List of gate operations approximating target unitary
            
        Time Complexity: O(max_depth * n³) where n is matrix dimension
        """
        if target_unitary.shape[0] == 2:
            return self._synthesize_single_qubit_unitary(target_unitary)
        else:
            return self._synthesize_multi_qubit_unitary(target_unitary, max_depth)
    
    def _synthesize_single_qubit_unitary(self, unitary: np.ndarray) -> List[GateOperation]:
        """Synthesize single-qubit unitary using U3 decomposition."""
        # Extract Euler angles from unitary
        # U = e^(iα) R_z(β) R_y(γ) R_z(δ)
        
        # This is a simplified decomposition - full implementation would use
        # proper Euler angle extraction
        theta = 2 * np.arccos(np.abs(unitary[0, 0]))
        phi = np.angle(unitary[1, 0]) - np.angle(unitary[0, 0])
        lam = np.angle(unitary[0, 1]) - np.angle(unitary[0, 0])
        
        return [GateOperation(
            gate_type=GateType.U3_GATE,
            target_qubits=[0],
            parameters=[theta, phi, lam]
        )]
    
    def _synthesize_multi_qubit_unitary(self, unitary: np.ndarray, 
                                       max_depth: int) -> List[GateOperation]:
        """Synthesize multi-qubit unitary using decomposition techniques."""
        # Placeholder for multi-qubit synthesis
        # Would implement techniques like:
        # - Two-qubit gate decomposition
        # - Cartan decomposition
        # - Quantum Shannon decomposition
        
        operations = []
        # Simple placeholder: identity
        operations.append(GateOperation(
            gate_type=GateType.IDENTITY,
            target_qubits=[0]
        ))
        
        return operations

# ============================================================================
# JSON Schema Output System
# ============================================================================

class ResultEncoder:
    """
    Encoder for quantum simulation results with versioned JSON schema.
    """
    
    VERSION = "qsim-v1.4.1"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert NumPy arrays and complex numbers to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return {
                    'real': obj.real.tolist(),
                    'imag': obj.imag.tolist()
                }
            else:
                return obj.tolist()
        elif isinstance(obj, (np.complex64, np.complex128, complex)):
            return {
                'real': float(obj.real),
                'imag': float(obj.imag)
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: ResultEncoder._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultEncoder._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    @classmethod
    def encode_result(cls, status: str, payload: Dict[str, Any], 
                     debug: Dict[str, Any] = None, 
                     metadata: Dict[str, Any] = None) -> SimulationResult:
        """
        Encode simulation result in versioned JSON schema.
        
        Args:
            status: Result status ("success", "error", "warning")
            payload: Main result payload
            debug: Debug information
            metadata: Additional metadata
            
        Returns:
            Structured simulation result
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        if debug is None:
            debug = {}
        
        return SimulationResult(
            timestamp=timestamp,
            version=cls.VERSION,
            status=status,
            payload=payload,
            debug=debug,
            metadata=metadata
        )
    
    @classmethod
    def encode_gate_result(cls, gate_operation: GateOperation, 
                          final_state: np.ndarray, 
                          unitary_check: bool = True) -> SimulationResult:
        """Encode gate operation result."""
        
        # Compute eigenvalues for debugging
        eigenvals = None
        if gate_operation.matrix is not None:
            eigenvals = cls._make_json_serializable(np.linalg.eigvals(gate_operation.matrix))
        
        # Convert state to JSON-serializable format
        if final_state.ndim == 1:
            output_state = cls._make_json_serializable(final_state)
            output_density = None
        else:
            output_state = None
            output_density = cls._make_json_serializable(final_state)
        
        payload = {
            "gate": f"{gate_operation.gate_type.value}",
            "target_qubits": gate_operation.target_qubits,
            "control_qubits": gate_operation.control_qubits,
            "parameters": gate_operation.parameters,
            "matrix_shape": list(gate_operation.matrix.shape) if gate_operation.matrix is not None else None,
            "output_state_vector": output_state,
            "output_density_matrix": output_density
        }
        
        debug = {
            "unitary_check": unitary_check,
            "norm": float(np.linalg.norm(final_state)),
            "eigenvalues": eigenvals
        }
        
        return cls.encode_result("success", payload, debug)
    
    @classmethod
    def encode_measurement_result(cls, measured_bits: List[int], 
                                 probabilities: np.ndarray,
                                 collapsed_state: np.ndarray) -> SimulationResult:
        """Encode measurement result."""
        
        payload = {
            "measurement_outcome": measured_bits,
            "measurement_probabilities": cls._make_json_serializable(probabilities),
            "collapsed_state": cls._make_json_serializable(collapsed_state)
        }
        
        # Avoid log(0) by adding small epsilon
        safe_probs = probabilities + 1e-15
        entropy = -np.sum(probabilities * np.log2(safe_probs))
        
        debug = {
            "total_probability": float(np.sum(probabilities)),
            "entropy": float(entropy)
        }
        
        return cls.encode_result("success", payload, debug)

# ============================================================================
# Comprehensive Testing Framework
# ============================================================================

class TestQuantumSimulator(unittest.TestCase):
    """
    Comprehensive test suite for quantum simulator.
    Tests include success cases, edge cases, and error conditions.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.sim_2q = QuantumSimulator(2)
        self.sim_3q = QuantumSimulator(3)
        self.gates = QuantumGates()
    
    # Success Tests
    def test_single_qubit_gates_success(self):
        """Test successful application of single-qubit gates."""
        # Test Pauli-X gate on qubit 1 (not 0) - this flips |00⟩ to |01⟩
        self.sim_2q.apply_gate(GateType.PAULI_X, [1])
        expected = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩ state
        np.testing.assert_array_almost_equal(self.sim_2q.state, expected)
        
        # Test Hadamard gate on fresh single qubit
        sim = QuantumSimulator(1)
        sim.apply_gate(GateType.HADAMARD, [0])
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(sim.state, expected)
    
    def test_two_qubit_gates_success(self):
        """Test successful application of two-qubit gates."""
        # Test CNOT gate
        self.sim_2q.apply_gate(GateType.PAULI_X, [0])  # Prepare |10⟩
        self.sim_2q.apply_gate(GateType.CNOT, [0, 1])  # Should give |11⟩
        expected = np.array([0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(self.sim_2q.state, expected)
    
    def test_parameterized_gates_success(self):
        """Test successful application of parameterized gates."""
        # Test rotation gates
        theta = np.pi / 4
        self.sim_2q.apply_gate(GateType.ROTATION_X, [0], parameters=[theta])
        
        # Verify state norm
        self.assertAlmostEqual(np.linalg.norm(self.sim_2q.state), 1.0, places=10)
    
    def test_measurement_success(self):
        """Test successful quantum measurement."""
        # Prepare superposition
        self.sim_2q.apply_gate(GateType.HADAMARD, [0])
        
        # Measure
        bits, collapsed_sim = self.sim_2q.measure([0])
        
        # Check result is valid
        self.assertIn(bits[0], [0, 1])
        self.assertAlmostEqual(np.linalg.norm(collapsed_sim.state), 1.0, places=10)
    
    def test_entanglement_measures_success(self):
        """Test successful computation of entanglement measures."""
        # Create Bell state
        self.sim_2q.apply_gate(GateType.HADAMARD, [0])
        self.sim_2q.apply_gate(GateType.CNOT, [0, 1])
        
        # Compute entanglement entropy
        entropy = self.sim_2q.compute_entanglement_entropy([0])
        self.assertAlmostEqual(entropy, 1.0, places=5)  # Maximally entangled
        
        # Compute Schmidt coefficients
        coeffs = self.sim_2q.compute_schmidt_coefficients([0])
        expected_coeffs = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.sort(coeffs)[::-1], expected_coeffs, decimal=5)
    
    # Edge Case Tests
    def test_identity_gate_edge(self):
        """Test identity gate leaves state unchanged."""
        initial_state = self.sim_2q.state.copy()
        self.sim_2q.apply_gate(GateType.IDENTITY, [0])
        np.testing.assert_array_almost_equal(self.sim_2q.state, initial_state)
    
    def test_zero_rotation_edge(self):
        """Test zero-angle rotation gates."""
        initial_state = self.sim_2q.state.copy()
        self.sim_2q.apply_gate(GateType.ROTATION_X, [0], parameters=[0.0])
        np.testing.assert_array_almost_equal(self.sim_2q.state, initial_state)
    
    def test_full_rotation_edge(self):
        """Test 2π rotation returns to original state."""
        initial_state = self.sim_2q.state.copy()
        self.sim_2q.apply_gate(GateType.ROTATION_Z, [0], parameters=[2*np.pi])
        # Note: global phase may differ, so check probabilities
        initial_probs = np.abs(initial_state)**2
        final_probs = np.abs(self.sim_2q.state)**2
        np.testing.assert_array_almost_equal(final_probs, initial_probs)
    
    def test_large_system_edge(self):
        """Test simulator works with larger systems."""
        sim = QuantumSimulator(4)  # 16-dimensional state space
        sim.apply_gate(GateType.HADAMARD, [0])
        sim.apply_gate(GateType.CNOT, [0, 1])
        sim.apply_gate(GateType.CNOT, [1, 2])
        
        # Verify normalization
        self.assertAlmostEqual(np.linalg.norm(sim.state), 1.0, places=10)
    
    # Error Tests
    def test_invalid_qubit_index_error(self):
        """Test error handling for invalid qubit indices."""
        with self.assertRaises(IndexError):
            self.sim_2q.apply_gate(GateType.PAULI_X, [5])  # Index out of range
    
    def test_invalid_gate_parameters_error(self):
        """Test error handling for invalid gate parameters."""
        with self.assertRaises(ValueError):
            # RX gate needs exactly one parameter
            self.sim_2q.apply_gate(GateType.ROTATION_X, [0], parameters=[1.0, 2.0])
        
        with self.assertRaises(ValueError):
            # U3 gate needs exactly three parameters
            self.sim_2q.apply_gate(GateType.U3_GATE, [0], parameters=[1.0])
    
    def test_invalid_noise_parameters_error(self):
        """Test error handling for invalid noise parameters."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        
        with self.assertRaises(ValueError):
            NoiseModel.depolarizing_channel(rho, -0.1)  # Negative probability
        
        with self.assertRaises(ValueError):
            NoiseModel.depolarizing_channel(rho, 1.5)   # Probability > 1
    
    def test_unitary_validation_error(self):
        """Test validation catches non-unitary matrices."""
        # Create non-unitary matrix
        non_unitary = np.array([[1, 1], [0, 1]], dtype=complex)
        
        # Test that validation correctly identifies non-unitary matrix
        result = self.sim_2q._validate_unitary(non_unitary)
        self.assertFalse(result)  # Should return False for non-unitary matrix

def run_comprehensive_example():
    """
    Run comprehensive example demonstrating all simulator features.
    """
    print("🎯 High-Fidelity Quantum Simulator Demo")
    print("=" * 50)
    
    # 1. Basic Gate Operations
    print("\n🧪 1. Basic Gate Operations")
    sim = QuantumSimulator(2)
    
    # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    sim.apply_gate(GateType.HADAMARD, [0])
    sim.apply_gate(GateType.CNOT, [0, 1])
    
    # Encode result with proper JSON serialization
    encoder = ResultEncoder()
    bell_result = encoder.encode_gate_result(
        GateOperation(GateType.CNOT, [0, 1]), 
        sim.state
    )
    
    print(f"Bell State Created: {sim.state}")
    # Convert to JSON-serializable format before printing
    json_data = encoder._make_json_serializable(asdict(bell_result))
    print(f"JSON Result: {json.dumps(json_data, indent=2)}")
    
    # 2. Entanglement Analysis
    print("\n🔬 2. Entanglement Analysis")
    entropy = sim.compute_entanglement_entropy([0])
    schmidt_coeffs = sim.compute_schmidt_coefficients([0])
    
    print(f"Von Neumann Entropy: {entropy:.6f}")
    print(f"Schmidt Coefficients: {schmidt_coeffs}")
    
    # 3. Measurement
    print("\n📊 3. Quantum Measurement")
    bits, collapsed_sim = sim.measure([0])
    measurement_result = encoder.encode_measurement_result(
        bits, np.abs(sim.state)**2, collapsed_sim.state
    )
    
    print(f"Measured bit: {bits[0]}")
    print(f"Collapsed state: {collapsed_sim.state}")
    
    # Convert measurement result to JSON
    json_measurement = encoder._make_json_serializable(asdict(measurement_result))
    print(f"Measurement JSON: {json.dumps(json_measurement, indent=2)}")
    
    # 4. Noise Modeling
    print("\n🌊 4. Noise Modeling")
    if sim.simulation_type == SimulationType.STATE_VECTOR:
        # Convert to density matrix for noise modeling
        rho = np.outer(sim.state, sim.state.conj())
    else:
        rho = sim.state
    
    # Apply depolarizing noise
    noisy_rho = NoiseModel.depolarizing_channel(rho[:2, :2], 0.1)  # Single qubit noise
    print(f"Original single-qubit state: \n{rho[:2, :2]}")
    print(f"After depolarizing noise: \n{noisy_rho}")
    
    # 5. Circuit Optimization
    print("\n🧠 5. AI-Augmented Circuit Optimization")
    optimizer = AIGateOptimizer()
    
    # Create a circuit with redundant gates
    operations = [
        GateOperation(GateType.PAULI_X, [0]),
        GateOperation(GateType.PAULI_X, [0]),  # Redundant - should cancel
        GateOperation(GateType.ROTATION_Z, [0], parameters=[np.pi/4]),
        GateOperation(GateType.ROTATION_Z, [0], parameters=[np.pi/4])  # Should merge
    ]
    
    optimized_ops = optimizer.optimize_circuit(operations)
    print(f"Original circuit: {len(operations)} gates")
    print(f"Optimized circuit: {len(optimized_ops)} gates")
    
    # 6. Large System Demo  
    print("\n⚡ 6. Large System Demonstration")
    large_sim = QuantumSimulator(3)  # Start with 3 qubits
    
    # Create proper GHZ state |000⟩ + |111⟩
    large_sim.apply_gate(GateType.HADAMARD, [0])  # Creates |0⟩ + |1⟩ on qubit 0
    large_sim.apply_gate(GateType.CNOT, [0, 1])   # Entangles qubit 1 with qubit 0
    large_sim.apply_gate(GateType.CNOT, [0, 2])   # Entangles qubit 2 with qubit 0
    
    print(f"GHZ state: {large_sim.state}")
    
    # For GHZ state, entropy of single qubit should be log2(2) = 1 (maximally mixed)
    ghz_entropy = large_sim.compute_entanglement_entropy([0])
    print(f"3-qubit GHZ state entropy (single qubit): {ghz_entropy:.6f}")
    
    # Also compute two-qubit subsystem entropy
    ghz_entropy_2q = large_sim.compute_entanglement_entropy([0, 1])
    print(f"3-qubit GHZ state entropy (two qubits): {ghz_entropy_2q:.6f}")
    
    # Final result
    final_result = encoder.encode_result(
        "success",
        {
            "demo_completed": True,
            "total_operations": len(sim.circuit_history),
            "final_state_norm": float(np.linalg.norm(sim.state)),
            "entanglement_entropy": entropy,
            "optimization_reduction": f"{len(operations)} → {len(optimized_ops)} gates"
        },
        {
            "simulator_version": encoder.VERSION,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    print(f"\n✅ Final Demo Result:")
    json_final = encoder._make_json_serializable(asdict(final_result))
    print(json.dumps(json_final, indent=2))

if __name__ == "__main__":
    # Run comprehensive example
    run_comprehensive_example()
    
    # Run test suite
    print("\n🧪 Running Test Suite...")
    unittest.main(argv=[''], exit=False, verbosity=2)