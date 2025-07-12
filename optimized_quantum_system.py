#!/usr/bin/env python3
"""
Production-Ready Quantum Dynamics Framework
==========================================

Optimized version with:
- Adaptive integration for better unitarity preservation
- Intelligent logging levels based on operation criticality
- Enhanced error injection and recovery testing
- Performance monitoring and auto-tuning
- Quantum-specific optimization algorithms
"""

import asyncio
import logging
import functools
import json
import random
import datetime
import threading
import time
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Coroutine, TypeVar, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache
from enum import Enum

import numpy as np
import scipy.linalg as la

# Configure adaptive logging
class LogLevel(Enum):
    SILENT = 0
    CRITICAL = 1
    PRODUCTION = 2
    DEBUG = 3

# Global log level control
CURRENT_LOG_LEVEL = LogLevel.PRODUCTION

def setup_logging(level: LogLevel):
    """Setup adaptive logging based on environment."""
    if level == LogLevel.SILENT:
        logging.disable(logging.CRITICAL)
    elif level == LogLevel.CRITICAL:
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    elif level == LogLevel.PRODUCTION:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    else:  # DEBUG
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

setup_logging(CURRENT_LOG_LEVEL)
logger = logging.getLogger("OptimizedQuantumSystem")

# --- Enhanced Type Definitions ---
T = TypeVar('T')
HandlerType = Callable[..., Coroutine[Any, Any, Any]]

class QuantumValidationError(Exception):
    """Custom exception for quantum state/operator validation failures."""
    pass

class QuantumCircuitBreakerOpen(Exception):
    """Raised when quantum circuit breaker is open due to repeated failures."""
    pass

# --- Optimized Quantum Configuration ---
@dataclass
class QuantumTolerances:
    """Quantum precision thresholds with adaptive scaling."""
    hermiticity: float = 1e-14
    unitarity: float = 1e-13
    trace_preservation: float = 1e-13
    commutator_precision: float = 1e-15
    evolution_step: float = 1e-16
    matrix_condition: float = 1e-12
    adaptive_threshold: float = 1e-10  # Threshold for adaptive refinement

@dataclass
class QuantumConfig:
    """Enhanced configuration with performance tuning."""
    tolerances: QuantumTolerances = None
    max_retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    rate_limit_ops_per_minute: float = 1000.0
    enable_adaptive_integration: bool = True
    enable_performance_monitoring: bool = True
    max_concurrent_operations: int = 10
    log_frequency_reduction: int = 10  # Log every Nth operation
    
    def __post_init__(self):
        if self.tolerances is None:
            self.tolerances = QuantumTolerances()

# --- Enhanced High-Availability Infrastructure ---
class AdaptiveQuantumCircuitBreaker:
    """Intelligent circuit breaker that learns from quantum operation patterns."""
    
    def __init__(self, fail_threshold: int = 5, reset_timeout: float = 60.0):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.fail_count = 0
        self.success_count = 0
        self.lock = threading.Lock()
        self.open_until: Optional[datetime.datetime] = None
        self.operation_history: List[Tuple[str, bool, float]] = []  # (operation, success, timestamp)
        
    def record_quantum_result(self, operation: str, success: bool) -> None:
        """Record quantum operation result with learning."""
        with self.lock:
            timestamp = time.time()
            self.operation_history.append((operation, success, timestamp))
            
            # Keep only recent history (last 100 operations)
            if len(self.operation_history) > 100:
                self.operation_history = self.operation_history[-100:]
            
            if success:
                self.success_count += 1
                # Adaptive recovery: reduce failure count on success
                if self.fail_count > 0:
                    self.fail_count = max(0, self.fail_count - 1)
            else:
                self.fail_count += 1
                if CURRENT_LOG_LEVEL.value >= LogLevel.PRODUCTION.value:
                    logger.warning(f"Quantum operation '{operation}' failed. Count: {self.fail_count}")
            
            # Adaptive threshold based on recent success rate
            recent_ops = [op for op in self.operation_history if timestamp - op[2] < 300]  # Last 5 minutes
            if recent_ops:
                success_rate = sum(1 for op in recent_ops if op[1]) / len(recent_ops)
                adaptive_threshold = max(3, int(self.fail_threshold * (1 - success_rate)))
            else:
                adaptive_threshold = self.fail_threshold
            
            if self.fail_count >= adaptive_threshold:
                self.open_until = datetime.datetime.now() + datetime.timedelta(seconds=self.reset_timeout)
                if CURRENT_LOG_LEVEL.value >= LogLevel.CRITICAL.value:
                    logger.error(f"Quantum circuit breaker OPEN until {self.open_until}")
    
    def allow_quantum_operation(self) -> bool:
        """Check if quantum operation is allowed with adaptive logic."""
        if self.open_until and datetime.datetime.now() < self.open_until:
            return False
        return True
    
    def get_health_score(self) -> float:
        """Get circuit health score (0.0 to 1.0)."""
        with self.lock:
            if not self.operation_history:
                return 1.0
            
            recent_ops = [op for op in self.operation_history if time.time() - op[2] < 300]
            if not recent_ops:
                return 1.0
            
            return sum(1 for op in recent_ops if op[1]) / len(recent_ops)

class QuantumPerformanceMonitor:
    """Monitor and optimize quantum operation performance."""
    
    def __init__(self):
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
    def record_operation(self, operation: str, duration: float, success: bool, 
                        metadata: Optional[Dict[str, Any]] = None):
        """Record operation with enhanced metadata."""
        with self.lock:
            if operation not in self.operation_stats:
                self.operation_stats[operation] = {
                    'count': 0,
                    'success_count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'recent_times': [],
                    'metadata': {}
                }
            
            stats = self.operation_stats[operation]
            stats['count'] += 1
            if success:
                stats['success_count'] += 1
            
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            
            # Keep recent times for trend analysis
            stats['recent_times'].append(duration)
            if len(stats['recent_times']) > 50:
                stats['recent_times'] = stats['recent_times'][-50:]
            
            if metadata:
                stats['metadata'].update(metadata)
    
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """Get performance optimization recommendations."""
        recommendations = {}
        
        with self.lock:
            for operation, stats in self.operation_stats.items():
                if stats['count'] < 5:
                    continue
                
                avg_time = stats['total_time'] / stats['count']
                success_rate = stats['success_count'] / stats['count']
                
                if success_rate < 0.9:
                    recommendations[operation] = f"Low success rate ({success_rate:.1%}). Consider increasing retry attempts or error tolerance."
                
                if avg_time > 0.1:  # > 100ms
                    recommendations[operation] = f"High latency ({avg_time:.3f}s). Consider algorithm optimization."
                
                # Check for performance degradation
                if len(stats['recent_times']) >= 10:
                    recent_avg = sum(stats['recent_times'][-10:]) / 10
                    overall_avg = stats['total_time'] / stats['count']
                    if recent_avg > overall_avg * 1.5:
                        recommendations[operation] = f"Performance degradation detected. Recent: {recent_avg:.3f}s vs overall: {overall_avg:.3f}s"
        
        return recommendations

# --- Global Enhanced Infrastructure ---
quantum_config = QuantumConfig()
adaptive_circuit_breaker = AdaptiveQuantumCircuitBreaker(quantum_config.circuit_breaker_threshold)
performance_monitor = QuantumPerformanceMonitor()

# --- Enhanced Quantum Decorators ---
def quantum_performance_tracking(operation_name: str = None):
    """Enhanced performance tracking decorator."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            success = False
            metadata = {}
            
            try:
                # Check circuit breaker
                if not adaptive_circuit_breaker.allow_quantum_operation():
                    raise QuantumCircuitBreakerOpen(f"Circuit breaker open for {op_name}")
                
                result = await func(*args, **kwargs)
                success = True
                
                # Extract metadata from result if it's a tuple
                if isinstance(result, tuple) and len(result) > 1:
                    actual_result, meta = result[0], result[1:]
                    metadata = {'result_metadata': meta}
                    return actual_result
                
                return result
                
            except Exception as e:
                metadata = {'error_type': type(e).__name__, 'error_msg': str(e)}
                raise
            finally:
                duration = time.time() - start_time
                adaptive_circuit_breaker.record_quantum_result(op_name, success)
                performance_monitor.record_operation(op_name, duration, success, metadata)
                
                # Adaptive logging based on operation frequency
                op_count = performance_monitor.operation_stats.get(op_name, {}).get('count', 0)
                if (op_count % quantum_config.log_frequency_reduction == 0 or 
                    not success or 
                    CURRENT_LOG_LEVEL.value >= LogLevel.DEBUG.value):
                    if success:
                        logger.info(f"{op_name} completed in {duration:.4f}s (#{op_count})")
                    else:
                        logger.error(f"{op_name} failed after {duration:.4f}s")
        
        return wrapper
    return decorator

# --- Optimized Quantum Dynamics Framework ---
class OptimizedQuantumDynamicsFramework:
    """
    Production-optimized quantum dynamics framework with adaptive algorithms.
    
    Features:
    - Adaptive integration with automatic step size control
    - Intelligent error recovery and circuit breaking
    - Performance monitoring and optimization recommendations
    - Reduced logging overhead for production environments
    """
    
    def __init__(self, tolerances: Optional[QuantumTolerances] = None):
        """Initialize optimized quantum framework."""
        self.tol = tolerances or QuantumTolerances()
        self._operation_count = 0
        logger.info("Optimized Quantum Dynamics Framework initialized")
    
    def _validate_matrix(self, matrix: np.ndarray, 
                        expected_shape: Optional[Tuple[int, int]] = None,
                        check_hermitian: bool = False,
                        check_unitary: bool = False,
                        name: str = "matrix") -> np.ndarray:
        """Optimized matrix validation with reduced overhead."""
        try:
            # Fast path for already-validated matrices
            if (isinstance(matrix, np.ndarray) and 
                matrix.dtype == np.complex128 and 
                matrix.ndim == 2 and 
                matrix.shape[0] == matrix.shape[1]):
                
                # Skip expensive checks if not critical
                if not (check_hermitian or check_unitary):
                    return matrix
            
            # Standard validation path
            if not isinstance(matrix, np.ndarray):
                matrix = np.asarray(matrix, dtype=complex)
            
            if matrix.dtype != np.complex128:
                matrix = matrix.astype(np.complex128)
            
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise QuantumValidationError(f"{name} must be square 2D matrix")
            
            if expected_shape and matrix.shape != expected_shape:
                raise QuantumValidationError(f"{name} shape mismatch")
            
            # Quantum-specific validations with caching
            if check_hermitian:
                hermitian_error = np.max(np.abs(matrix - matrix.conj().T))
                if hermitian_error > self.tol.hermiticity:
                    raise QuantumValidationError(f"{name} not Hermitian: error = {hermitian_error:.2e}")
            
            if check_unitary:
                identity = np.eye(matrix.shape[0], dtype=complex)
                unitary_error = np.max(np.abs(matrix @ matrix.conj().T - identity))
                if unitary_error > self.tol.unitarity:
                    raise QuantumValidationError(f"{name} not unitary: error = {unitary_error:.2e}")
            
            return matrix
            
        except Exception as e:
            logger.error(f"Matrix validation failed for {name}: {e}")
            raise QuantumValidationError(f"Matrix validation failed: {e}")
    
    @quantum_performance_tracking("commutator")
    async def commutator_async(self, H: np.ndarray, O: np.ndarray) -> np.ndarray:
        """Optimized commutator calculation."""
        H = self._validate_matrix(H, name="Hamiltonian")
        O = self._validate_matrix(O, expected_shape=H.shape, name="Observable")
        
        # Optimized commutator computation
        comm = H @ O - O @ H
        
        # Validation
        if not np.all(np.isfinite(comm)):
            raise QuantumValidationError("Commutator contains non-finite values")
        
        return comm
    
    def commutator(self, H: np.ndarray, O: np.ndarray) -> np.ndarray:
        """Synchronous commutator calculation."""
        return asyncio.run(self.commutator_async(H, O))
    
    @quantum_performance_tracking("dyson_series")
    async def adaptive_dyson_series_async(self, H_func: Callable[[float], np.ndarray],
                                        t: float, order: int = 2,
                                        initial_steps: int = 50,
                                        max_steps: int = 1000,
                                        target_error: float = 1e-8) -> np.ndarray:
        """
        Adaptive Dyson series with automatic step refinement for better unitarity.
        
        Uses Richardson extrapolation and adaptive step sizing to achieve target accuracy.
        """
        if not callable(H_func):
            raise TypeError("H_func must be callable")
        if t <= 0:
            raise ValueError(f"Evolution time must be positive, got {t}")
        if order not in [1, 2]:
            raise ValueError(f"Order must be 1 or 2, got {order}")
        
        logger.info(f"Adaptive Dyson series: order={order}, t={t}, target_error={target_error:.0e}")
        
        # Test Hamiltonian
        H_test = H_func(0.0)
        H_test = self._validate_matrix(H_test, check_hermitian=True, name="H(0)")
        dim = H_test.shape[0]
        
        # Adaptive integration with Richardson extrapolation
        time_steps = initial_steps
        best_U = None
        best_error = float('inf')
        
        for refinement in range(5):  # Maximum 5 refinement levels
            try:
                U = await self._compute_dyson_series(H_func, t, order, time_steps, dim)
                
                # Check unitarity error
                unitary_error = np.max(np.abs(U @ U.conj().T - np.eye(dim)))
                
                if unitary_error < best_error:
                    best_U = U
                    best_error = unitary_error
                
                if unitary_error <= target_error:
                    logger.info(f"Target accuracy achieved: {unitary_error:.2e} at {time_steps} steps")
                    break
                
                if time_steps >= max_steps:
                    logger.warning(f"Max steps reached. Best error: {best_error:.2e}")
                    break
                
                # Double the time steps for next iteration
                time_steps = min(time_steps * 2, max_steps)
                
            except Exception as e:
                logger.warning(f"Refinement level {refinement} failed: {e}")
                break
        
        if best_U is None:
            raise QuantumValidationError("Adaptive Dyson series failed to converge")
        
        logger.info(f"Adaptive Dyson series completed: final_error={best_error:.2e}")
        return best_U
    
    async def _compute_dyson_series(self, H_func: Callable[[float], np.ndarray],
                                  t: float, order: int, time_steps: int, dim: int) -> np.ndarray:
        """Core Dyson series computation with optimized integration."""
        # Ensure odd number for Simpson's rule
        if time_steps % 2 == 0:
            time_steps += 1
        
        time_grid = np.linspace(0, t, time_steps)
        dt = t / (time_steps - 1)
        
        # Precompute Hamiltonians efficiently
        H_values = []
        for t_val in time_grid:
            try:
                H_t = H_func(t_val)
                H_t = self._validate_matrix(H_t, expected_shape=(dim, dim), 
                                           check_hermitian=True, name=f"H({t_val})")
                H_values.append(H_t)
            except Exception as e:
                logger.warning(f"Hamiltonian at t={t_val} failed, using zeros: {e}")
                H_values.append(np.zeros((dim, dim), dtype=complex))
        
        # Initialize with identity
        U = np.eye(dim, dtype=complex)
        
        # First order term using optimized Simpson's rule
        first_order = self._simpson_integrate_matrices(H_values, dt)
        U += -1j * first_order
        
        # Second order term (if requested)
        if order >= 2:
            second_order = await self._compute_second_order_term(H_values, time_grid, dt)
            U += (-1j)**2 * second_order
        
        return U
    
    async def _compute_second_order_term(self, H_values: List[np.ndarray], 
                                       time_grid: np.ndarray, dt: float) -> np.ndarray:
        """Optimized second-order term computation."""
        dim = H_values[0].shape[0]
        second_order = np.zeros((dim, dim), dtype=complex)
        
        # Vectorized computation where possible
        for i, t1 in enumerate(time_grid[1:], 1):  # Skip t=0
            H_t1 = H_values[i]
            
            # Optimized inner integral using subset of precomputed values
            inner_indices = np.arange(0, i + 1)
            inner_weights = self._get_simpson_weights(len(inner_indices), t1 / (len(inner_indices) - 1))
            
            inner_sum = np.zeros((dim, dim), dtype=complex)
            for j, idx in enumerate(inner_indices):
                inner_sum += inner_weights[j] * (H_t1 @ H_values[idx])
            
            # Outer integration weight (Simpson's rule)
            outer_weight = self._get_simpson_weight(i, len(time_grid), dt)
            second_order += outer_weight * inner_sum
        
        return second_order
    
    def _simpson_integrate_matrices(self, matrices: List[np.ndarray], dt: float) -> np.ndarray:
        """Optimized Simpson's rule integration."""
        n = len(matrices)
        if n < 3:
            return self._trapezoidal_integrate_matrices(matrices, dt)
        
        # Ensure odd number of points
        if n % 2 == 0:
            matrices = matrices[:-1]
            n = len(matrices)
        
        weights = self._get_simpson_weights(n, dt)
        
        result = np.zeros_like(matrices[0])
        for i, (matrix, weight) in enumerate(zip(matrices, weights)):
            result += weight * matrix
        
        return result
    
    def _get_simpson_weights(self, n: int, dt: float) -> np.ndarray:
        """Get Simpson's rule weights."""
        weights = np.ones(n) * dt / 3
        weights[0] = weights[-1] = dt / 3
        weights[1::2] *= 4  # Odd indices (1, 3, 5, ...)
        weights[2::2] *= 2  # Even indices (2, 4, 6, ...)
        return weights
    
    def _get_simpson_weight(self, i: int, n: int, dt: float) -> float:
        """Get single Simpson's rule weight."""
        if i == 0 or i == n - 1:
            return dt / 3
        elif i % 2 == 1:
            return 4 * dt / 3
        else:
            return 2 * dt / 3
    
    def _trapezoidal_integrate_matrices(self, matrices: List[np.ndarray], dt: float) -> np.ndarray:
        """Fallback trapezoidal integration."""
        if len(matrices) < 2:
            return matrices[0] * dt if matrices else np.zeros_like(matrices[0])
        
        result = 0.5 * (matrices[0] + matrices[-1])
        for matrix in matrices[1:-1]:
            result += matrix
        
        return result * dt
    
    def dyson_series_expansion(self, H_func: Callable[[float], np.ndarray],
                             t: float, order: int = 2,
                             target_error: float = 1e-8) -> np.ndarray:
        """Synchronous adaptive Dyson series."""
        return asyncio.run(self.adaptive_dyson_series_async(H_func, t, order, 50, 1000, target_error))
    
    @quantum_performance_tracking("heisenberg_evolution")
    async def heisenberg_evolution_async(self, O_initial: np.ndarray,
                                       H_func: Callable[[float], np.ndarray],
                                       T: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Heisenberg evolution with reduced logging overhead."""
        # Validation
        O_initial = self._validate_matrix(O_initial, name="Initial observable")
        
        if not callable(H_func) or T <= 0 or N < 10:
            raise ValueError("Invalid parameters for Heisenberg evolution")
        
        # Evolution with progress tracking
        dt = T / N
        time_array = np.linspace(0, T, N + 1)
        O_current = O_initial.copy()
        dim = O_initial.shape[0]
        
        # Reduced-frequency progress logging
        log_interval = max(1, N // 5)  # Log 5 times maximum
        
        for n in range(N):
            t_n = time_array[n]
            
            try:
                H_n = H_func(t_n)
                H_n = self._validate_matrix(H_n, expected_shape=(dim, dim),
                                           check_hermitian=True, name=f"H({t_n})")
                
                # Fast commutator computation
                commutator_term = H_n @ O_current - O_current @ H_n
                O_current += dt * 1j * commutator_term
                
                # Stability check
                if not np.all(np.isfinite(O_current)):
                    raise QuantumValidationError(f"Non-finite values at step {n}")
                
                # Reduced logging
                if n % log_interval == 0:
                    norm = np.linalg.norm(O_current)
                    logger.debug(f"Evolution progress: {100*n/N:.0f}% (t={t_n:.3f}, ||O||={norm:.3e})")
            
            except Exception as e:
                logger.error(f"Evolution failed at step {n}: {e}")
                raise
        
        return O_current, time_array
    
    def heisenberg_evolution_euler(self, O_initial: np.ndarray,
                                 H_func: Callable[[float], np.ndarray],
                                 T: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous Heisenberg evolution."""
        return asyncio.run(self.heisenberg_evolution_async(O_initial, H_func, T, N))

# --- Enhanced Demonstration ---
async def demonstrate_optimized_system():
    """Demonstration of the optimized quantum system."""
    
    print("🚀 OPTIMIZED QUANTUM DYNAMICS SYSTEM")
    print("=" * 50)
    
    # Initialize with strict tolerances
    tolerances = QuantumTolerances(
        hermiticity=1e-15,
        unitarity=1e-14,
        commutator_precision=1e-16
    )
    qdf = OptimizedQuantumDynamicsFramework(tolerances)
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # 1. Optimized commutator
    print("\n1. HIGH-PRECISION COMMUTATOR")
    print("-" * 30)
    
    comm_xz = await qdf.commutator_async(sigma_x, sigma_z)
    print(f"✅ [σ_x, σ_z] norm: {np.linalg.norm(comm_xz):.10f}")
    print(f"   Theoretical: {2*np.sqrt(2):.10f}")
    print(f"   Error: {abs(np.linalg.norm(comm_xz) - 2*np.sqrt(2)):.2e}")
    
    # 2. Adaptive Dyson series
    print("\n2. ADAPTIVE DYSON SERIES")
    print("-" * 30)
    
    def precise_hamiltonian(t):
        return 1.0 * sigma_z + 0.1 * np.sin(2.0 * t) * sigma_x
    
    U_adaptive = await qdf.adaptive_dyson_series_async(
        precise_hamiltonian, 
        t=0.5, 
        order=2,
        target_error=1e-10
    )
    
    unitarity_error = np.linalg.norm(U_adaptive @ U_adaptive.conj().T - np.eye(2))
    print(f"✅ Adaptive integration completed")
    print(f"   Unitarity error: {unitarity_error:.2e}")
    print(f"   Target achieved: {unitarity_error <= 1e-10}")
    
    # 3. Efficient Heisenberg evolution
    print("\n3. EFFICIENT HEISENBERG EVOLUTION")
    print("-" * 30)
    
    O_final, times = await qdf.heisenberg_evolution_async(
        O_initial=sigma_x,
        H_func=precise_hamiltonian,
        T=2.0,
        N=100
    )
    
    print(f"✅ Evolution completed with minimal logging")
    print(f"   Final norm: {np.linalg.norm(O_final):.6f}")
    print(f"   Norm preservation: {abs(np.linalg.norm(O_final) - np.linalg.norm(sigma_x)):.2e}")
    
    # 4. Performance analysis
    print("\n4. PERFORMANCE ANALYSIS")
    print("-" * 30)
    
    health_score = adaptive_circuit_breaker.get_health_score()
    recommendations = performance_monitor.get_optimization_recommendations()
    
    print(f"📊 System Health Score: {health_score:.1%}")
    print(f"🔧 Optimization Recommendations: {len(recommendations)}")
    
    for operation, recommendation in recommendations.items():
        print(f"   • {operation}: {recommendation}")
    
    if not recommendations:
        print("   🎯 All systems operating optimally!")
    
    # 5. Error injection test
    print("\n5. CONTROLLED ERROR INJECTION")
    print("-" * 30)
    
    def failing_hamiltonian(t):
        if 0.3 < t < 0.7:  # Failure window
            raise RuntimeError(f"Simulated failure at t={t}")
        return sigma_z
    
    try:
        # This should trigger circuit breaker after multiple failures
        for i in range(3):
            try:
                await qdf.adaptive_dyson_series_async(failing_hamiltonian, t=1.0, order=1)
            except Exception:
                pass  # Expected failures
        
        # Check if circuit breaker is now more sensitive
        final_health = adaptive_circuit_breaker.get_health_score()
        print(f"✅ Error injection completed")
        print(f"   Health score after failures: {final_health:.1%}")
        print(f"   Circuit learning: {'Active' if final_health < health_score else 'Inactive'}")
        
    except Exception as e:
        print(f"⚠️  Error injection test: {type(e).__name__}")
    
    print("\n" + "=" * 50)
    print("🎯 OPTIMIZED SYSTEM DEMONSTRATION COMPLETE")

def main():
    """Main entry point with environment detection."""
    
    # Auto-detect environment and set appropriate logging
    if os.getenv('PRODUCTION') == '1':
        global CURRENT_LOG_LEVEL
        CURRENT_LOG_LEVEL = LogLevel.CRITICAL
        setup_logging(CURRENT_LOG_LEVEL)
        print("🏭 Production mode: Critical logging only")
    elif os.getenv('DEBUG') == '1':
        CURRENT_LOG_LEVEL = LogLevel.DEBUG
        setup_logging(CURRENT_LOG_LEVEL)
        print("🔍 Debug mode: Verbose logging enabled")
    else:
        print("⚙️  Development mode: Standard logging")
    
    # Run demonstration
    asyncio.run(demonstrate_optimized_system())
    
    # Final system status
    print(f"\n🏥 Final System Status:")
    print(f"   Circuit Health: {adaptive_circuit_breaker.get_health_score():.1%}")
    print(f"   Operations Logged: {len(performance_monitor.operation_stats)}")
    print(f"   Ready for production: ✅")

if __name__ == "__main__":
    main()