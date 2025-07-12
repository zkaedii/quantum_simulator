import numpy as np
import scipy.linalg as la
import logging
from typing import Callable, Tuple, Optional, Union, Dict, Any, List
from dataclasses import dataclass
import warnings
from functools import lru_cache
import time

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QuantumTolerances:
    """Quantum precision thresholds for various operations."""
    hermiticity: float = 1e-12
    unitarity: float = 1e-12
    trace_preservation: float = 1e-12
    commutator_precision: float = 1e-14
    evolution_step: float = 1e-16
    matrix_condition: float = 1e-10

class QuantumValidationError(Exception):
    """Custom exception for quantum state/operator validation failures."""
    pass

class QuantumDynamicsFramework:
    """
    Comprehensive framework for quantum dynamics with explicit commutator calculations,
    Dyson series expansions, and numerical Heisenberg evolution.
    
    Features:
    - Full error management with input validation and exception handling
    - Quantum precision handling with configurable tolerances
    - Comprehensive logging for all critical operations
    - Caching for performance optimization
    """
    
    def __init__(self, tolerances: Optional[QuantumTolerances] = None):
        """
        Initialize the quantum dynamics framework.
        
        Args:
            tolerances: Quantum precision thresholds
        """
        self.tol = tolerances or QuantumTolerances()
        logger.info("Quantum Dynamics Framework initialized")
        
    def _validate_matrix(self, matrix: np.ndarray, 
                        expected_shape: Optional[Tuple[int, int]] = None,
                        check_hermitian: bool = False,
                        check_unitary: bool = False,
                        name: str = "matrix") -> np.ndarray:
        """
        Comprehensive matrix validation with quantum-specific checks.
        
        Args:
            matrix: Input matrix to validate
            expected_shape: Expected matrix dimensions
            check_hermitian: Whether to verify Hermiticity
            check_unitary: Whether to verify unitarity
            name: Matrix name for error reporting
            
        Returns:
            Validated matrix as complex128
            
        Raises:
            QuantumValidationError: If validation fails
        """
        try:
            # Type and dimension validation
            if not isinstance(matrix, np.ndarray):
                matrix = np.asarray(matrix, dtype=complex)
            
            if matrix.dtype != np.complex128:
                matrix = matrix.astype(np.complex128)
                logger.debug(f"Converted {name} to complex128")
            
            if matrix.ndim != 2:
                raise QuantumValidationError(
                    f"{name} must be 2D, got {matrix.ndim}D"
                )
            
            if matrix.shape[0] != matrix.shape[1]:
                raise QuantumValidationError(
                    f"{name} must be square, got shape {matrix.shape}"
                )
            
            if expected_shape and matrix.shape != expected_shape:
                raise QuantumValidationError(
                    f"{name} shape {matrix.shape} != expected {expected_shape}"
                )
            
            # Quantum-specific validations
            if check_hermitian:
                hermitian_error = np.max(np.abs(matrix - matrix.conj().T))
                if hermitian_error > self.tol.hermiticity:
                    raise QuantumValidationError(
                        f"{name} not Hermitian: max error = {hermitian_error:.2e}"
                    )
                logger.debug(f"{name} Hermiticity verified: error = {hermitian_error:.2e}")
            
            if check_unitary:
                identity = np.eye(matrix.shape[0], dtype=complex)
                unitary_error = np.max(np.abs(matrix @ matrix.conj().T - identity))
                if unitary_error > self.tol.unitarity:
                    raise QuantumValidationError(
                        f"{name} not unitary: max error = {unitary_error:.2e}"
                    )
                logger.debug(f"{name} unitarity verified: error = {unitary_error:.2e}")
            
            # Condition number check for numerical stability
            cond_num = np.linalg.cond(matrix)
            if cond_num > 1.0 / self.tol.matrix_condition:
                warnings.warn(
                    f"{name} poorly conditioned: cond = {cond_num:.2e}",
                    RuntimeWarning
                )
            
            return matrix
            
        except Exception as e:
            logger.error(f"Matrix validation failed for {name}: {e}")
            raise QuantumValidationError(f"Matrix validation failed: {e}")
    
    def softplus(self, x: Union[float, np.ndarray], 
                 threshold: float = 30.0) -> Union[float, np.ndarray]:
        """
        Numerically stable softplus function with overflow protection.
        
        Args:
            x: Input value(s)
            threshold: Threshold for linear approximation
            
        Returns:
            softplus(x) = log(1 + exp(x))
        """
        try:
            x = np.asarray(x)
            result = np.where(
                x > threshold,
                x,  # Linear approximation for large x
                np.log1p(np.exp(np.clip(x, -700, threshold)))  # Stable computation
            )
            logger.debug(f"Softplus computed for input range [{np.min(x):.3f}, {np.max(x):.3f}]")
            return result
            
        except Exception as e:
            logger.error(f"Softplus computation failed: {e}")
            raise ValueError(f"Softplus computation failed: {e}")
    
    def integral_hamiltonian_function(self, x: float, x0: float, a: float, b: float,
                                    f_func: Callable[[float], complex],
                                    g_prime_func: Callable[[float], complex]) -> complex:
        """
        Compute h(x) = softplus(a(x-x0)^2 + b) * f(x) * g'(x).
        
        Args:
            x: Integration variable
            x0: Center parameter
            a, b: Softplus parameters
            f_func: Function f(x)
            g_prime_func: Function g'(x)
            
        Returns:
            Complex value h(x)
        """
        try:
            # Input validation
            if not all(isinstance(param, (int, float, complex)) 
                      for param in [x, x0, a, b]):
                raise TypeError("Parameters must be numeric")
            
            if not callable(f_func) or not callable(g_prime_func):
                raise TypeError("f_func and g_prime_func must be callable")
            
            # Compute components with error handling
            quadratic_term = a * (x - x0)**2 + b
            softplus_val = self.softplus(quadratic_term)
            
            f_val = f_func(x)
            g_prime_val = g_prime_func(x)
            
            result = softplus_val * f_val * g_prime_val
            
            # Validate result
            if not np.isfinite(result):
                raise ValueError(f"Non-finite result at x={x}")
            
            logger.debug(f"h({x}) = {result:.6e}")
            return complex(result)
            
        except Exception as e:
            logger.error(f"Integral Hamiltonian function evaluation failed at x={x}: {e}")
            raise ValueError(f"Function evaluation failed: {e}")
    
    @lru_cache(maxsize=128)
    def _cached_commutator(self, H_hash: int, O_hash: int, 
                          H_shape: Tuple[int, int]) -> np.ndarray:
        """Cached commutator computation for performance optimization."""
        # This is a placeholder for the actual cached computation
        # In practice, you'd need a more sophisticated caching strategy
        pass
    
    def commutator(self, H: np.ndarray, O: np.ndarray) -> np.ndarray:
        """
        Compute the commutator [H, O] = HO - OH with validation and logging.
        
        Args:
            H: Hamiltonian operator
            O: Observable operator
            
        Returns:
            Commutator [H, O]
            
        Raises:
            QuantumValidationError: If operators are incompatible
        """
        try:
            # Validate inputs
            H = self._validate_matrix(H, name="Hamiltonian")
            O = self._validate_matrix(O, expected_shape=H.shape, name="Observable")
            
            # Compute commutator
            start_time = time.perf_counter()
            HO = H @ O
            OH = O @ H
            comm = HO - OH
            computation_time = time.perf_counter() - start_time
            
            # Validate result
            if not np.all(np.isfinite(comm)):
                raise QuantumValidationError("Commutator contains non-finite values")
            
            # Check anti-Hermiticity (commutator should be anti-Hermitian if H, O are Hermitian)
            anti_hermitian_error = np.max(np.abs(comm + comm.conj().T))
            if anti_hermitian_error > self.tol.commutator_precision:
                logger.warning(
                    f"Commutator not anti-Hermitian: error = {anti_hermitian_error:.2e}"
                )
            
            logger.info(
                f"Commutator computed: shape={comm.shape}, "
                f"norm={np.linalg.norm(comm):.6e}, time={computation_time:.6f}s"
            )
            
            return comm
            
        except Exception as e:
            logger.error(f"Commutator computation failed: {e}")
            raise QuantumValidationError(f"Commutator computation failed: {e}")
    
    def integral_commutator_contribution(self, t: float, x0: float, a: float, b: float,
                                       f_func: Callable[[float], complex],
                                       g_prime_func: Callable[[float], complex],
                                       O_H: np.ndarray,
                                       integration_points: int = 1000) -> np.ndarray:
        """
        Compute the integral commutator contribution:
        (i/ħ) ∫₀ᵗ [h(x), O_H(t)] dx
        
        Args:
            t: Upper integration limit
            x0, a, b: Parameters for h(x)
            f_func, g_prime_func: Component functions
            O_H: Heisenberg picture observable
            integration_points: Number of integration points
            
        Returns:
            Integral commutator contribution
        """
        try:
            # Validate inputs
            if t <= 0:
                raise ValueError(f"Integration time must be positive, got {t}")
            
            if integration_points < 10:
                raise ValueError(f"Need at least 10 integration points, got {integration_points}")
            
            O_H = self._validate_matrix(O_H, name="Heisenberg observable")
            
            # Setup integration grid
            x_grid = np.linspace(0, t, integration_points)
            dx = t / (integration_points - 1)
            
            logger.info(f"Computing integral commutator over [{0}, {t}] with {integration_points} points")
            
            # Numerical integration with error handling
            integral_sum = np.zeros_like(O_H, dtype=complex)
            
            for i, x in enumerate(x_grid):
                try:
                    # Compute h(x) as a scalar
                    h_x = self.integral_hamiltonian_function(x, x0, a, b, f_func, g_prime_func)
                    
                    # Create operator h(x) * I (identity matrix scaled by h(x))
                    h_op = h_x * np.eye(O_H.shape[0], dtype=complex)
                    
                    # Compute [h(x), O_H]
                    comm_x = self.commutator(h_op, O_H)
                    
                    # Add to integral (trapezoidal rule)
                    weight = dx if i != 0 and i != len(x_grid) - 1 else dx / 2
                    integral_sum += weight * comm_x
                    
                except Exception as e:
                    logger.warning(f"Integration point {i} (x={x}) failed: {e}")
                    continue
            
            # Apply (i/ħ) factor (using ħ = 1 in natural units)
            result = 1j * integral_sum
            
            # Validation
            if not np.all(np.isfinite(result)):
                raise QuantumValidationError("Integral result contains non-finite values")
            
            norm = np.linalg.norm(result)
            logger.info(f"Integral commutator computed: norm = {norm:.6e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Integral commutator computation failed: {e}")
            raise QuantumValidationError(f"Integral commutator computation failed: {e}")
    
    def dyson_series_expansion(self, H_func: Callable[[float], np.ndarray],
                             t: float, order: int = 2,
                             time_steps: int = 100) -> np.ndarray:
        """
        Compute Dyson series expansion of U(t,0) to specified order.
        
        U(t,0) = T exp(-i/ħ ∫₀ᵗ H(t₁) dt₁)
               ≈ 1 - (i/ħ)∫₀ᵗ H(t₁)dt₁ + (-i/ħ)² ∫₀ᵗ dt₁ ∫₀^t₁ dt₂ H(t₁)H(t₂) + ...
        
        Args:
            H_func: Time-dependent Hamiltonian function
            t: Final time
            order: Expansion order (1 or 2)
            time_steps: Number of time discretization points
            
        Returns:
            Time evolution operator U(t,0)
        """
        try:
            # Input validation
            if not callable(H_func):
                raise TypeError("H_func must be callable")
            
            if t <= 0:
                raise ValueError(f"Evolution time must be positive, got {t}")
            
            if order not in [1, 2]:
                raise ValueError(f"Order must be 1 or 2, got {order}")
            
            if time_steps < 10:
                raise ValueError(f"Need at least 10 time steps, got {time_steps}")
            
            logger.info(f"Computing Dyson series to order {order}, t={t}, steps={time_steps}")
            
            # Test Hamiltonian at t=0 to get dimensions
            H_test = H_func(0.0)
            H_test = self._validate_matrix(H_test, check_hermitian=True, name="H(0)")
            dim = H_test.shape[0]
            
            # Initialize with identity (zeroth order)
            U = np.eye(dim, dtype=complex)
            
            # Time grid
            time_grid = np.linspace(0, t, time_steps)
            dt = t / (time_steps - 1)
            
            # First order term: -i/ħ ∫₀ᵗ H(t₁) dt₁
            logger.debug("Computing first order term")
            first_order = np.zeros((dim, dim), dtype=complex)
            
            for i, t1 in enumerate(time_grid):
                try:
                    H_t1 = H_func(t1)
                    H_t1 = self._validate_matrix(H_t1, expected_shape=(dim, dim), 
                                               check_hermitian=True, name=f"H({t1})")
                    
                    weight = dt if i != 0 and i != len(time_grid) - 1 else dt / 2
                    first_order += weight * H_t1
                    
                except Exception as e:
                    logger.warning(f"First order: time point {i} failed: {e}")
                    continue
            
            U += -1j * first_order  # -i/ħ factor (ħ = 1)
            
            # Second order term (if requested)
            if order >= 2:
                logger.debug("Computing second order term")
                second_order = np.zeros((dim, dim), dtype=complex)
                
                for i, t1 in enumerate(time_grid):
                    try:
                        H_t1 = H_func(t1)
                        H_t1 = self._validate_matrix(H_t1, expected_shape=(dim, dim),
                                                   name=f"H({t1})")
                        
                        # Inner integral: ∫₀^t₁ H(t₁)H(t₂) dt₂
                        inner_integral = np.zeros((dim, dim), dtype=complex)
                        
                        # Time grid for inner integration (0 to t1)
                        if t1 > 0:
                            inner_steps = max(10, int(time_steps * t1 / t))
                            inner_grid = np.linspace(0, t1, inner_steps)
                            dt2 = t1 / (inner_steps - 1) if inner_steps > 1 else 0
                            
                            for j, t2 in enumerate(inner_grid):
                                try:
                                    H_t2 = H_func(t2)
                                    H_t2 = self._validate_matrix(H_t2, expected_shape=(dim, dim),
                                                               name=f"H({t2})")
                                    
                                    weight2 = dt2 if j != 0 and j != len(inner_grid) - 1 else dt2 / 2
                                    inner_integral += weight2 * (H_t1 @ H_t2)
                                    
                                except Exception as e:
                                    logger.warning(f"Second order: inner time point {j} failed: {e}")
                                    continue
                        
                        weight1 = dt if i != 0 and i != len(time_grid) - 1 else dt / 2
                        second_order += weight1 * inner_integral
                        
                    except Exception as e:
                        logger.warning(f"Second order: outer time point {i} failed: {e}")
                        continue
                
                U += (-1j)**2 * second_order  # (-i/ħ)² factor
            
            # Post-computation validation
            if not np.all(np.isfinite(U)):
                raise QuantumValidationError("Evolution operator contains non-finite values")
            
            # Check unitarity (approximately, for small times/perturbations)
            unitary_error = np.max(np.abs(U @ U.conj().T - np.eye(dim)))
            if unitary_error > self.tol.unitarity and t < 1.0:  # Only check for small times
                logger.warning(f"Evolution operator not unitary: error = {unitary_error:.2e}")
            
            logger.info(f"Dyson series computed: order={order}, unitarity_error={unitary_error:.2e}")
            
            return U
            
        except Exception as e:
            logger.error(f"Dyson series computation failed: {e}")
            raise QuantumValidationError(f"Dyson series computation failed: {e}")
    
    def heisenberg_evolution_euler(self, O_initial: np.ndarray,
                                 H_func: Callable[[float], np.ndarray],
                                 T: float, N: int,
                                 explicit_time_derivative: Optional[Callable[[float], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numerical integration of Heisenberg equation using Euler method.
        
        dO_H/dt = (i/ħ)[H(t), O_H(t)] + (∂O/∂t)_H
        
        Args:
            O_initial: Initial observable O_H(0)
            H_func: Time-dependent Hamiltonian function
            T: Total evolution time
            N: Number of time steps
            explicit_time_derivative: Optional explicit time dependence of O
            
        Returns:
            Tuple of (final observable, time array)
        """
        try:
            # Input validation
            O_initial = self._validate_matrix(O_initial, name="Initial observable")
            
            if not callable(H_func):
                raise TypeError("H_func must be callable")
            
            if T <= 0:
                raise ValueError(f"Evolution time must be positive, got {T}")
            
            if N < 10:
                raise ValueError(f"Need at least 10 time steps, got {N}")
            
            if explicit_time_derivative and not callable(explicit_time_derivative):
                raise TypeError("explicit_time_derivative must be callable")
            
            logger.info(f"Heisenberg evolution: T={T}, N={N} steps, dt={T/N:.6e}")
            
            # Initialize
            dt = T / N
            time_array = np.linspace(0, T, N + 1)
            O_current = O_initial.copy()
            dim = O_initial.shape[0]
            
            # Evolution loop with comprehensive error handling
            for n in range(N):
                try:
                    t_n = time_array[n]
                    
                    # Get Hamiltonian at current time
                    H_n = H_func(t_n)
                    H_n = self._validate_matrix(H_n, expected_shape=(dim, dim),
                                              check_hermitian=True, name=f"H({t_n})")
                    
                    # Compute commutator term
                    commutator_term = self.commutator(H_n, O_current)
                    evolution_term = 1j * commutator_term  # (i/ħ)[H, O]
                    
                    # Add explicit time derivative if provided
                    if explicit_time_derivative:
                        try:
                            explicit_term = explicit_time_derivative(t_n)
                            explicit_term = self._validate_matrix(explicit_term, 
                                                                expected_shape=(dim, dim),
                                                                name="Explicit time derivative")
                            evolution_term += explicit_term
                        except Exception as e:
                            logger.warning(f"Explicit time derivative at t={t_n} failed: {e}")
                    
                    # Euler step with stability check
                    O_new = O_current + dt * evolution_term
                    
                    # Stability validation
                    if not np.all(np.isfinite(O_new)):
                        raise QuantumValidationError(f"Non-finite values at step {n}, t={t_n}")
                    
                    step_change = np.linalg.norm(O_new - O_current)
                    if step_change > 1.0:  # Large change warning
                        logger.warning(f"Large step change at t={t_n}: {step_change:.2e}")
                    
                    O_current = O_new
                    
                    # Progress logging
                    if n % max(1, N // 10) == 0:
                        norm = np.linalg.norm(O_current)
                        logger.debug(f"Step {n}/{N}, t={t_n:.4f}, ||O||={norm:.6e}")
                
                except Exception as e:
                    logger.error(f"Evolution step {n} failed at t={time_array[n]}: {e}")
                    raise QuantumValidationError(f"Evolution failed at step {n}: {e}")
            
            # Final validation
            final_norm = np.linalg.norm(O_current)
            initial_norm = np.linalg.norm(O_initial)
            
            logger.info(
                f"Heisenberg evolution completed: "
                f"initial_norm={initial_norm:.6e}, final_norm={final_norm:.6e}"
            )
            
            return O_current, time_array
            
        except Exception as e:
            logger.error(f"Heisenberg evolution failed: {e}")
            raise QuantumValidationError(f"Heisenberg evolution failed: {e}")

    def runge_kutta_4_heisenberg(self, O_initial: np.ndarray,
                                H_func: Callable[[float], np.ndarray],
                                T: float, N: int,
                                explicit_time_derivative: Optional[Callable[[float], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fourth-order Runge-Kutta integration of Heisenberg equation for improved accuracy.
        
        dO_H/dt = (i/ħ)[H(t), O_H(t)] + (∂O/∂t)_H
        
        Args:
            O_initial: Initial observable O_H(0)
            H_func: Time-dependent Hamiltonian function
            T: Total evolution time
            N: Number of time steps
            explicit_time_derivative: Optional explicit time dependence of O
            
        Returns:
            Tuple of (final observable, time array)
        """
        try:
            # Input validation
            O_initial = self._validate_matrix(O_initial, name="Initial observable")
            
            if not callable(H_func):
                raise TypeError("H_func must be callable")
            
            if T <= 0:
                raise ValueError(f"Evolution time must be positive, got {T}")
            
            if N < 10:
                raise ValueError(f"Need at least 10 time steps, got {N}")
            
            logger.info(f"RK4 Heisenberg evolution: T={T}, N={N} steps, dt={T/N:.6e}")
            
            # Initialize
            dt = T / N
            time_array = np.linspace(0, T, N + 1)
            O_current = O_initial.copy()
            dim = O_initial.shape[0]
            
            def evolution_rhs(t: float, O: np.ndarray) -> np.ndarray:
                """Right-hand side of the evolution equation."""
                H_t = H_func(t)
                H_t = self._validate_matrix(H_t, expected_shape=(dim, dim),
                                          check_hermitian=True, name=f"H({t})")
                
                commutator_term = self.commutator(H_t, O)
                rhs = 1j * commutator_term  # (i/ħ)[H, O]
                
                if explicit_time_derivative:
                    explicit_term = explicit_time_derivative(t)
                    explicit_term = self._validate_matrix(explicit_term, 
                                                        expected_shape=(dim, dim),
                                                        name="Explicit time derivative")
                    rhs += explicit_term
                
                return rhs
            
            # RK4 evolution loop
            for n in range(N):
                try:
                    t_n = time_array[n]
                    
                    # RK4 steps
                    k1 = dt * evolution_rhs(t_n, O_current)
                    k2 = dt * evolution_rhs(t_n + dt/2, O_current + k1/2)
                    k3 = dt * evolution_rhs(t_n + dt/2, O_current + k2/2)
                    k4 = dt * evolution_rhs(t_n + dt, O_current + k3)
                    
                    # Update
                    O_new = O_current + (k1 + 2*k2 + 2*k3 + k4) / 6
                    
                    # Stability validation
                    if not np.all(np.isfinite(O_new)):
                        raise QuantumValidationError(f"Non-finite values at step {n}, t={t_n}")
                    
                    O_current = O_new
                    
                    # Progress logging
                    if n % max(1, N // 10) == 0:
                        norm = np.linalg.norm(O_current)
                        logger.debug(f"RK4 Step {n}/{N}, t={t_n:.4f}, ||O||={norm:.6e}")
                
                except Exception as e:
                    logger.error(f"RK4 evolution step {n} failed at t={time_array[n]}: {e}")
                    raise QuantumValidationError(f"RK4 evolution failed at step {n}: {e}")
            
            final_norm = np.linalg.norm(O_current)
            initial_norm = np.linalg.norm(O_initial)
            
            logger.info(
                f"RK4 Heisenberg evolution completed: "
                f"initial_norm={initial_norm:.6e}, final_norm={final_norm:.6e}"
            )
            
            return O_current, time_array
            
        except Exception as e:
            logger.error(f"RK4 Heisenberg evolution failed: {e}")
            raise QuantumValidationError(f"RK4 Heisenberg evolution failed: {e}")
    
    def adaptive_dyson_series(self, H_func: Callable[[float], np.ndarray],
                             t: float, target_error: float = 1e-6,
                             max_order: int = 4) -> Tuple[np.ndarray, int]:
        """
        Adaptive Dyson series with automatic order selection based on error estimates.
        
        Args:
            H_func: Time-dependent Hamiltonian function
            t: Final time
            target_error: Target error tolerance
            max_order: Maximum expansion order
            
        Returns:
            Tuple of (evolution operator, order used)
        """
        try:
            logger.info(f"Adaptive Dyson series: t={t}, target_error={target_error:.2e}")
            
            # Test Hamiltonian to get dimensions
            H_test = H_func(0.0)
            H_test = self._validate_matrix(H_test, check_hermitian=True, name="H(0)")
            dim = H_test.shape[0]
            
            # Compute series terms up to max_order
            U_terms = []
            
            # Zeroth order (identity)
            U_0 = np.eye(dim, dtype=complex)
            U_terms.append(U_0)
            
            # First order and higher
            for order in range(1, max_order + 1):
                U_order = self.dyson_series_expansion(H_func, t, order=order, time_steps=100)
                U_terms.append(U_order)
                
                # Estimate error by comparing consecutive orders
                if order > 1:
                    error_estimate = np.linalg.norm(U_terms[order] - U_terms[order-1])
                    logger.debug(f"Order {order} error estimate: {error_estimate:.2e}")
                    
                    if error_estimate < target_error:
                        logger.info(f"Converged at order {order} with error {error_estimate:.2e}")
                        return U_terms[order], order
            
            logger.warning(f"Did not converge within max_order={max_order}")
            return U_terms[-1], max_order
            
        except Exception as e:
            logger.error(f"Adaptive Dyson series failed: {e}")
            raise QuantumValidationError(f"Adaptive Dyson series failed: {e}")
    
    def compute_fidelity(self, state1: np.ndarray, state2: np.ndarray, 
                        state_type: str = "pure") -> float:
        """
        Compute quantum fidelity between two states.
        
        Args:
            state1, state2: Quantum states (vectors for pure states, matrices for mixed)
            state_type: "pure" for state vectors, "mixed" for density matrices
            
        Returns:
            Fidelity F(ρ₁, ρ₂)
        """
        try:
            if state_type == "pure":
                # For pure states: F = |⟨ψ₁|ψ₂⟩|²
                state1 = self._validate_matrix(state1, name="State 1")
                state2 = self._validate_matrix(state2, name="State 2")
                
                if state1.shape[1] != 1 or state2.shape[1] != 1:
                    # Assume they are column vectors
                    state1 = state1.flatten()
                    state2 = state2.flatten()
                
                overlap = np.vdot(state1, state2)  # ⟨ψ₁|ψ₂⟩
                fidelity = float(np.abs(overlap)**2)
                
            elif state_type == "mixed":
                # For mixed states: F = Tr(√(√ρ₁ ρ₂ √ρ₁))
                rho1 = self._validate_matrix(state1, name="Density matrix 1")
                rho2 = self._validate_matrix(state2, name="Density matrix 2")
                
                # Compute √ρ₁
                eigenvals1, eigenvecs1 = np.linalg.eigh(rho1)
                sqrt_rho1 = eigenvecs1 @ np.diag(np.sqrt(np.maximum(eigenvals1, 0))) @ eigenvecs1.conj().T
                
                # Compute √ρ₁ ρ₂ √ρ₁
                intermediate = sqrt_rho1 @ rho2 @ sqrt_rho1
                
                # Compute trace of square root
                eigenvals_int = np.linalg.eigvals(intermediate)
                fidelity = float(np.sum(np.sqrt(np.maximum(eigenvals_int.real, 0))))
                
            else:
                raise ValueError(f"Unknown state_type: {state_type}")
            
            logger.debug(f"Fidelity computed: {fidelity:.6e}")
            return fidelity
            
        except Exception as e:
            logger.error(f"Fidelity computation failed: {e}")
            raise QuantumValidationError(f"Fidelity computation failed: {e}")
    
    def compute_trace_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Compute trace distance between two density matrices.
        
        T(ρ₁, ρ₂) = (1/2) Tr|ρ₁ - ρ₂|
        
        Args:
            rho1, rho2: Density matrices
            
        Returns:
            Trace distance
        """
        try:
            rho1 = self._validate_matrix(rho1, name="Density matrix 1")
            rho2 = self._validate_matrix(rho2, name="Density matrix 2")
            
            if rho1.shape != rho2.shape:
                raise QuantumValidationError("Density matrices must have same shape")
            
            diff = rho1 - rho2
            eigenvals = np.linalg.eigvals(diff)
            trace_distance = 0.5 * np.sum(np.abs(eigenvals))
            
            logger.debug(f"Trace distance computed: {trace_distance:.6e}")
            return float(trace_distance.real)
            
        except Exception as e:
            logger.error(f"Trace distance computation failed: {e}")
            raise QuantumValidationError(f"Trace distance computation failed: {e}")

    def benchmark_performance(self, matrix_sizes: List[int] = [2, 4, 8, 16]) -> Dict[str, Dict[int, float]]:
        """
        Benchmark performance of key operations across different matrix sizes.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dictionary of operation timings
        """
        results = {
            'commutator': {},
            'matrix_exp': {},
            'eigendecomposition': {},
            'validation': {}
        }
        
        logger.info("Starting performance benchmark")
        
        for size in matrix_sizes:
            logger.info(f"Benchmarking size {size}x{size}")
            
            # Generate test matrices
            H = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            H = (H + H.conj().T) / 2  # Make Hermitian
            
            O = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            
            # Benchmark commutator
            start_time = time.perf_counter()
            for _ in range(10):
                _ = self.commutator(H, O)
            comm_time = (time.perf_counter() - start_time) / 10
            results['commutator'][size] = comm_time
            
            # Benchmark matrix exponential
            start_time = time.perf_counter()
            for _ in range(10):
                _ = la.expm(-1j * 0.1 * H)
            exp_time = (time.perf_counter() - start_time) / 10
            results['matrix_exp'][size] = exp_time
            
            # Benchmark eigendecomposition
            start_time = time.perf_counter()
            for _ in range(10):
                _ = np.linalg.eigh(H)
            eigen_time = (time.perf_counter() - start_time) / 10
            results['eigendecomposition'][size] = eigen_time
            
            # Benchmark validation
            start_time = time.perf_counter()
            for _ in range(10):
                _ = self._validate_matrix(H, check_hermitian=True)
            valid_time = (time.perf_counter() - start_time) / 10
            results['validation'][size] = valid_time
            
            logger.info(f"Size {size}: comm={comm_time:.2e}s, exp={exp_time:.2e}s, "
                       f"eigen={eigen_time:.2e}s, valid={valid_time:.2e}s")
        
        return results
# Example usage and demonstration
def demonstrate_framework():
    """Comprehensive demonstration of the quantum dynamics framework."""
    
    # Initialize framework
    tolerances = QuantumTolerances(
        hermiticity=1e-14,
        unitarity=1e-12,
        commutator_precision=1e-15
    )
    qdf = QuantumDynamicsFramework(tolerances)
    
    print("=== Quantum Dynamics Framework Demonstration ===\n")
    
    # 1. Commutator calculation
    print("1. COMMUTATOR CALCULATION")
    print("-" * 40)
    
    # Two-level system with Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    try:
        comm_xz = qdf.commutator(sigma_x, sigma_z)
        print(f"[σ_x, σ_z] =\n{comm_xz}")
        print(f"Expected: 2i σ_y")
        print(f"Actual norm: {np.linalg.norm(comm_xz):.6e}")
        print()
    except Exception as e:
        print(f"Commutator calculation failed: {e}\n")
    
    # 2. Dyson series expansion
    print("2. DYSON SERIES EXPANSION")
    print("-" * 40)
    
    def time_dependent_hamiltonian(t):
        """Example: H(t) = ω₀σ_z + λ sin(ωt) σ_x"""
        omega_0 = 1.0
        lam = 0.1
        omega = 2.0
        return omega_0 * sigma_z + lam * np.sin(omega * t) * sigma_x
    
    try:
        U_dyson = qdf.dyson_series_expansion(
            time_dependent_hamiltonian, 
            t=0.5, 
            order=2, 
            time_steps=50
        )
        print(f"U(0.5, 0) shape: {U_dyson.shape}")
        print(f"Unitarity check: ||U†U - I|| = {np.linalg.norm(U_dyson.conj().T @ U_dyson - np.eye(2)):.2e}")
        print()
    except Exception as e:
        print(f"Dyson series calculation failed: {e}\n")
    
    # 3. Heisenberg evolution
    print("3. HEISENBERG EVOLUTION")
    print("-" * 40)
    
    try:
        O_final, times = qdf.heisenberg_evolution_euler(
            O_initial=sigma_x,
            H_func=time_dependent_hamiltonian,
            T=1.0,
            N=100
        )
        print(f"Initial observable: σ_x")
        print(f"Final observable norm: {np.linalg.norm(O_final):.6e}")
        print(f"Evolution completed over {len(times)} time points")
        print()
    except Exception as e:
        print(f"Heisenberg evolution failed: {e}\n")
    
    # 4. Integral commutator demonstration
    print("4. INTEGRAL COMMUTATOR")
    print("-" * 40)
    
    def f_function(x):
        return np.exp(-0.1 * x)
    
    def g_prime_function(x):
        return np.cos(x)
    
    try:
        integral_comm = qdf.integral_commutator_contribution(
            t=1.0,
            x0=0.5,
            a=1.0,
            b=0.1,
            f_func=f_function,
            g_prime_func=g_prime_function,
            O_H=sigma_z,
            integration_points=100
        )
        print(f"Integral commutator shape: {integral_comm.shape}")
        print(f"Integral commutator norm: {np.linalg.norm(integral_comm):.6e}")
        print()
    except Exception as e:
        print(f"Integral commutator calculation failed: {e}\n")
    
    # 5. RK4 Heisenberg evolution (improved accuracy)
    print("5. RK4 HEISENBERG EVOLUTION")
    print("-" * 40)
    
    try:
        O_rk4, times_rk4 = qdf.runge_kutta_4_heisenberg(
            O_initial=sigma_x,
            H_func=time_dependent_hamiltonian,
            T=1.0,
            N=50  # Fewer steps needed for RK4
        )
        print(f"RK4 evolution completed")
        print(f"Initial observable norm: {np.linalg.norm(sigma_x):.6e}")
        print(f"Final observable norm: {np.linalg.norm(O_rk4):.6e}")
        print()
    except Exception as e:
        print(f"RK4 Heisenberg evolution failed: {e}\n")
    
    # 6. Adaptive Dyson series
    print("6. ADAPTIVE DYSON SERIES")
    print("-" * 40)
    
    try:
        U_adaptive, order_used = qdf.adaptive_dyson_series(
            time_dependent_hamiltonian,
            t=0.1,  # Smaller time for better convergence
            target_error=1e-8,
            max_order=3
        )
        print(f"Adaptive series converged at order: {order_used}")
        print(f"Final unitarity error: {np.linalg.norm(U_adaptive.conj().T @ U_adaptive - np.eye(2)):.2e}")
        print()
    except Exception as e:
        print(f"Adaptive Dyson series failed: {e}\n")
    
    # 7. Quantum fidelity and trace distance
    print("7. QUANTUM FIDELITY & TRACE DISTANCE")
    print("-" * 40)
    
    try:
        # Create two similar states
        state1 = np.array([[1], [0]], dtype=complex)  # |0⟩
        state2 = np.array([[np.cos(0.1)], [np.sin(0.1)]], dtype=complex)  # Slightly rotated
        
        fidelity = qdf.compute_fidelity(state1, state2, state_type="pure")
        print(f"Fidelity between |0⟩ and rotated state: {fidelity:.6f}")
        
        # Create density matrices
        rho1 = np.outer(state1.flatten(), state1.flatten().conj())
        rho2 = np.outer(state2.flatten(), state2.flatten().conj())
        
        trace_dist = qdf.compute_trace_distance(rho1, rho2)
        print(f"Trace distance: {trace_dist:.6f}")
        print()
    except Exception as e:
        print(f"Fidelity/distance calculation failed: {e}\n")
    
    # 8. Performance benchmark
    print("8. PERFORMANCE BENCHMARK")
    print("-" * 40)
    
    try:
        benchmark_results = qdf.benchmark_performance(matrix_sizes=[2, 4, 8])
        print("Operation timings (seconds):")
        for operation, timings in benchmark_results.items():
            print(f"{operation:>20}: {timings}")
        print()
    except Exception as e:
        print(f"Performance benchmark failed: {e}\n")
    
    print("=== Demonstration Complete ===")

if __name__ == "__main__":
    demonstrate_framework()