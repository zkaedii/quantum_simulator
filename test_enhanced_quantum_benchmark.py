import pytest
import numpy as np
from enhanced_quantum_benchmark import EnhancedQuantumBenchmark, QuantumBenchmark

@pytest.fixture
def benchmark():
    return EnhancedQuantumBenchmark()

def test_compute_fixed_dyson_returns_unitary_matrix(benchmark):
    def hamiltonian(t):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        return sigma_z
    U = benchmark._compute_fixed_dyson(hamiltonian, t=1.0, steps=51)
    assert U.shape == (2, 2)
    # Check if U is close to unitary
    np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-10)

def test_compute_adaptive_dyson_returns_tuple(benchmark):
    def hamiltonian(t):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        return sigma_z
    U, steps, refinements = benchmark._compute_adaptive_dyson(hamiltonian, t=1.0, target_error=1e-8)
    assert isinstance(U, np.ndarray)
    assert isinstance(steps, int)
    assert isinstance(refinements, int)
    assert U.shape == (2, 2)

def test_simpson_integrate_matrices_basic(benchmark):
    matrices = [np.eye(2), np.eye(2), np.eye(2)]
    dt = 0.1
    result = benchmark._simpson_integrate_matrices(matrices, dt)
    assert result.shape == (2, 2)
    # Should be proportional to identity
    assert np.allclose(result, np.eye(2) * dt)

def test_random_hermitian_matrix_properties(benchmark):
    n = 4
    A = benchmark._random_hermitian_matrix(n)
    assert A.shape == (n, n)
    # Hermitian property: A == A.conj().T
    np.testing.assert_allclose(A, A.conj().T, atol=1e-12)

def test_standard_commutator(benchmark):
    A = np.array([[0, 1], [1, 0]], dtype=complex)
    B = np.array([[1, 0], [0, -1]], dtype=complex)
    comm = benchmark._standard_commutator(A, B)
    expected = A @ B - B @ A
    np.testing.assert_allclose(comm, expected)

def test_optimized_commutator(benchmark):
    A = np.array([[0, 1], [1, 0]], dtype=complex)
    B = np.array([[1, 0], [0, -1]], dtype=complex)
    comm = benchmark._optimized_commutator(A, B)
    expected = A @ B - B @ A
    np.testing.assert_allclose(comm, expected)

def test_cache_aware_commutator_small_matrix(benchmark):
    A = np.eye(4, dtype=complex)
    B = np.eye(4, dtype=complex)
    comm = benchmark._cache_aware_commutator(A, B)
    np.testing.assert_allclose(comm, np.zeros_like(A))

def test_benchmark_adaptive_vs_fixed_dyson(benchmark):
    result = benchmark.benchmark_adaptive_vs_fixed_dyson()
    assert isinstance(result, dict)
    assert 'adaptive_achieved_target' in result
    assert 'adaptive_error' in result

def test_benchmark_commutator_algorithms(benchmark):
    result = benchmark.benchmark_commutator_algorithms()
    assert isinstance(result, dict)
    assert result['status'] == 'completed'
    assert 'algorithms_tested' in result

def test_benchmark_error_recovery(benchmark):
    result = benchmark.benchmark_error_recovery()
    assert isinstance(result, dict)
    assert result['status'] == 'completed'
    assert 'scenarios_tested' in result

def test_generate_enhanced_report(benchmark):
    # Populate some results for report
    benchmark.results.append(
        QuantumBenchmark(
            operation="dyson_series",
            method="Adaptive",
            time_taken=0.01,
            error_achieved=1e-12,
            target_met=True,
            steps_used=51,
            convergence_rate=5100,
            status="✅ TARGET MET"
        )
    )
    report = benchmark.generate_enhanced_report()
    assert isinstance(report, str)
    assert "Enhanced quantum system validated" in report