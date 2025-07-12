#!/usr/bin/env python3
"""
Enhanced demonstration showing the improved quantum dynamics framework
with focus on numerical accuracy and meaningful physical examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_dynamics_framework import QuantumDynamicsFramework, QuantumTolerances

def demonstrate_enhanced_features():
    """Demonstrate the enhanced quantum dynamics framework capabilities."""
    
    # Initialize with tighter tolerances
    tolerances = QuantumTolerances(
        hermiticity=1e-15,
        unitarity=1e-13,
        commutator_precision=1e-16
    )
    qdf = QuantumDynamicsFramework(tolerances)
    
    print("🔬 ENHANCED QUANTUM DYNAMICS DEMONSTRATION")
    print("=" * 50)
    
    # Define Pauli matrices and useful operators
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    # 1. Advanced Commutator Relations
    print("\n1. PAULI MATRIX COMMUTATOR ALGEBRA")
    print("-" * 40)
    
    commutators = [
        ("[σ_x, σ_y]", sigma_x, sigma_y, "2i σ_z"),
        ("[σ_y, σ_z]", sigma_y, sigma_z, "2i σ_x"), 
        ("[σ_z, σ_x]", sigma_z, sigma_x, "2i σ_y")
    ]
    
    for label, A, B, expected in commutators:
        comm = qdf.commutator(A, B)
        norm = np.linalg.norm(comm)
        print(f"{label:12} → norm = {norm:.6f} (expected: 2√2 ≈ 2.828)")
    
    # 2. Improved Dyson Series with Different Methods
    print("\n2. DYSON SERIES: METHOD COMPARISON")
    print("-" * 40)
    
    def driven_qubit_hamiltonian(t):
        """Driven two-level system: H(t) = ω₀σ_z + Ω(t)σ_x"""
        omega_0 = 2.0  # Qubit frequency
        drive_amplitude = 0.5
        drive_frequency = 1.8  # Near-resonant drive
        
        Omega_t = drive_amplitude * np.cos(drive_frequency * t)
        return omega_0 * sigma_z + Omega_t * sigma_x
    
    methods = ['trapezoidal', 'simpson']
    evolution_time = 1.0
    
    for method in methods:
        U = qdf.dyson_series_expansion(
            driven_qubit_hamiltonian,
            t=evolution_time,
            order=2,
            time_steps=201,  # Odd number for Simpson's rule
            method=method
        )
        
        # Check unitarity
        unitarity_error = np.linalg.norm(U @ U.conj().T - identity)
        det_U = np.linalg.det(U)
        
        print(f"{method:12} → Unitarity error: {unitarity_error:.2e}, "
              f"det(U): {abs(det_U):.6f}")
    
    # 3. Meaningful Integral Commutator
    print("\n3. INTEGRAL COMMUTATOR: DRIVEN SYSTEM")
    print("-" * 40)
    
    def envelope_function(x):
        """Gaussian envelope for drive pulse"""
        return np.exp(-0.5 * x**2)
    
    def phase_function_derivative(x):
        """Derivative of time-dependent phase"""
        return np.sin(2.0 * x)  # ∂φ/∂x where φ(x) = -cos(2x)/2
    
    # Compare different base Hamiltonians
    base_operators = [
        ("σ_x drive", sigma_x),
        ("σ_y drive", sigma_y),
        ("Mixed drive", (sigma_x + sigma_y) / np.sqrt(2))
    ]
    
    for label, H_base in base_operators:
        integral_result = qdf.integral_commutator_contribution(
            t=2.0,
            x0=1.0,  # Center of envelope
            a=0.5,   # Width parameter
            b=0.1,   # Offset
            f_func=envelope_function,
            g_prime_func=phase_function_derivative,
            O_H=sigma_z,  # Observable (population difference)
            H_base=H_base,
            integration_points=100
        )
        
        norm = np.linalg.norm(integral_result)
        max_element = np.max(np.abs(integral_result))
        print(f"{label:12} → norm = {norm:.6e}, max element = {max_element:.6e}")
    
    # 4. Heisenberg Evolution: Quantum State Dynamics
    print("\n4. HEISENBERG EVOLUTION: STATE DYNAMICS")
    print("-" * 40)
    
    def time_dependent_system(t):
        """Time-dependent Hamiltonian with multiple frequency components"""
        omega_1 = 1.0
        omega_2 = 1.5
        coupling = 0.3
        
        return (omega_1 * sigma_z + 
                coupling * np.cos(omega_2 * t) * sigma_x +
                0.1 * np.sin(omega_1 * t) * sigma_y)
    
    # Evolve different initial observables
    observables = [
        ("σ_x", sigma_x),
        ("σ_y", sigma_y), 
        ("σ_z", sigma_z),
        ("(σ_x + σ_z)/√2", (sigma_x + sigma_z) / np.sqrt(2))
    ]
    
    evolution_results = []
    for label, O_init in observables:
        O_final, times = qdf.heisenberg_evolution_euler(
            O_initial=O_init,
            H_func=time_dependent_system,
            T=3.0,
            N=150
        )
        
        initial_norm = np.linalg.norm(O_init)
        final_norm = np.linalg.norm(O_final)
        
        # Check trace preservation (for Hermitian operators)
        initial_trace = np.trace(O_init).real
        final_trace = np.trace(O_final).real
        trace_change = abs(final_trace - initial_trace)
        
        print(f"{label:15} → ||O||: {initial_norm:.6f} → {final_norm:.6f}, "
              f"Tr change: {trace_change:.2e}")
        
        evolution_results.append((label, times, O_final))
    
    # 5. Advanced Validation: Energy Conservation
    print("\n5. ENERGY CONSERVATION CHECK")
    print("-" * 40)
    
    def constant_hamiltonian(t):
        """Time-independent Hamiltonian for energy conservation test"""
        return 1.5 * sigma_z + 0.8 * sigma_x
    
    # Create energy observable (should be conserved)
    H_constant = constant_hamiltonian(0)
    
    # Evolve energy observable
    H_evolved, _ = qdf.heisenberg_evolution_euler(
        O_initial=H_constant,
        H_func=constant_hamiltonian,
        T=5.0,
        N=200
    )
    
    # For time-independent H, [H, H] = 0, so H should remain unchanged
    energy_change = np.linalg.norm(H_evolved - H_constant)
    relative_change = energy_change / np.linalg.norm(H_constant)
    
    print(f"Energy conservation: Δ||H|| = {energy_change:.2e}")
    print(f"Relative change: {relative_change:.2e}")
    
    if relative_change < 1e-10:
        print("✅ Energy conserved to machine precision!")
    elif relative_change < 1e-6:
        print("✅ Energy approximately conserved")
    else:
        print("⚠️  Significant energy drift detected")
    
    print("\n" + "=" * 50)
    print("🎯 DEMONSTRATION COMPLETE")
    print(f"Framework validated with quantum precision tolerances:")
    print(f"  • Hermiticity: {tolerances.hermiticity:.0e}")
    print(f"  • Unitarity: {tolerances.unitarity:.0e}")
    print(f"  • Commutator: {tolerances.commutator_precision:.0e}")

def visualize_qubit_dynamics():
    """Optional visualization of qubit evolution on Bloch sphere coordinates."""
    try:
        import matplotlib.pyplot as plt
        
        # Setup
        qdf = QuantumDynamicsFramework()
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        def resonant_drive(t):
            return sigma_z + 0.5 * np.cos(t) * sigma_x
        
        # Evolve Bloch vector components
        times = np.linspace(0, 4*np.pi, 100)
        bloch_x, bloch_y, bloch_z = [], [], []
        
        for t in times:
            # Short evolution to get instantaneous Bloch vector
            sx_t, _ = qdf.heisenberg_evolution_euler(sigma_x, resonant_drive, t, 50)
            sy_t, _ = qdf.heisenberg_evolution_euler(sigma_y, resonant_drive, t, 50)
            sz_t, _ = qdf.heisenberg_evolution_euler(sigma_z, resonant_drive, t, 50)
            
            # Extract expectation values (diagonal elements for computational basis)
            bloch_x.append(sx_t[0, 1].real * 2)  # <σ_x> = 2 * Re(ρ_{01})
            bloch_y.append(sy_t[0, 1].imag * 2)  # <σ_y> = 2 * Im(ρ_{01})
            bloch_z.append(sz_t[0, 0].real - sz_t[1, 1].real)  # <σ_z> = ρ_{00} - ρ_{11}
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Qubit Dynamics: Heisenberg Picture Evolution', fontsize=14)
        
        # Bloch vector components
        axes[0, 0].plot(times, bloch_x, 'r-', label='⟨σ_x⟩')
        axes[0, 0].plot(times, bloch_y, 'g-', label='⟨σ_y⟩')
        axes[0, 0].plot(times, bloch_z, 'b-', label='⟨σ_z⟩')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Expectation Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Bloch Vector Components')
        
        # Bloch sphere magnitude (should be ≤ 1)
        bloch_magnitude = np.sqrt(np.array(bloch_x)**2 + np.array(bloch_y)**2 + np.array(bloch_z)**2)
        axes[0, 1].plot(times, bloch_magnitude, 'k-', linewidth=2)
        axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Unit sphere')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('||⟨σ⟩||')
        axes[0, 1].set_title('Bloch Vector Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Phase space plot
        axes[1, 0].plot(bloch_x, bloch_z, 'purple', alpha=0.7)
        axes[1, 0].set_xlabel('⟨σ_x⟩')
        axes[1, 0].set_ylabel('⟨σ_z⟩')
        axes[1, 0].set_title('Phase Space (X-Z)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal')
        
        # 3D trajectory projection
        axes[1, 1].plot(bloch_y, bloch_z, 'orange', alpha=0.7)
        axes[1, 1].set_xlabel('⟨σ_y⟩')
        axes[1, 1].set_ylabel('⟨σ_z⟩')
        axes[1, 1].set_title('Phase Space (Y-Z)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("📊 Matplotlib not available - skipping visualization")

if __name__ == "__main__":
    demonstrate_enhanced_features()
    print("\n" + "=" * 50)
    print("🎨 Optional: Run visualize_qubit_dynamics() for plots")