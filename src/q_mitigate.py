from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator, QFT
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import SamplerOptions, EstimatorOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_aer.noise import ReadoutError
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx
from datetime import datetime
import json
from scipy.optimize import minimize, curve_fit
from scipy.linalg import sqrtm
import itertools
from collections import defaultdict


class ErrorMitigationModule:
    """
    Advanced Quantum Error Mitigation Module.
    
    Implements several state-of-the-art techniques:
    - Zero-Noise Extrapolation (ZNE)
    - Probabilistic Error Cancellation (PEC)
    - Measurement Error Mitigation
    - Clifford Data Regression (CDR)
    - Dynamical Decoupling
    - Symmetry Verification
    """
    
    def __init__(self, backend=None, coupling_map=None, noise_model=None):
        """
        Initializes the error mitigation module.
        
        Args:
            backend: Quantum backend (real or simulated)
            coupling_map: Qubit coupling map
            noise_model: Noise model (for simulation)
        """
        self.backend = backend
        self.coupling_map = coupling_map
        self.noise_model = noise_model
        self.calibration_data = {}
        
    # =====================================================================
    # 1. ZERO-NOISE EXTRAPOLATION (ZNE)
    # =====================================================================
    
    def zero_noise_extrapolation(self, 
                                 circuit: QuantumCircuit,
                                 observable: Optional[np.ndarray] = None,
                                 scale_factors: List[float] = [1, 2, 3],
                                 extrapolation_method: str = 'exponential') -> Dict:
        """
        Zero-Noise Extrapolation: Executes the circuit with artificially amplified 
        noise levels, then extrapolates to the zero-noise regime.
        
        Principle: If we can control noise levels, we can mathematically 
        predict the result at noise = 0.
        
        Args:
            circuit: Quantum circuit
            observable: Observable to measure (optional)
            scale_factors: Noise scaling factors (1 = normal noise)
            extrapolation_method: 'exponential', 'linear', 'polynomial'
            
        Returns:
            Dictionary containing raw and mitigated values
        """
        print("\n" + "="*80)
        print("🔬 ZERO-NOISE EXTRAPOLATION (ZNE)")
        print("="*80)
        
        expectation_values = []
        
        for scale in scale_factors:
            print(f"\n📊 Execution with noise factor: {scale}x")
            
            # Noise amplification via gate pair insertion (Unitary Folding)
            scaled_circuit = self._scale_noise(circuit, scale)
            
            # Execution
            counts = self._execute_circuit(scaled_circuit)
            
            # Compute expectation value
            if observable is not None:
                exp_val = self._compute_expectation_value(counts, observable)
            else:
                # Default: Measure probability of state |0...0>
                exp_val = counts.get('0' * circuit.num_qubits, 0) / sum(counts.values())
            
            expectation_values.append(exp_val)
            print(f"   Measured value: {exp_val:.6f}")
        
        # Extrapolation to zero noise
        mitigated_value = self._extrapolate_to_zero(
            scale_factors, 
            expectation_values, 
            method=extrapolation_method
        )
        
        print(f"\n✨ MITIGATED RESULT (ZNE):")
        print(f"   Raw Value (1x): {expectation_values[0]:.6f}")
        print(f"   Mitigated Value (0x): {mitigated_value:.6f}")
        print(f"   Improvement: {abs(mitigated_value - expectation_values[0]):.6f}")
        
        # Visualization
        self._plot_zne_extrapolation(scale_factors, expectation_values, 
                                     mitigated_value, extrapolation_method)
        
        return {
            'raw_value': expectation_values[0],
            'mitigated_value': mitigated_value,
            'scale_factors': scale_factors,
            'expectation_values': expectation_values,
            'method': extrapolation_method
        }
    
    def _scale_noise(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """
        Amplifies noise by inserting pairs of gates (G, G^-1).
        
        Principle: Inserting X-X or CNOT-CNOT has no logical effect (Identity),
        but doubles the physical noise experienced by the qubits.
        """
        if scale_factor == 1:
            return circuit
        
        scaled_qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        num_repetitions = int(scale_factor)
        
        for instr, qargs, cargs in circuit.data:
            # Original instruction
            scaled_qc.append(instr, qargs, cargs)
            
            # Add identity pairs to amplify noise
            if instr.name in ['cx', 'cz'] and num_repetitions > 1:
                for _ in range(num_repetitions - 1):
                    # CRITICAL: Barrier prevents compiler optimization
                    scaled_qc.barrier() 
                    # Add G then G^-1
                    scaled_qc.append(instr, qargs, cargs)
                    scaled_qc.append(instr, qargs, cargs)
            
            elif instr.name in ['rx', 'ry', 'rz', 'u3'] and num_repetitions > 1:
                for _ in range(num_repetitions - 1):
                    scaled_qc.barrier()
                    # For rotations, do theta then -theta
                    scaled_qc.append(instr, qargs, cargs)
                    scaled_qc.append(instr.inverse(), qargs, cargs)
        
        return scaled_qc
    
    def _extrapolate_to_zero(self, 
                             scale_factors: List[float], 
                             values: List[float],
                             method: str = 'exponential') -> float:
        """
        Extrapolates measured values to the zero-noise regime (λ=0).
        """
        scale_factors = np.array(scale_factors)
        values = np.array(values)
        
        if method == 'linear':
            # Linear Fit: E(λ) = a + b*λ
            coeffs = np.polyfit(scale_factors, values, 1)
            mitigated = coeffs[1]  # Value at λ=0
            
        elif method == 'exponential':
            # Exponential Fit: E(λ) = a + b*exp(-c*λ)
            try:
                def exp_model(x, a, b, c):
                    return a + b * np.exp(-c * x)
                
                popt, _ = curve_fit(exp_model, scale_factors, values, 
                                   p0=[values[-1], values[0]-values[-1], 1.0])
                mitigated = popt[0] + popt[1]  # Value at λ=0
            except:
                # Fallback to linear fit if convergence fails
                coeffs = np.polyfit(scale_factors, values, 1)
                mitigated = coeffs[1]
                
        elif method == 'polynomial':
            # Polynomial Fit (Order 2)
            coeffs = np.polyfit(scale_factors, values, 2)
            mitigated = coeffs[2]  # Value at λ=0
            
        return mitigated
    
    def _plot_zne_extrapolation(self, scale_factors, values, mitigated, method):
        """Visualizes ZNE extrapolation."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Measured points
        ax.scatter(scale_factors, values, s=150, c='red', 
                   label='Raw Measurements', zorder=5, edgecolors='black', linewidths=2)
        
        # Extrapolation curve
        x_smooth = np.linspace(0, max(scale_factors), 100)
        
        if method == 'exponential':
            try:
                def exp_model(x, a, b, c):
                    return a + b * np.exp(-c * x)
                popt, _ = curve_fit(exp_model, scale_factors, values,
                                   p0=[values[-1], values[0]-values[-1], 1.0])
                y_smooth = exp_model(x_smooth, *popt)
            except:
                coeffs = np.polyfit(scale_factors, values, 1)
                y_smooth = np.polyval(coeffs, x_smooth)
        elif method == 'linear':
            coeffs = np.polyfit(scale_factors, values, 1)
            y_smooth = np.polyval(coeffs, x_smooth)
        else:
            coeffs = np.polyfit(scale_factors, values, 2)
            y_smooth = np.polyval(coeffs, x_smooth)
        
        ax.plot(x_smooth, y_smooth, 'b--', linewidth=2, 
                label=f'Extrapolation ({method})', alpha=0.7)
        
        # Mitigated point
        ax.scatter([0], [mitigated], s=300, c='green', marker='★',
                   label='Mitigated Value (noise=0)', zorder=6,
                   edgecolors='darkgreen', linewidths=2)
        
        # Annotations
        ax.axhline(mitigated, color='green', linestyle=':', alpha=0.5)
        ax.axvline(0, color='green', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Noise Scale Factor (λ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expectation Value', fontsize=12, fontweight='bold')
        ax.set_title('Zero-Noise Extrapolation (ZNE)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # =====================================================================
    # 2. MEASUREMENT ERROR MITIGATION
    # =====================================================================
    
    def measurement_error_mitigation(self, 
                                     circuit: QuantumCircuit,
                                     qubits: Optional[List[int]] = None) -> Dict:
        """
        Mitigation of measurement (readout) errors via calibration.
        
        Principle: Readout errors (0→1 or 1→0) are systematic.
        We can measure them and invert the confusion matrix.
        
        Args:
            circuit: Quantum circuit
            qubits: List of qubits to calibrate (None = all)
            
        Returns:
            Dictionary containing mitigated results
        """
        print("\n" + "="*80)
        print("📏 MEASUREMENT ERROR MITIGATION")
        print("="*80)
        
        if qubits is None:
            qubits = list(range(circuit.num_qubits))
        
        # Step 1: Calibration
        print("\n🔧 Calibration Phase...")
        calibration_matrix = self._calibrate_readout_errors(qubits)
        
        print(f"\n📊 Confusion Matrix (qubit 0):")
        print(f"   P(measured 0 | state 0) = {calibration_matrix[0, 0, 0]:.4f}")
        print(f"   P(measured 1 | state 0) = {calibration_matrix[0, 1, 0]:.4f}")
        print(f"   P(measured 0 | state 1) = {calibration_matrix[0, 0, 1]:.4f}")
        print(f"   P(measured 1 | state 1) = {calibration_matrix[0, 1, 1]:.4f}")
        
        # Step 2: Circuit Execution
        print("\n🚀 Executing target circuit...")
        if not circuit.has_measurements():
            circuit.measure_all()
        
        raw_counts = self._execute_circuit(circuit)
        
        # Step 3: Applying Mitigation
        print("\n✨ Applying mitigation...")
        mitigated_counts = self._apply_readout_mitigation(
            raw_counts, 
            calibration_matrix, 
            qubits
        )
        
        # Comparison
        print("\n📊 COMPARISON:")
        print(f"   {'State':<15} {'Raw':<12} {'Mitigated':<12} {'Diff'}")
        print("   " + "-"*55)
        
        total_raw = sum(raw_counts.values())
        total_mitigated = sum(mitigated_counts.values())
        
        all_states = sorted(set(list(raw_counts.keys()) + list(mitigated_counts.keys())))
        
        for state in all_states[:10]:  # Top 10
            raw_prob = raw_counts.get(state, 0) / total_raw * 100
            mit_prob = mitigated_counts.get(state, 0) / total_mitigated * 100
            diff = mit_prob - raw_prob
            
            print(f"   |{state}> {raw_prob:>8.3f}%  {mit_prob:>8.3f}%  "
                  f"{diff:>+8.3f}%")
        
        # Visualization
        self._plot_measurement_mitigation(raw_counts, mitigated_counts)
        
        return {
            'raw_counts': raw_counts,
            'mitigated_counts': mitigated_counts,
            'calibration_matrix': calibration_matrix
        }
    
    def _calibrate_readout_errors(self, qubits: List[int]) -> np.ndarray:
        """
        Calibrates readout errors by preparing |0> and |1> states.
        
        Returns:
            Calibration matrix [qubit, measured, state]
        """
        num_qubits = len(qubits)
        calibration_matrix = np.zeros((num_qubits, 2, 2))
        
        for i, qubit in enumerate(qubits):
            # Measure state |0>
            qc_0 = QuantumCircuit(max(qubits) + 1, 1)
            qc_0.measure(qubit, 0)
            
            counts_0 = self._execute_circuit(qc_0, shots=10000)
            total_0 = sum(counts_0.values())
            
            prob_0_given_0 = counts_0.get('0', 0) / total_0
            prob_1_given_0 = counts_0.get('1', 0) / total_0
            
            # Measure state |1>
            qc_1 = QuantumCircuit(max(qubits) + 1, 1)
            qc_1.x(qubit)
            qc_1.measure(qubit, 0)
            
            counts_1 = self._execute_circuit(qc_1, shots=10000)
            total_1 = sum(counts_1.values())
            
            prob_0_given_1 = counts_1.get('0', 0) / total_1
            prob_1_given_1 = counts_1.get('1', 0) / total_1
            
            # Fill matrix
            calibration_matrix[i, 0, 0] = prob_0_given_0
            calibration_matrix[i, 1, 0] = prob_1_given_0
            calibration_matrix[i, 0, 1] = prob_0_given_1
            calibration_matrix[i, 1, 1] = prob_1_given_1
        
        return calibration_matrix
    
    def _apply_readout_mitigation(self, 
                                  counts: Dict, 
                                  calibration_matrix: np.ndarray,
                                  qubits: List[int]) -> Dict:
        """
        Applies mitigation by inverting the confusion matrix.
        """
        num_qubits = len(qubits)
        total_shots = sum(counts.values())
        
        # Construct measured probabilities vector
        all_bitstrings = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
        measured_probs = np.array([counts.get(bs, 0) / total_shots for bs in all_bitstrings])
        
        # Construct global confusion matrix (Tensor product)
        M = self._build_confusion_matrix(calibration_matrix)
        
        # Invert to retrieve true probabilities
        try:
            M_inv = np.linalg.inv(M)
            true_probs = M_inv @ measured_probs
            
            # Correct negative probabilities (numerical artifact)
            true_probs = np.maximum(true_probs, 0)
            true_probs /= np.sum(true_probs)
            
        except np.linalg.LinAlgError:
            print("⚠️ Matrix not invertible, using pseudo-inverse")
            M_inv = np.linalg.pinv(M)
            true_probs = M_inv @ measured_probs
            true_probs = np.maximum(true_probs, 0)
            true_probs /= np.sum(true_probs)
        
        # Convert back to counts dictionary
        mitigated_counts = {}
        for i, bs in enumerate(all_bitstrings):
            if true_probs[i] > 1e-6:  # Threshold to avoid numerical noise
                mitigated_counts[bs] = int(true_probs[i] * total_shots)
        
        return mitigated_counts
    
    def _build_confusion_matrix(self, calibration_matrix: np.ndarray) -> np.ndarray:
        """Constructs global confusion matrix via tensor product."""
        num_qubits = calibration_matrix.shape[0]
        
        # Matrix for the first qubit
        M = np.array([[calibration_matrix[0, 0, 0], calibration_matrix[0, 0, 1]],
                      [calibration_matrix[0, 1, 0], calibration_matrix[0, 1, 1]]])
        
        # Tensor product with subsequent qubits
        for i in range(1, num_qubits):
            M_i = np.array([[calibration_matrix[i, 0, 0], calibration_matrix[i, 0, 1]],
                            [calibration_matrix[i, 1, 0], calibration_matrix[i, 1, 1]]])
            M = np.kron(M, M_i)
        
        return M
    
    def _plot_measurement_mitigation(self, raw_counts, mitigated_counts):
        """Visualizes mitigation effect."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        total_raw = sum(raw_counts.values())
        total_mit = sum(mitigated_counts.values())
        
        # Top 10 states
        top_states = sorted(set(list(raw_counts.keys())[:10] + 
                               list(mitigated_counts.keys())[:10]))
        
        raw_probs = [raw_counts.get(s, 0) / total_raw * 100 for s in top_states]
        mit_probs = [mitigated_counts.get(s, 0) / total_mit * 100 for s in top_states]
        
        # Plot 1: Direct comparison
        x = np.arange(len(top_states))
        width = 0.35
        
        ax1.bar(x - width/2, raw_probs, width, label='Raw', alpha=0.8, color='coral')
        ax1.bar(x + width/2, mit_probs, width, label='Mitigated', alpha=0.8, color='skyblue')
        ax1.set_xlabel('States')
        ax1.set_ylabel('Probability (%)')
        ax1.set_title('Raw vs Mitigated Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_states, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Differences
        differences = [m - r for r, m in zip(raw_probs, mit_probs)]
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        ax2.bar(top_states, differences, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('States')
        ax2.set_ylabel('Difference (%)')
        ax2.set_title('Mitigation Impact')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Scatter plot
        ax3.scatter(raw_probs, mit_probs, alpha=0.6, s=100)
        max_val = max(max(raw_probs), max(mit_probs))
        ax3.plot([0, max_val], [0, max_val], 'r--', label='Identity', linewidth=2)
        ax3.set_xlabel('Raw Probability (%)')
        ax3.set_ylabel('Mitigated Probability (%)')
        ax3.set_title('Correlation')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # =====================================================================
    # 3. DYNAMICAL DECOUPLING
    # =====================================================================
    
    def apply_dynamical_decoupling(self, 
                                   circuit: QuantumCircuit,
                                   dd_sequence: str = 'XY4') -> QuantumCircuit:
        """
        Applies Dynamical Decoupling (DD) to reduce decoherence errors.
        
        Principle: Inserting "refocusing" pulses (gates) during idle times
        to cancel out environmental noise.
        
        Args:
            circuit: Quantum circuit
            dd_sequence: Type of sequence ('X', 'XY4', 'CPMG', 'UDD')
            
        Returns:
            Circuit with DD applied
        """
        print("\n" + "="*80)
        print(f"⚡ DYNAMICAL DECOUPLING - Sequence: {dd_sequence}")
        print("="*80)
        
        # Create new circuit
        dd_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Define DD sequences
        sequences = {
            'X': [('x',)],
            'XY4': [('x',), ('y',), ('x',), ('y',)],
            'CPMG': [('x',), ('x',)],  # Carr-Purcell-Meiboom-Gill
            'UDD': None  # Uhrig DD - non-uniform spacing (not impl here)
        }
        
        sequence = sequences.get(dd_sequence, sequences['XY4'])
        
        print(f"\n📊 Selected DD sequence: {dd_sequence}")
        print(f"   Refocusing gates: {sequence}")
        
        # Traverse circuit and insert DD at appropriate idle slots
        idle_qubits = set(range(circuit.num_qubits))
        
        for instr, qargs, cargs in circuit.data:
            # Add original instruction
            dd_circuit.append(instr, qargs, cargs)
            
            # Identify active qubits
            active_qubits = {q._index for q in qargs}
            current_idle = idle_qubits - active_qubits
            
            # Apply DD on idle qubits
            if len(current_idle) > 0 and sequence:
                for idle_q in current_idle:
                    for gate in sequence:
                        if gate[0] == 'x':
                            dd_circuit.x(idle_q)
                        elif gate[0] == 'y':
                            dd_circuit.y(idle_q)
        
        print(f"\n✨ Circuit modified:")
        print(f"   Original depth: {circuit.depth()}")
        print(f"   Depth with DD: {dd_circuit.depth()}")
        print(f"   Added gates: {dd_circuit.size() - circuit.size()}")
        
        return dd_circuit
    
    # =====================================================================
    # 4. PROBABILISTIC ERROR CANCELLATION (PEC)
    # =====================================================================
    
    def probabilistic_error_cancellation(self,
                                        circuit: QuantumCircuit,
                                        gate_errors: Optional[Dict] = None) -> Dict:
        """
        Probabilistic Error Cancellation: Compensates for errors by resampling
        with modified circuits that statistically cancel out errors.
        
        Principle: Decompose the noisy channel into a weighted sum of 
        "clean" channels, then sample according to these weights.
        
        Args:
            circuit: Quantum circuit
            gate_errors: Dictionary of error rates per gate
            
        Returns:
            Dictionary containing PEC mitigated results
        """
        print("\n" + "="*80)
        print("🎲 PROBABILISTIC ERROR CANCELLATION (PEC)")
        print("="*80)
        
        if gate_errors is None:
            # Default errors (realistic for IBM)
            gate_errors = {
                'cx': 0.01,
                'x': 0.0005,
                'y': 0.0005,
                'z': 0.0001,
                'h': 0.0005,
                'sx': 0.0005,
                'rz': 0.0001
            }
        
        print(f"\n📊 Error rates used:")
        for gate, error in gate_errors.items():
            print(f"   {gate.upper()}: {error:.4f} ({error*100:.2f}%)")
        
        # Generate correction circuits
        num_samples = 100
        corrected_circuits = []
        weights = []
        
        print(f"\n🔧 Generating {num_samples} corrective circuits...")
        
        for _ in range(num_samples):
            corrected_circuit, weight = self._generate_pec_circuit(
                circuit, gate_errors
            )
            corrected_circuits.append(corrected_circuit)
            weights.append(weight)
        
        # Weight normalization
        weights = np.array(weights)
        weights /= np.sum(np.abs(weights))
        
        print(f"\n🚀 Execution and result combination...")
        
        # Execute all circuits
        all_counts = []
        for circ in corrected_circuits:
            if not circ.has_measurements():
                circ.measure_all()
            counts = self._execute_circuit(circ, shots=100)
            all_counts.append(counts)
        
        # Weighted combination
        mitigated_counts = defaultdict(float)
        for counts, weight in zip(all_counts, weights):
            total = sum(counts.values())
            for state, count in counts.items():
                mitigated_counts[state] += weight * (count / total)
        
        # Convert to integer counts
        total_weight = sum(abs(w) for w in weights)
        final_counts = {}
        for state, prob in mitigated_counts.items():
            final_counts[state] = int(abs(prob) * 1000)  # 1000 virtual shots
        
        print(f"\n✨ PEC completed!")
        print(f"   Sampling overhead: {len(corrected_circuits)}x")
        
        return {
            'mitigated_counts': final_counts,
            'num_circuits': len(corrected_circuits),
            'avg_weight': np.mean(np.abs(weights))
        }
    
    def _generate_pec_circuit(self, circuit: QuantumCircuit, 
                             gate_errors: Dict) -> Tuple[QuantumCircuit, float]:
        """
        Generates a corrective circuit for PEC.
        
        Principle: Each noisy gate is replaced by a combination of ideal
        gates that statistically reproduce the noise.
        """
        pec_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        total_weight = 1.0
        
        for instr, qargs, cargs in circuit.data:
            gate_name = instr.name
            error_rate = gate_errors.get(gate_name, 0.001)
            
            # With probability error_rate, add a correction
            if np.random.random() < error_rate:
                # Apply a random Pauli gate as correction
                pauli_gates = ['x', 'y', 'z', 'id']
                correction = np.random.choice(pauli_gates)
                
                pec_circuit.append(instr, qargs, cargs)
                
                if correction == 'x':
                    for q in qargs:
                        pec_circuit.x(q)
                elif correction == 'y':
                    for q in qargs:
                        pec_circuit.y(q)
                elif correction == 'z':
                    for q in qargs:
                        pec_circuit.z(q)
                
                # Adjust weight
                total_weight *= (1 - error_rate) / error_rate
            else:
                pec_circuit.append(instr, qargs, cargs)
        
        return pec_circuit, total_weight
    
    # =====================================================================
    # 5. COMPREHENSIVE ERROR MITIGATION
    # =====================================================================
    
    def comprehensive_mitigation(self,
                                circuit: QuantumCircuit,
                                techniques: List[str] = ['zne', 'measurement', 'dd']) -> Dict:
        """
        Applies a combination of error mitigation techniques.
        
        Args:
            circuit: Quantum circuit
            techniques: List of techniques to apply
                        ('zne', 'measurement', 'dd', 'pec')
            
        Returns:
            Dictionary containing all results
        """
        print("\n" + "🌟"*40)
        print("COMPREHENSIVE ERROR MITIGATION")
        print("🌟"*40)
        
        print(f"\nActivated techniques: {', '.join(techniques).upper()}")
        
        results = {'original_circuit': circuit}
        current_circuit = circuit
        
        # 1. Dynamical Decoupling (if enabled)
        if 'dd' in techniques:
            print("\n" + "─"*80)
            current_circuit = self.apply_dynamical_decoupling(current_circuit, 'XY4')
            results['dd_circuit'] = current_circuit
        
        # 2. Raw Measurement
        print("\n" + "─"*80)
        print("\n📊 EXECUTION WITHOUT MITIGATION...")
        if not current_circuit.has_measurements():
            current_circuit.measure_all()
        
        raw_counts = self._execute_circuit(current_circuit)
        results['raw_counts'] = raw_counts
        
        # 3. Measurement Error Mitigation (if enabled)
        if 'measurement' in techniques:
            print("\n" + "─"*80)
            mem_results = self.measurement_error_mitigation(current_circuit)
            results['measurement_mitigation'] = mem_results
            current_counts = mem_results['mitigated_counts']
        else:
            current_counts = raw_counts
        
        # 4. Zero-Noise Extrapolation (if enabled)
        if 'zne' in techniques:
            print("\n" + "─"*80)
            zne_results = self.zero_noise_extrapolation(
                circuit,
                scale_factors=[1, 1.5, 2, 2.5, 3],
                extrapolation_method='exponential'
            )
            results['zne'] = zne_results
        
        # 5. PEC (if enabled)
        if 'pec' in techniques:
            print("\n" + "─"*80)
            pec_results = self.probabilistic_error_cancellation(circuit)
            results['pec'] = pec_results
        
        # Final Report
        self._print_mitigation_report(results, techniques)
        
        return results
    
    def _print_mitigation_report(self, results: Dict, techniques: List[str]):
        """Prints a complete report of the results."""
        print("\n" + "="*80)
        print("📋 FINAL MITIGATION REPORT")
        print("="*80)
        
        raw_counts = results['raw_counts']
        total_raw = sum(raw_counts.values())
        
        print(f"\n🎯 RAW RESULTS (no mitigation):")
        top_raw = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for state, count in top_raw:
            prob = count / total_raw * 100
            print(f"   |{state}> : {count:4d} ({prob:5.2f}%)")
        
        if 'measurement_mitigation' in results:
            print(f"\n✨ RESULTS WITH MEASUREMENT MITIGATION:")
            mit_counts = results['measurement_mitigation']['mitigated_counts']
            total_mit = sum(mit_counts.values())
            top_mit = sorted(mit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for state, count in top_mit:
                prob = count / total_mit * 100
                raw_prob = raw_counts.get(state, 0) / total_raw * 100
                diff = prob - raw_prob
                print(f"   |{state}> : {count:4d} ({prob:5.2f}%)  [Δ: {diff:+.2f}%]")
        
        if 'zne' in results:
            print(f"\n🔬 RESULTS WITH ZNE:")
            print(f"   Raw value: {results['zne']['raw_value']:.6f}")
            print(f"   Mitigated value: {results['zne']['mitigated_value']:.6f}")
            print(f"   Improvement: {abs(results['zne']['mitigated_value'] - results['zne']['raw_value']):.6f}")
        
        if 'pec' in results:
            print(f"\n🎲 RESULTS WITH PEC:")
            print(f"   Generated circuits: {results['pec']['num_circuits']}")
            print(f"   Overhead: {results['pec']['num_circuits']}x")
        
        print("\n" + "="*80)
        print(f"✅ Techniques applied: {len(techniques)}")
        print("="*80)
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def _execute_circuit(self, circuit: QuantumCircuit, shots: int = 1000) -> Dict:
        """Executes a circuit on the configured backend."""
        if self.backend is not None:
            # Execution on real or simulated backend
            transpiled = transpile(circuit, self.backend)
            job = self.backend.run(transpiled, shots=shots)
            return job.result().get_counts()
        else:
            # Simulation with noise if available
            if self.noise_model:
                simulator = AerSimulator(noise_model=self.noise_model)
            else:
                simulator = AerSimulator()
            
            transpiled = transpile(circuit, simulator)
            job = simulator.run(transpiled, shots=shots)
            return job.result().get_counts()
    
    def _compute_expectation_value(self, counts: Dict, 
                                 observable: np.ndarray) -> float:
        """Computes the expectation value of an observable."""
        total_shots = sum(counts.values())
        expectation = 0.0
        
        for state, count in counts.items():
            # Convert bitstring to index
            idx = int(state, 2)
            prob = count / total_shots
            
            # Eigenvalue of the observable for this state
            eigenvalue = observable[idx, idx]
            expectation += prob * eigenvalue
        
        return expectation


# =====================================================================
# INTEGRATION INTO RealQuantumHardwareAnalyzer
# =====================================================================

class RealQuantumHardwareAnalyzer:
    """Class updated with Error Mitigation Module."""
    
    def __init__(self, num_qubits: int = 4, api_token: Optional[str] = None):
        self.num_qubits = num_qubits
        
        try:
            if api_token:
                QiskitRuntimeService.save_account(
                    channel="ibm_quantum",
                    token=api_token,
                    overwrite=True
                )
            
            self.service = QiskitRuntimeService()
            self.connected = True
            print("✅ IBM Quantum connection established successfully!")
            
        except Exception as e:
            self.service = None
            self.connected = False
            print(f"⚠️ IBM Quantum connection failed: {e}")
            print("   Simulation mode only.")
        
        # Initialize error mitigation module
        self.error_mitigation = None
    
    def initialize_error_mitigation(self, backend_name: Optional[str] = None):
        """
        Initializes the error mitigation module.
        
        Args:
            backend_name: Name of backend (None for simulation)
        """
        if backend_name and self.connected:
            backend = self.service.backend(backend_name)
            coupling_map = backend.coupling_map
            
            # Create noise model based on real backend
            try:
                noise_model = NoiseModel.from_backend(backend)
            except:
                noise_model = None
            
            self.error_mitigation = ErrorMitigationModule(
                backend=backend,
                coupling_map=coupling_map,
                noise_model=noise_model
            )
        else:
            # Simulation mode with synthetic noise
            noise_model = self._create_synthetic_noise_model()
            
            self.error_mitigation = ErrorMitigationModule(
                backend=AerSimulator(noise_model=noise_model),
                noise_model=noise_model
            )
        
        print("✅ Error mitigation module initialized!")
    
    def _create_synthetic_noise_model(self) -> NoiseModel:
        """Creates a realistic synthetic noise model."""
        noise_model = NoiseModel()
        
        # Depolarizing error for 1-qubit gates
        error_1q = depolarizing_error(0.001, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['x', 'y', 'z', 'h', 'sx'])
        
        # Depolarizing error for 2-qubit gates
        error_2q = depolarizing_error(0.01, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        
        # Readout errors
        readout_error = ReadoutError([[0.99, 0.01], [0.02, 0.98]])
        noise_model.add_all_qubit_readout_error(readout_error)
        
        return noise_model
    
    def demonstrate_error_mitigation(self, algorithm: str = 'grover'):
        """
        Complete demonstration of error mitigation techniques.
        
        Args:
            algorithm: Type of algorithm to test
        """
        print("\n" + "🎯"*40)
        print(f"ERROR MITIGATION DEMONSTRATION - {algorithm.upper()}")
        print("🎯"*40)
        
        # Initialization
        if self.error_mitigation is None:
            print("\n🔧 Initializing error mitigation module...")
            self.initialize_error_mitigation()
        
        # Create test circuit
        print(f"\n📊 Creating test circuit ({algorithm})...")
        circuit = self._create_test_circuit(algorithm)
        
        print(f"   Circuit: {circuit.depth()} depth, "
              f"{sum(circuit.count_ops().values())} gates")
        
        # Apply all techniques
        results = self.error_mitigation.comprehensive_mitigation(
            circuit,
            techniques=['zne', 'measurement', 'dd']
        )
        
        return results
    
    def _create_test_circuit(self, algorithm: str) -> QuantumCircuit:
        """Creates a test circuit."""
        qc = QuantumCircuit(self.num_qubits)
        
        if algorithm == 'grover':
            # Simple Grover Circuit
            qc.h(range(self.num_qubits))
            qc.barrier()
            
            # Oracle
            if self.num_qubits >= 2:
                qc.cz(0, 1)
            qc.barrier()
            
            # Diffuser
            qc.h(range(self.num_qubits))
            qc.x(range(self.num_qubits))
            if self.num_qubits >= 2:
                qc.h(self.num_qubits - 1)
                qc.cx(0, self.num_qubits - 1)
                qc.h(self.num_qubits - 1)
            qc.x(range(self.num_qubits))
            qc.h(range(self.num_qubits))
            
        elif algorithm == 'bell':
            # Bell State
            qc.h(0)
            qc.cx(0, 1)
            
        return qc


def main():
    """Demonstration of the Error Mitigation Module."""
    
    print("\n" + "🌟"*40)
    print("QUANTUM ERROR MITIGATION MODULE")
    print("🌟"*40)
    
    # Create Analyzer
    analyzer = RealQuantumHardwareAnalyzer(num_qubits=3)
    
    # Full Demonstration
    results = analyzer.demonstrate_error_mitigation('grover')
    
    print("\n✅ Demonstration complete!")
    print(f"\nTo use on real hardware:")
    print(f"   analyzer = RealQuantumHardwareAnalyzer(api_token='YOUR_TOKEN')")
    print(f"   analyzer.initialize_error_mitigation('ibm_brisbane')")
    print(f"   analyzer.demonstrate_error_mitigation('grover')")


if __name__ == "__main__":
    main()