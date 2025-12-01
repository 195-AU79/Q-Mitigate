# 🛡️ Q-Mitigate: Advanced Quantum Error Mitigation

![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple)

**Q-Mitigate** is a comprehensive Python module designed to bridge the gap between noisy quantum hardware (NISQ) and theoretical results. It implements a unified pipeline of state-of-the-art error mitigation techniques.

## 🚀 Key Features

* **Zero-Noise Extrapolation (ZNE):** Extrapolates results to the zero-noise limit using Unitary Folding (exponential/linear fitting).
* **Measurement Error Mitigation:** Calibrates and inverts the readout confusion matrix to correct measurement errors.
* **Dynamical Decoupling (DD):** Automatically inserts XY4 sequences to suppress decoherence on idle qubits.
* **Probabilistic Error Cancellation (PEC):** Implements Monte-Carlo sampling to statistically cancel gate errors.
* **Real Hardware Analysis:** Seamless integration with IBM Quantum services via `QiskitRuntimeService`.

  FR : 
* **Extrapolation à Bruit Nul (ZNE):** Extrapole les résultats vers la limite de bruit nul en utilisant le Repliement Unitaire (ajustement exponentiel/linéaire).
* **Atténuation des Erreurs de Mesure:** Calibre et inverse la matrice de confusion de lecture pour corriger les erreurs de mesure.
* **Découplage Dynamique (DD):** Insère automatiquement des séquences XY4 pour supprimer la décohérence sur les qubits au repos.
* **Annulation Probabiliste d'Erreurs (PEC):** Implémente un échantillonnage de Monte-Carlo pour annuler statistiquement les erreurs de portes.
* **Analyse sur Matériel Réel:** Intégration transparente avec les services IBM Quantum via QiskitRuntimeService.
 
## 📦 Installation

```bash
git clone [https://github.com/195-AU79/Q-Mitigate.git](https://github.com/.../Q-Mitigate.git)
cd Q-Mitigate
pip install -r requirements.txt




