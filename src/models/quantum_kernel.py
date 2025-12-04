import numpy as np
from math import ceil, log2

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumKernel:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.n_qubits = ceil(log2(n_features))
        self.dim = 2 ** self.n_qubits

    def _encode_state(self, x: np.ndarray) -> Statevector:
        # Zero-pad
        vec = np.zeros(self.dim)
        vec[: self.n_features] = x

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Cannot amplitude encode zero vector.")
        vec = vec / norm

        # Build circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(vec, range(self.n_qubits))

        # Get statevector (new API)
        state = Statevector.from_instruction(qc)
        return state

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

        # Encode all states
        states1 = [self._encode_state(x) for x in X1]
        states2 = [self._encode_state(z) for z in X2]

        # Compute kernel matrix
        K = np.zeros((len(states1), len(states2)))
        for i, s1 in enumerate(states1):
            for j, s2 in enumerate(states2):
                inner = np.vdot(s1.data, s2.data)
                K[i, j] = np.abs(inner) ** 2

        return K
