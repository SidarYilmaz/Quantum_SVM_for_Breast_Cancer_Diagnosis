import numpy as np
from math import ceil, log2
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

class QuantumKernel:
    def __init__(self, n_features):
        self.n_features = n_features
        self.n_qubits = ceil(log2(n_features))
        self.dim = 2 ** self.n_qubits

    def _encode_state(self, x: np.ndarray) -> Statevector:
        vec = np.zeros(self.dim)
        vec[:self.n_features] = x

        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Input vector must not be the zero vector.")
        vec /= norm

        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(vec, range(self.n_qubits))

        backend = Aer.get_backend('statevector_simulator')
        result = execute(qc, backend).result()
        state= result.get_statevector()

        return state
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1) 
        X2 = np.asarray(X2)

        states1 = [self._encode_state(x) for x in X1]
        states2 = [self._encode_state(x) for x in X2]

        K = np.zeros((len(states1), len(states2)))

        for i, s1 in enumerate(states1):
            for j, s2 in enumerate(states2):
                inner = np.vdot(s1.data, s2.data)
                K[i, j] = np.abs(inner) ** 2
        return K
    


        
    