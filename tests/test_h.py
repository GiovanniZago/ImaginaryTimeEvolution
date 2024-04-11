import numpy as np
import scipy.sparse as sp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def get_hamiltonian(N, lambd):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    fmt = "csr"
    cross = sp.kron(sigma_z, sigma_z, fmt)
    total = sp.csr_matrix((2 ** N, 2 ** N))
    for i in range(N - 1):
        left = sp.kron(sp.identity(2 ** i, format=fmt), cross, format=fmt)
        total += sp.kron(left, sp.identity(2 ** (N - i - 2), format=fmt), format=fmt)

    if lambd:
        for i in range(N):
            left = sp.kron(sp.identity(2 ** i, format=fmt), sigma_x, format=fmt)
            total += lambd * sp.kron(left, sp.identity(2 ** (N - i - 1), format=fmt), format=fmt)

    return total.toarray()


def get_qiskit_h(N, lambd):
    hlist = []
    coeffs = []
    for i in range(N - 1):
        text = "I" * i + "ZZ" + "I" * (N - i - 2)
        hlist.append(text)
        coeffs.append(1)

    if lambd:
        for i in range(N):
            text = "I" * i + "X" + "I" * (N - i - 1)
            hlist.append(text)
            coeffs.append(lambd)

    mat = SparsePauliOp(hlist, coeffs=coeffs).to_matrix()
    assert np.all(mat.imag == 0)
    return mat.real


def test1():
    for n in range(2, 8):
        for lambd in np.arange(0, 3, 0.5):
            h1 = get_hamiltonian(n, lambd)
            h2 = get_qiskit_h(n, lambd)
            assert np.all(h1 == h2)

