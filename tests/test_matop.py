import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.linalg
import scipy.sparse.linalg as spalg
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, IGate
from qiskit.circuit.library import RYGate, RZGate, HGate


def get_hamiltonian(N, lambd):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    fmt = "csr"
    cross = sp.kron(sigma_z, sigma_z, fmt)
    total = sp.csc_matrix((2 ** N, 2 ** N))
    for i in range(N - 1):
        left = sp.kron(sp.identity(2 ** i, format=fmt), cross, format=fmt)
        total += sp.kron(left, sp.identity(2 ** (N - i - 2), format=fmt), format=fmt)

    if lambd:
        for i in range(N):
            left = sp.kron(sp.identity(2 ** i, format=fmt), sigma_x, format=fmt)
            total += lambd * sp.kron(left, sp.identity(2 ** (N - i - 1), format=fmt), format=fmt)

    return total


def get_hamilt_op(N, lambd):
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

    # mat = SparsePauliOp(hlist, coeffs=coeffs).to_matrix()
    # assert np.all(mat.imag == 0)
    # return mat.real
    return SparsePauliOp(hlist, coeffs=coeffs)

def get_Sop(hamilt, time):
    N = round(np.log2(hamilt.dim[0]))
    # alphap = 0.5 * (-1 + 1j)
    # alpham = 0.5 * (-1 - 1j)
    # mat = 2 * np.eye(2 ** N) + scipy.linalg.expm(1j * hamilt.toarray() * time) * alphap + scipy.linalg.expm(-1j * hamilt.toarray() * time) * alpham
    # return mat





if __name__ == '__main__':
    N = 10
    lambd = 0.1
    hamilt = get_hamiltonian(N, lambd)
    mat = get_Sop(hamilt, 0.1)
    a = 1
