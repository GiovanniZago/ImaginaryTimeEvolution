import numpy as np
from matplotlib import pyplot as plt
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, IGate
from qiskit.circuit.library import RYGate, RZGate, HGate


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

    # mat = SparsePauliOp(hlist, coeffs=coeffs).to_matrix()
    # assert np.all(mat.imag == 0)
    # return mat.real
    return SparsePauliOp(hlist, coeffs=coeffs)


def create_Sgate(hamilt, time):
    N = round(np.log2(hamilt.dim[0]))
    qccust = QuantumCircuit(N)
    qccust.pauli("II", list(np.arange(N)))
    evo2 = PauliEvolutionGate(hamilt, time=time)
    qccust.append(evo2, list(np.arange(N)))
    qccust.draw("mpl")
    plt.show()


def test2():
    N = 2
    lambd = 0.1
    hamilt = get_qiskit_h(N, lambd)
    create_Sgate(hamilt, lambd)

if __name__ == '__main__':
    test2()