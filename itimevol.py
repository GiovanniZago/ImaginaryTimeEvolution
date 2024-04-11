import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, HGate
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

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

    return SparsePauliOp(hlist, coeffs=coeffs)

def main():
    backend = Aer.get_backend("qasm_simulator")
    num_shots = 4096

    N = 3 # no. of system qubits
    M = 2 # no. of time evolution steps
    n_anc = 2 * M # no. of needed ancillary qubits

    anc_idxs = list(range(n_anc))
    sys_idxs = list(range(n_anc, n_anc + N))

    alpha = 2 * np.arctan(np.sqrt(0.5))
    beta = 0.5 * np.pi
    gamma = -1.5 * np.pi

    anc = QuantumRegister(n_anc, name="a")
    syst = QuantumRegister(N, name="q")
    qc = QuantumCircuit(anc, syst)

    """ Initialize system qubits to an 
        equiprobable superposition of all
        the possible states.
    """
    const = np.sqrt(1 / (2 ** N))
    qc.initialize([const] * (2 ** N), sys_idxs)

    H = get_hamilt_op(N, 0.)
    t = 1.5

    for i in range(2 * M - 1, 0, -2): 
        """ Start from the last couple of qubits (most significative ones)
            and then go up until the least significative couple by steps 
            of two qubits.
        """
        # setup gates for ancillary qubits
        qc.ry(alpha, i - 1)
        qc.ry(beta, i)
        qc.cx(i - 1, i, ctrl_state = 0)
        qc.crz(gamma, control_qubit = i - 1, target_qubit = i)

        evo = PauliEvolutionGate(H, -t / M).control(1) # specify 1 control bit
        qc.append(evo, [i - 1] + sys_idxs) # control on qubit i - 1 and act on the system qubits

        evo = PauliEvolutionGate(H, 2 * t / M).control(2) # specify 2 control bit
        qc.append(evo, [i - 1, i] + sys_idxs) # control on qubits i - 1 and i and act on the system qubits

    # setup measurement qubits
    qc.h(anc_idxs)

    print(qc)

    """ Perform measurement and collect samples of the 
        system state on the subspace |00> of the 
        ancillary system.
    """
    qc.measure_all()

    results = execute(qc, backend=backend, num_shots=num_shots)
    counts = results.result().get_counts()

    evo_state_dict = {}

    for k, v in counts.items():
        pieces = list(k)

        """ Notice that in the following if statement the search for the state |0...0> of
            the ancillary qubits is performed starting from the right, because according 
            to Quiskit convenction, qubits with lower indexes are the less significant ones, 
            so they will be written from right to left.
        """
        if np.all([pieces[i] == '0' for i in range(N, N + n_anc)]): # check if the ancillary qubits are in the |0...0> state
            k_new = ''.join(pieces[:N])
            evo_state_dict[k_new] = v

    states = evo_state_dict.keys()
    counts = evo_state_dict.values()

    print(counts)
    plt.bar(states, counts)
    plt.xticks(rotation=90)
    plt.title(f"N = {N}, M = {M}, t = {t}, num_shots = {num_shots}")
    plt.show()


if __name__ == "__main__":
    main()