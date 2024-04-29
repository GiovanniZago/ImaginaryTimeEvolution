import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate, PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter


def get_ising_H(N, lambd):
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

def get_random_H(N, bound):
    N_h = 2 ** N
    A = 0.5 * (2 * bound * np.random.rand(N_h, N_h) + 2 * bound * 1j * np.random.rand(N_h, N_h) - bound) 
    H = A + A.conj().T

    return H

def get_itimevol_circuit(N, M, H, t, psi_0):
    n_anc = 2 * M
    

    anc_idxs = list(range(n_anc))
    sys_idxs = list(range(n_anc, n_anc + N))
    alpha    = 2 * np.arctan(np.sqrt(0.5))
    beta     = 0.5 * np.pi
    gamma    = -1.5 * np.pi
    anc      = QuantumRegister(n_anc, name="a")
    syst     = QuantumRegister(N, name="q")
    qc       = QuantumCircuit(anc, syst)

    qc.initialize(psi_0, sys_idxs)

    for i in range(2 * M - 1, 0, -2): 
        # setup gates for ancillary qubits
        qc.ry(alpha, i - 1)
        qc.ry(beta, i)
        qc.cx(i - 1, i, ctrl_state = 0)
        qc.crz(gamma, control_qubit = i - 1, target_qubit = i)

        U_1    = expm(1j * H * t / M)
        U_1_op = UnitaryGate(U_1, label="U(-t/M)").control(1)
        qc.append(U_1_op, [i - 1] + sys_idxs) # control on qubit i - 1 and act on the system qubits

        U_2    = expm(-2j * H * t / M)
        U_2_op = UnitaryGate(U_2, label="U(2t/M)").control(2)
        qc.append(U_2_op, [i - 1, i] + sys_idxs) # control on qubits i - 1 and i and act on the system qubits

    # setup measurement qubits
    qc.h(anc_idxs)

    return qc

def get_itimevol_circuit_ising(N, M, H, t, reps, psi_0):
    n_anc = 2 * M

    anc_idxs = list(range(n_anc))
    sys_idxs = list(range(n_anc, n_anc + N))
    alpha    = 2 * np.arctan(np.sqrt(0.5))
    beta     = 0.5 * np.pi
    gamma    = -1.5 * np.pi
    anc      = QuantumRegister(n_anc, name="a")
    syst     = QuantumRegister(N, name="q")
    qc       = QuantumCircuit(anc, syst)

    qc.initialize(psi_0, sys_idxs)

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

        evo = PauliEvolutionGate(H, -t / M, synthesis = LieTrotter(reps=reps)).control(1) # specify 1 control bit
        qc.append(evo, [i - 1] + sys_idxs) # control on qubit i - 1 and act on the system qubits

        evo = PauliEvolutionGate(H, 2 * t / M, synthesis = LieTrotter(reps=reps)).control(2) # specify 2 control bit
        qc.append(evo, [i - 1, i] + sys_idxs) # control on qubits i - 1 and i and act on the system qubits

    # setup measurement qubits
    qc.h(anc_idxs)

    return qc

def print_state(coeff):
    L = len(coeff)
    N = int(np.log2(L))
    b_vect = -1

    for c in coeff:
        if np.real(c) < 0:
            space = ""
        else:
            space = " "

        b_vect += 1
        state = "{:0{}b}".format(b_vect, N)
        p = np.abs(c) ** 2
        print(state + f": {space}{c:.4f} ({p:.4f})")

def compare_states(s1, s2):
    if len(s1) != len(s2):
        print("States have different length")

    L = len(s1)
    N = int(np.log2(L))
    b_vect = -1

    for c1, c2 in zip(s1, s2):
        if np.real(c1) < 0:
            space1 = ""
        else:
            space1 = " "

        if np.real(c2) < 0:
            space2 = ""
        else:
            space2 = " "

        b_vect += 1
        state = "{:0{}b}".format(b_vect, N)
        p1 = np.abs(c1) ** 2
        p2 = np.abs(c2) ** 2
        print(state + f": {space1}{c1:.4f} ({p1:.4f})     {space2}{c2:.4f} ({p2:.4f})")

def get_approx_itimeop(ham, M, t):
    evop1 = expm(ham * t/M * 1j)
    evop2 = expm(ham * t/M * -1j)
    alpha_p = np.exp(3/4 * np.pi * 1j) / np.sqrt(2)
    alpha_m = np.exp(3/4 * np.pi * -1j) / np.sqrt(2)
    S_op = 2 * np.identity(ham.shape[0]) + alpha_p * evop1 + alpha_m * evop2

    return S_op

def get_exact_itimeop(ham, M, t):
    exp_op = expm(-ham * t/M)

    return exp_op