import numpy as np
from scipy.linalg import eigh, expm
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import Aer

from time import monotonic

np.random.seed(9112001)












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






def get_random_H(N, bound):
    N_h = 2 ** N
    A = 0.5 * (2 * bound * np.random.rand(N_h, N_h) + 2 * bound * 1j * np.random.rand(N_h, N_h) - bound) 
    H = A + A.conj().T

    return H


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
        b_vect += 1
        state = "{:0{}b}".format(b_vect, N)
        print(state + f": {c:.4f}")

def print_state_probs(coeff):
    L = len(coeff)
    N = int(np.log2(L))
    b_vect = -1

    for c in coeff:
        b_vect += 1
        state = "{:0{}b}".format(b_vect, N)
        p = np.abs(c) ** 2
        print(state + f": {p:.4f}")






def main():
    """ 
    =================================================================================
    ============================= Hyperparameters ===================================
    =================================================================================
    """
    N            = 2 # no. of system qubits
    M            = 8 # no. of time evolution steps
    n_anc        = 2 * M # no. of needed ancillary qubits
    t            = 1
    lam          = 1
    reps         = 10
    device       = "GPU"

    # print general info
    print("Statevector simulation with Ising Hamiltonian")
    print(f"N = {N}, M = {M}, t = {t}, lambda = {lam}, device = {device}")
    print("=============================================================")











    """ 
    ================================================================================
    ============================= Create circuit ===================================
    ================================================================================
    """
    # setup Hamiltonian
    H = get_hamilt_op(N, lam)

    evs, w = eigh(H)
    w      = w.T
    gs     = w[0] / np.linalg.norm(w[0]) # we need to normalize it because later we calculate eps

    v_min = np.min(np.abs(evs))
    v_max = np.max(np.abs(evs))
    print(f"Smallest eigenvalue: {evs[0]:.5f}")
    print(f"Biggest eigenvalue: {evs[-1]:.5f}")
    print(f"Smallest eigenvalue in magnitude v_min = {v_min:.5f}")
    print(f"Biggest eigenvalue in magnitude v_max = {v_max:.5f}")
    print(f"v_max * t / M = {v_max * t / M:.5f}")

    psi0 = np.ones(2 ** N)
    psi0 /= np.linalg.norm(psi0)
        
    qc = get_itimevol_circuit_ising(N, M, H, t, reps, psi0)














    """ 
    ===================================================================================
    ============================= Circuit simulation ==================================
    ===================================================================================
    """
    qc.save_statevector()

    t_i = monotonic()

    if device == "GPU":
        device_str = "_" + device.lower()
    else:
        device_str = ""

    aer_sim = Aer.get_backend("aer_simulator_statevector" + f"{device_str}")
    qc_tp = transpile(qc, backend=aer_sim)
    result = aer_sim.run(qc_tp, shots=1).result()
    final_state_dict = result.get_statevector(qc_tp).to_dict()

    t_f = monotonic()

    print(f"Simulation time: {(t_f - t_i):.2f} s")















    """ 
    =================================================================================
    ============================= State selection ===================================
    ============================= and run stats   ===================================
    =================================================================================
    """
    final_state_sys_dict = {}
    for k, v in final_state_dict.items():
        if k[-n_anc:] == '0' * n_anc: # check if the ancillary qubits are in the |0...0> state
            k_new = k[:N]
            final_state_sys_dict[k_new] = v

    final_state_sys_list = [(k, v) for k, v in final_state_sys_dict.items()] # reverse keys because of default Qiskit little endian

    # convert final state to numpy array
    final_state = np.array([t[1] for t in sorted(final_state_sys_list)])
    final_state /= np.linalg.norm(final_state)

    # print theoretical ground state
    print("The theoretical ground state is")
    print_state(gs)

    # print probabilities 
    print("The theoretical ground state final probability densities are")
    print_state_probs(gs)

    # print state
    print("The final state is")
    for el in final_state_sys_list:
        print(f"{el[0]}:    {[np.real(el[1]) / np.linalg.norm(final_state), np.imag(el[1]) / np.linalg.norm(final_state)]}")

    # print probabilities 
    print("The final probability densities are")
    print_state_probs(final_state)


    # print success probability
    print(f"The success probability is {np.abs(np.vdot(gs, final_state)) ** 2:.5f}")










if __name__ == "__main__":
    main()