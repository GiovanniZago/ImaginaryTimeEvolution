import numpy as np
from scipy.linalg import eigh
from qiskit import transpile
from qiskit_aer import Aer
from time import monotonic

import utils

np.random.seed(9112001)

def main():
    """ 
    =================================================================================
    ============================= Hyperparameters ===================================
    =================================================================================
    """
    N            = 2 # no. of system qubits
    M            = 2 # no. of time evolution steps
    n_anc        = 2 * M # no. of needed ancillary qubits
    t            = 1
    lam          = 0
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
    H = utils.get_ising_H(N, lam)

    evs, w = eigh(H)
    w      = w.T
    gs     = w[0] / np.linalg.norm(w[0]) # we need to normalize it because later we calculate eps

    print(f"Eigenvalues: {np.matrix(evs)}")

    psi0 = np.ones(2 ** N)
    psi0 /= np.linalg.norm(psi0)
        
    qc = utils.get_itimevol_circuit_ising(N, M, H, t, reps, psi0)

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

    # convert final state to numpy array
    final_state_sys_list = [(k, v) for k, v in final_state_sys_dict.items()]
    final_state = np.array([t[1] for t in sorted(final_state_sys_list)])
    final_state /= np.linalg.norm(final_state)

    # print initial state
    print("The initial state is")
    utils.print_state(psi0)

    # compare final state with another eigenvector
    w_index = 0
    print(f"Final state (Left) vs Eigenvector with index {w_index} (Right)")
    utils.compare_states(final_state, w[w_index])

    w_index = 1
    print(f"Final state (Left) vs Eigenvector with index {w_index} (Right)")
    utils.compare_states(final_state, w[w_index])

    print(f"Final state energy: {np.vdot(final_state, np.dot(H.to_matrix(), final_state)):.5f}")

if __name__ == "__main__":
    main()