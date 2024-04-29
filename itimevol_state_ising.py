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
    M            = 8 # no. of time evolution steps
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

    v_min = np.min(np.abs(evs))
    v_max = np.max(np.abs(evs))
    print(f"Smallest eigenvalue: {evs[0]:.5f}")
    print(f"Biggest eigenvalue: {evs[-1]:.5f}")
    print(f"Smallest eigenvalue in magnitude v_min = {v_min:.5f}")
    print(f"Biggest eigenvalue in magnitude v_max = {v_max:.5f}")
    print(f"v_min * t / M = {v_min * t / M:.5f}     (v_min * t / M) ^ 3 = {(v_min * t / M) ** 3:.5f}")
    print(f"v_max * t / M = {v_max * t / M:.5f}     (v_max * t / M) ^ 3 = {(v_max * t / M) ** 3:.5f}")

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

    # compare final state with ground state
    print("Final state (Left) vs Ground state (Right)")
    utils.compare_states(final_state, gs)


    # compare final state with ground state
    print("Ground state (Left) vs First excited state (Right)")
    utils.compare_states(gs, w[1] / np.linalg.norm(w[1]))


    # print success probability
    print(f"The success probability wrt the gs is {np.abs(np.vdot(gs, final_state)) ** 2:.5f}")

if __name__ == "__main__":
    main()