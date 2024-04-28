import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

from time import monotonic















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
    """ 
    =================================================================================
    ============================= Hyperparameters ===================================
    =================================================================================
    """
    N            = 3 # no. of system qubits
    M            = 3 # no. of time evolution steps
    n_anc        = 2 * M # no. of needed ancillary qubits
    t            = 1
    lam          = 0.5
    reps         = 10
    device       = "GPU"

    show_hist    = False
    show_circuit = False














    """ 
    ================================================================================
    ============================= Create circuit ===================================
    ================================================================================
    """
    H = get_hamilt_op(N, lam)
    
    anc_idxs = list(range(n_anc))
    sys_idxs = list(range(n_anc, n_anc + N))
    alpha    = 2 * np.arctan(np.sqrt(0.5))
    beta     = 0.5 * np.pi
    gamma    = -1.5 * np.pi
    anc      = QuantumRegister(n_anc, name="a")
    syst     = QuantumRegister(N, name="q")
    qc       = QuantumCircuit(anc, syst)

    const = np.sqrt(1 / (2 ** N)) 
    qc.initialize([const] * (2 ** N), sys_idxs) # initialize system state

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

    # princ circuit
    if show_circuit:
        print(qc)

    # print general info
    print("Qiskit Simulation")
    print(f"N = {N}, M = {M}, t = {t}, lambda = {lam}, reps = {reps}, device = {device}")















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
    result = aer_sim.run(qc_tp).result()
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

    # normalize system final state
    const = 0
    for v in final_state_sys_dict.values():
        const += np.abs(v) ** 2

    const = np.sqrt(const)
    final_state_sys_dict.update((k, v / const) for k, v in final_state_sys_dict.items())

    # print state
    print("The final state is")
    for k, v in final_state_sys_dict.items():
        print(f"{k[::-1]}: {v:.4f}")

    # print probabilities 
    print("The final probability densities are")
    for k, v in final_state_sys_dict.items():
        print(f"{k[::-1]}: {np.abs(v) ** 2:.4f}")














if __name__ == "__main__":
    main()