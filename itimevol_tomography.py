import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, partial_trace
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import Aer
from qiskit_experiments.library import StateTomography
from qiskit.visualization import array_to_latex

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
    N            = 2 # no. of system qubits
    M            = 1 # no. of time evolution steps
    n_anc        = 2 * M # no. of needed ancillary qubits
    t            = 1
    num_shots    = 1_000_000
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
    print(f"N = {N}, M = {M}, t = {t}, lambda = {lam}, reps = {reps}, num_shots = {num_shots}, device = {device}")















    """ 
    ===================================================================================
    ============================= Circuit simulation ==================================
    ===================================================================================
    """
    qc.measure_all()

    t_i = monotonic()

    qc_tom = StateTomography(qc)
    aer_bke = Aer.get_backend('aer_simulator_density_matrix_gpu')
    qc_job = qc_tom.run(backend=aer_bke, num_shots=num_shots, seed_simulation=100).block_for_results()

    t_f = monotonic()

    print(f"Simulation time: {(t_f - t_i):.2f} s")

    # get circuit total density matrix
    final_state = qc_job.analysis_results("state")
    final_dmatrix = final_state.value

    # trace out ancillary qubits and get system density matrix
    sys_dmatrix = partial_trace(final_dmatrix, anc_idxs)
    sys_dmatrix_dict = sys_dmatrix.to_dict()
    for k, v in sys_dmatrix_dict.items():
        print(f"{k}\t{v:.4f}")

if __name__ == "__main__":
    main()