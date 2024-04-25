import warnings

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import scipy.linalg as spalg


class Result:
    @staticmethod
    def get_matrix(n, m, counts: dict):
        mat = np.zeros((2 ** n) * (2 ** (m * 2)), dtype=int).reshape(2 ** (m * 2), (2 ** n))
        labels1 = [''] * 2 ** (m * 2)
        labels2 = [''] * (2 ** n)
        for k, v in counts.items():
            m_str = k[n:][::-1]
            n_str = k[:n][::-1]

            mi = int(m_str, 2)
            ni = int(n_str, 2)
            mat[mi, ni] = v

        for i in range(len(labels1)):
            labels1[i] = bin(i)[2:]

        for i in range(len(labels2)):
            labels2[i] = bin(i)[2:]

        return mat  # , (labels1, labels2)


def get_fidelity(x, y):
    if len(x.shape) == 1:
        return np.abs(np.sum(x @ y))
    
    elif len(x.shape) == 2:
        trace_xy = np.sum(np.diag(x @ y))
        trace_x2 = np.sum(np.diag(x @ x))
        trace_y2 = np.sum(np.diag(y @ y))
        return (trace_xy / np.sqrt(trace_x2 * trace_y2)).real
    
    raise ValueError("Cannot calcolate fidelity if operator has dim > 2")
    


def get_S(ham, tau):
    alfa_p = 0.5 * (-1 + 1j)
    alfa_m = 0.5 * (-1 + -1j)
    mat = alfa_m * spalg.expm(-1j * ham * tau) + alfa_p * spalg.expm(1j * ham * tau) + 2 * np.eye(ham.shape[0])
    return mat


def get_ite(ham, tau):
    return spalg.expm(-1 * ham * tau)


def normalize(x):
    return x / np.sqrt(np.sum(np.abs(x) ** 2))


def get_est_value(psi, op):
    psi_h = psi.conj()
    energy = psi_h @ op @ psi
    return energy


def get_energy(psi, ham):
    energy = get_est_value(psi, ham)

    # if matrix too big, do this:
    # for coeff, paul in zip(H.coeffs, H.paulis):
    #     energy += coeff * (state_h @ (paul.to_matrix() @ state))
    assert np.isclose(energy.imag, 0, atol=1e-14)
    return energy.real

def get_ground_eigh(ham):
    eigvals, eigvect = np.linalg.eigh(ham)
    return eigvals[0], eigvect[:, 0]


class ITEQCircuit:

    def __init__(self, n, m, tau, trot_reps=1, ham_kwargs=None, psi0=None):
        self.n = n
        self.m = m
        self.tau = tau
        self.trot_reps = trot_reps
        self.ham_kwargs = ham_kwargs or {}
        self._use_initial_hadamard = False
        self.qcircuit: QuantumCircuit = None

        self.debug_log = False

        if psi0 is None:
            self._use_initial_hadamard = True
            self.psi0 = np.ones(2 ** n) / np.sqrt(2 ** n)
        elif isinstance(psi0, (list, tuple)):
            self.psi0 = np.array(psi0)
        elif not isinstance(psi0, np.ndarray):
            raise ValueError(f"Psi0 must be a numpy array or a list or a tuple, not {psi0.__class__.__name__}")

        self._create_circuit()

        self.counts = None
        self.final_state = None

    def _print_log(self, *args):
        if self.debug_log:
            print(*args)

    def draw_circuit(self, decomposed=False, ax=None):
        if decomposed:
            qc = self.qcircuit.decompose()
        else:
            qc = self.qcircuit
        qc.draw("mpl", ax=ax)

    def get_real_eigenvectors(self, subset_by_index=None):
        ham = self.get_hamilt_op(self.n, **self.ham_kwargs).to_matrix()
        vals, vects = spalg.eigh(ham, subset_by_index=subset_by_index)
        return vals, vects

    def run_simulation(self, num_shots=1_000_000):
        self._print_log("Creating simulation")
        backend = AerSimulator(method="statevector")
        # backend = AerSimulator()
        tqc = transpile(self.qcircuit, backend)
        self._print_log("Running simulation")
        result = backend.run(tqc, shots=num_shots).result()
        self._print_log("Simulation Done")
        state_final = result.data()["final"].data
        counts = result.get_counts()
        self._print_log("Counts:", counts)
        self._store_results(counts, state_final)

    def get_theoretical_results(self):
        ham = self.get_hamilt_op(self.n, **self.ham_kwargs).to_matrix()
        ite_op = get_ite(ham, self.tau)
        res = ite_op @ self.psi0
        return normalize(res)

    def get_real_ground_eigs(self):
        vals, vects = self.get_real_eigenvectors(subset_by_index=[0, 1])
        state = vects[:, 0]
        if vals[0] == vals[1]:
            warnings.warn("Detected Degeneracy while calculating the real ground state")
            state = (vects[:, 0] + vects[:, 1]) / np.sqrt(2)

        return vals[0], state

    def _store_results(self, counts, final_state):
        self.counts = counts
        self.final_state = final_state

    def show_summary(self, num_shots=1_000_000, show_plot=True):
        ham = self.get_hamilt_op(self.n, **self.ham_kwargs).to_matrix()

        real_energy, real_gs = self.get_real_ground_eigs()
        print("Theoretical results from the eigen decomposition of H:")
        print(f"Energy: {real_energy}, state:", real_gs, "prob:", np.abs(real_gs) ** 2)

        state_th_ite = self.get_theoretical_results()
        energy_th_ite = get_energy(state_th_ite, ham)
        print("Theoretical results of the circuit using the ITE operator:")
        print(f"Energy: {energy_th_ite}, state:", state_th_ite, "prob:", np.abs(state_th_ite) ** 2)

        self.run_simulation(num_shots)
        counts, state_res = self.counts, self.final_state
        state_res = normalize(state_res[::2 ** (self.m * 2)])
        res_mat = Result.get_matrix(self.n, self.m, counts)
        state = "0" * (2 * self.m)  # careful that qiskit use little endian to index (in case of state != 00..00)
        evo_state_dict = {k[:-len(state)][::-1].strip(): v for k, v in counts.items() if k.endswith(state)}
        tot_state_shots = np.sum(res_mat[0, :])
        freqs = res_mat[0, :] / tot_state_shots
        energy = get_energy(state_res, ham)

        print("Results from the circuit simulation:")
        print(f"Energy: {energy}, state:", state_res, "prob:", np.abs(state_res) ** 2)
        print(f"Actual freqs: {freqs}")

        if show_plot:
            plt.figure(figsize=(12, 6))
            ax = plt.subplot(2, 1, 1)
            ax1 = plt.subplot(2, 2, 3)
            ax2 = plt.subplot(2, 2, 4)
            self.draw_circuit(ax=ax, decomposed=True)

            plot_histogram(evo_state_dict, ax=ax1)
            plt.tight_layout()
            plt.show()

    @classmethod
    def get_hamilt_op(cls, n: int, **kwargs):
        raise NotImplementedError()

    def _create_circuit(self):
        N = self.n
        M = self.m

        qr_anc = QuantumRegister(2 * M, "a")
        qr_work = QuantumRegister(N, "q")
        cr_anc = ClassicalRegister(2 * M, "c_a")
        cr_work = ClassicalRegister(N, "c_q")

        qc = QuantumCircuit(qr_anc, qr_work, cr_anc, cr_work)
        if self._use_initial_hadamard:
            qc.h(qr_work)
        else:
            qc.initialize(self.psi0, qr_work)

        ite_qc = self.get_ite_circuit(N, M, dt=self.tau / M, trot_reps=self.trot_reps, ham_kwargs=self.ham_kwargs)
        qc.append(ite_qc, qr_anc[:] + qr_work[:])
        qc.h(qr_anc)

        qc.save_statevector("final")
        qc.measure(qr_anc, cr_anc)
        qc.measure(qr_work, cr_work)
        # with qc.if_test((cr_anc, 0b00)) as else_:
        #     qc.save_statevector("final")

        self.qcircuit = qc

    @classmethod
    def get_ite_comp(cls, ham, dt, trot_reps):
        n = round(np.log2(ham.dim[0]))
        qc = QuantumCircuit(n + 2, name="S")

        alpha = 2 * np.arctan(np.sqrt(0.5))
        beta = 0.5 * np.pi
        gamma = -1.5 * np.pi

        qc.ry(alpha, 0)
        qc.ry(beta, 1)
        qc.cx(0, 1, ctrl_state=0)
        qc.crz(gamma, control_qubit=0, target_qubit=1)

        evo = PauliEvolutionGate(ham, -dt, label="U(-t/M)", synthesis=LieTrotter(reps=trot_reps)).control(1)
        qc.append(evo, [0, *range(2, n + 2)])  # control on qubit 0 and act on qubits 2 and 3

        evo = PauliEvolutionGate(ham, 2 * dt, label="U(2t/M)", synthesis=LieTrotter(reps=trot_reps)).control(2)
        qc.append(evo, [0, 1, *range(2, n + 2)])  # control on qubits 0 and 1 and act on qubits 2 and 3

        return qc

    @classmethod
    def get_ite_circuit(cls, n, m, dt, trot_reps, ham_kwargs=None) -> QuantumCircuit:
        ham_kwargs = ham_kwargs or {}
        ham = cls.get_hamilt_op(n=n, **ham_kwargs)
        ite_comp = cls.get_ite_comp(ham, dt, trot_reps)

        qc = QuantumCircuit(n + 2 * m, name="ITE")
        for i in range(m):
            qc = qc.compose(ite_comp, qubits=[i * 2, i * 2 + 1, *range(-n, 0, 1)])

        return qc


class IsingQCircuit(ITEQCircuit):
    def __init__(self, n: int, m: int, lambd: float, tau: float, psi0=None, trot_reps=1):
        super().__init__(n=n, m=m, tau=tau, psi0=psi0, trot_reps=trot_reps, ham_kwargs={"lambd": lambd})
        self.lambd = lambd

    @classmethod
    def get_hamilt_op(cls, n: int, **kwargs):
        hlist = []
        coeffs = []
        lambd = kwargs["lambd"]

        for i in range(n - 1):
            text = "I" * i + "ZZ" + "I" * (n - i - 2)
            hlist.append(text)
            coeffs.append(1)

        if lambd:
            for i in range(n):
                text = "I" * i + "X" + "I" * (n - i - 1)
                hlist.append(text)
                coeffs.append(lambd)

        return SparsePauliOp(hlist, coeffs=coeffs)


class H2QCircuit(ITEQCircuit):
    def __init__(self, m: int, tau: float, psi0=None, trot_reps=1):
        super().__init__(n=2, m=m, tau=tau, psi0=psi0, trot_reps=trot_reps)

    @classmethod
    def get_hamilt_op(cls, n, **kwargs):
        
        if "radius" in kwargs:
            # Table1 at https://journals.aps.org/prx/supplemental/10.1103/PhysRevX.8.011021/Supplementary.pdf
            r_coeffs = np.array([
                [0.05, 1.00777E+01, -1.05533E+00, 1.55708E-01, -1.05533E+00, 1.39333E-02],
                [0.10, 4.75665E+00, -1.02731E+00, 1.56170E-01, -1.02731E+00, 1.38667E-02],
                [0.15, 2.94817E+00, -9.84234E-01, 1.56930E-01, -9.84234E-01, 1.37610E-02],
                [0.20, 2.01153E+00, -9.30489E-01, 1.57973E-01, -9.30489E-01, 1.36238E-02],
                [0.25, 1.42283E+00, -8.70646E-01, 1.59277E-01, -8.70646E-01, 1.34635E-02],
                [0.30, 1.01018E+00, -8.08649E-01, 1.60818E-01, -8.08649E-01, 1.32880E-02],
                [0.35, 7.01273E-01, -7.47416E-01, 1.62573E-01, -7.47416E-01, 1.31036E-02],
                [0.40, 4.60364E-01, -6.88819E-01, 1.64515E-01, -6.88819E-01, 1.29140E-02],
                [0.45, 2.67547E-01, -6.33890E-01, 1.66621E-01, -6.33890E-01, 1.27192E-02],
                [0.50, 1.10647E-01, -5.83080E-01, 1.68870E-01, -5.83080E-01, 1.25165E-02],
                [0.55, -1.83734E-02, -5.36489E-01, 1.71244E-01, -5.36489E-01, 1.23003E-02],
                [0.65, -2.13932E-01, -4.55433E-01, 1.76318E-01, -4.55433E-01, 1.18019E-02],
                [0.75, -3.49833E-01, -3.88748E-01, 1.81771E-01, -3.88748E-01, 1.11772E-02],
                [0.85, -4.45424E-01, -3.33747E-01, 1.87562E-01, -3.33747E-01, 1.04061E-02],
                [0.95, -5.13548E-01, -2.87796E-01, 1.93650E-01, -2.87796E-01, 9.50345E-03],
                [1.05, -5.62600E-01, -2.48783E-01, 1.99984E-01, -2.48783E-01, 8.50998E-03]
            ])
            
            radius = kwargs["radius"]
            index = np.where(r_coeffs[:, 0] == radius)[0]
            if len(index) == 0:
                raise ValueError(f"Radius {radius} not found in table for the H2 hamiltonian")
            
            coeffs = r_coeffs[index, 1:].flatten()
        else:
            # Coeffs for R = 0.45 angstrong. 
            coeffs = [2.67547E-01, -6.33890E-01, 1.66621E-01, -6.33890E-01, 1.27192E-02]
        return SparsePauliOp(["II", "ZI", "XX", "IZ", "ZZ"], coeffs=coeffs)
