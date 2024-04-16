import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import scipy.linalg as spalg


class Result:
    def __init__(self, n, m, counts):
        self.mat, labels = Result._unwrap_results(n, m, counts)
        self.labels1 = labels[0]
        self.labels2 = labels[1]
        self.counts = counts

    @staticmethod
    def _unwrap_results(n, m, counts: dict):
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

        return mat, (labels1, labels2)


class ITEQCircuit:
    def __init__(self, n: int, m: int, lambd: float, tau: float, psi0=None, trot_reps=1):
        self.n = n
        self.m = m
        self.lambd = lambd
        self.tau = tau
        self.qcircuit: QuantumCircuit = None
        self._use_had = False
        self.trot_reps = trot_reps

        if psi0 is None:
            self._use_had = True
            self.psi0 = np.ones(2 ** n) / np.sqrt(2 ** n)
        elif isinstance(psi0, (list, tuple)):
            self.psi0 = np.array(psi0)
        elif not isinstance(psi0, np.ndarray):
            raise ValueError(f"Psi0 must be a numpy array or a list or a tuple, not {psi0.__class__.__name__}")

        self._create_circuit()

    def _create_circuit(self):
        N = self.n
        M = self.m

        qc = QuantumCircuit(N + 2 * M)
        if self._use_had:
            qc.h([*range(2 * M, 2 * M + N)])
        else:
            qc.initialize(self.psi0, [*range(2 * M, 2 * M + N)])
        ite_qc = get_ite_circuit(N, M, self.lambd, self.tau / M, self.trot_reps)
        qc.append(ite_qc, [*range(0, 2 * M + N)])
        qc.h([*range(0, 2 * M)])
        qc.measure_all()

        self.qcircuit = qc

    def draw_circuit(self, decomposed=False, ax=None):
        if decomposed:
            qc = self.qcircuit.decompose()
        else:
            qc = self.qcircuit
        qc.draw("mpl", ax=ax)

    def get_results(self, num_shots=1_000_000) -> Result:
        backend = AerSimulator()
        tqc = transpile(self.qcircuit, backend)

        counts = backend.run(tqc, shots=num_shots).result().get_counts()
        return Result(self.n, self.m, counts)

    def get_theoretical_results(self):
        H = get_hamilt_op(self.n, self.lambd).to_matrix()
        alfa_p = 0.5 * (-1 + 1j)
        alfa_m = 0.5 * (-1 + -1j)
        # mat = alfa_m * spalg.expm(-1j * H * self.tau) + alfa_p * spalg.expm(1j * H * self.tau) + 2 * np.eye(H.shape[0])
        mat = spalg.expm(-H * self.tau)
        res = mat @ self.psi0
        res = res / np.sqrt(np.sum(np.abs(res) ** 2))
        return res

    def get_real_eigenvectors(self):
        H = get_hamilt_op(self.n, self.lambd).to_matrix()
        vals, vects = spalg.eigh(H, subset_by_index=[0, 1])
        return vals, vects

    def plot_summary(self, num_shots=1_000_000):
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(2, 1, 1)
        ax1 = plt.subplot(2, 2, 3)
        ax2 = plt.subplot(2, 2, 4)
        self.draw_circuit(ax=ax, decomposed=True)
        res = self.get_results(num_shots=num_shots)

        state = "0" * (2 * self.m)  # careful that qiskit use little endian to index
        evo_state_dict = {k[:-len(state)][::-1]: v for k, v in res.counts.items() if k.endswith(state)}

        thres = self.get_theoretical_results()
        print(f"For the state {state}, the theoretical probabilities are:")
        print(np.abs(thres) ** 2)
        print("The frequency of the states of the circuits are:")
        tot_state_shots = np.sum(res.mat[0, :])
        print(res.mat[0] / tot_state_shots)

        _, vects = self.get_real_eigenvectors()

        gs = ((np.abs(vects[:, 0]) + np.abs(vects[:, 1])) / 2) ** 2
        print("The groundstate of the hamiltonian is:")
        print(gs)
        plot_histogram(evo_state_dict, ax=ax1)
        plt.tight_layout()
        plt.show()

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


def get_ite_comp(H, dt, trot_reps):
    n = round(np.log2(H.dim[0]))
    qc = QuantumCircuit(n + 2, name="S")

    alpha = 2 * np.arctan(np.sqrt(0.5))
    beta = 0.5 * np.pi
    gamma = -1.5 * np.pi

    qc.ry(alpha, 0)
    qc.ry(beta, 1)
    qc.cx(0, 1, ctrl_state=0)
    qc.crz(gamma, control_qubit=0, target_qubit=1)

    evo = PauliEvolutionGate(H, -dt, label="U(-t/M)", synthesis=LieTrotter(reps=trot_reps)).control(1)
    qc.append(evo, [0, *range(2, n + 2)])  # control on qubit 0 and act on qubits 2 and 3

    evo = PauliEvolutionGate(H, 2 * dt, label="U(2t/M)", synthesis=LieTrotter(reps=trot_reps)).control(2)
    qc.append(evo, [0, 1, *range(2, n + 2)])  # control on qubits 0 and 1 and act on qubits 2 and 3

    return qc


def get_ite_circuit(n, m, lambd, dt, trot_reps=1) -> QuantumCircuit:
    H = get_hamilt_op(n, lambd)
    Scomp = get_ite_comp(H, dt, trot_reps)

    qc = QuantumCircuit(n + 2 * m, name="ITE")

    for i in range(m):
        qc = qc.compose(Scomp, qubits=[i * 2, i * 2 + 1, *range(-n, 0, 1)])

    return qc
