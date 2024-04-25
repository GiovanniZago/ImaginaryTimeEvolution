from library import *


def main_ising():
    N = 2
    M = 5
    lambd = 0.5
    tau = 3.5
    iteqc = IsingQCircuit(n=N, m=M, lambd=lambd, tau=tau, trot_reps=10)
    iteqc.show_summary(num_shots=1_000_000)


def main_hydro():
    M = 5
    tau = 3.5
    h2qc = H2QCircuit(m=M, tau=tau, trot_reps=10)
    h2qc.show_summary(num_shots=1_000_000)


if __name__ == '__main__':
    # main_ising()
    main_hydro()
