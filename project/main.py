from library import *


def main():
    N = 4
    M = 5
    tau = 2
    lambd = 0.5
    iteqc = ITEQCircuit(n=N, m=M, lambd=lambd, tau=tau, trot_reps=10)
    iteqc.plot_summary(num_shots=4_000_000)


if __name__ == '__main__':
    main()
