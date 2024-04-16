import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

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
    np.set_printoptions(precision=5)

    N = 3
    M = 4
    t = 1

    # setup Hamiltonian
    lam = 0.5
    ham = get_hamilt_op(N, lam).to_matrix()

    # print general info
    print("Matrix simulation")
    print(f"N = {N}, M = {M}, t = {t}, lambda = {lam}")

    # setup time evolution operator
    evop1 = expm(ham * t/M * 1j)
    evop2 = expm(ham * t/M * -1j)
    alpha_p = np.exp(3/4 * np.pi * 1j) / np.sqrt(2)
    alpha_m = np.exp(3/4 * np.pi * -1j) / np.sqrt(2)
    S_op = 2 * np.identity(2 ** N) + alpha_p * evop1 + alpha_m * evop2

    # inital state
    s_0 = np.ones(2 ** N)
    s_0 /= np.sqrt(np.sum(np.abs(s_0) ** 2))
    print("The initial state is ", s_0)

    cur_state = s_0

    for _ in range(M):
        cur_state = S_op @ cur_state

    cur_state /= np.sqrt(np.sum(np.abs(cur_state) ** 2))
    print("The final state is ", cur_state)
    print("The final probability densities are")
    print_state_probs(cur_state)

if __name__ == "__main__":
    main()