import numpy as np
import pandas as pd
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm, eigvalsh

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

def approx_evolution_step(ham, N, M, t, s_0, verb = False):
    evop1 = expm(ham * t/M * 1j)
    evop2 = expm(ham * t/M * -1j)
    alpha_p = np.exp(3/4 * np.pi * 1j) / np.sqrt(2)
    alpha_m = np.exp(3/4 * np.pi * -1j) / np.sqrt(2)
    S_op = 2 * np.identity(2 ** N) + alpha_p * evop1 + alpha_m * evop2

    evo_state = S_op @ s_0
    evo_state /= np.sqrt(np.sum(np.abs(evo_state) ** 2))

    if verb:
        print("Approximate evolution using S(t/M)")
        print("The final state is")
        print_state(evo_state)
        print("The final probability densities are")
        print_state_probs(evo_state)
        print("\n")

    return evo_state, S_op

def exact_evolution_step(ham, M, t, s_0, verb = False):
    exp_op = expm(-ham * t/M)
    
    evo_state = exp_op @ s_0
    evo_state /= np.sqrt(np.sum(np.abs(evo_state) ** 2))

    if verb:
        print("Exact evolution using exp(-Ht/M)")
        print("The final state is")
        print_state(evo_state)
        print("The final probability densities are")
        print_state_probs(evo_state)
        print("\n")

    return evo_state, exp_op
    

        

def main():
    np.set_printoptions(precision=5)

    N = 3
    M = 1
    t = 1

    verb = True

    # setup Hamiltonian
    lam = 0.5
    ham = get_hamilt_op(N, lam).to_matrix()

    # inital state
    s_0_S = np.ones(2 ** N)
    s_0_S /= np.sqrt(np.sum(np.abs(s_0_S) ** 2))
    s_0_exp = np.ones(2 ** N)
    s_0_exp /= np.sqrt(np.sum(np.abs(s_0_exp) ** 2))
    print("The initial state is ", s_0_S)

    # print general info
    print("Matrix simulation")
    print(f"N = {N}, M = {M}, t = {t}, lambda = {lam}")
    print("=============================================================")

    for i in range(M):
        if verb:
            print(f"** Step {i} **")
        state_cur_S, _ = exact_evolution_step(ham, M, t, s_0_exp, verb=verb)
        state_cur_exp, _ = approx_evolution_step(ham, N, M, t, s_0_S, verb=verb)
        s_0_S = state_cur_S
        s_0_exp = state_cur_exp

if __name__ == "__main__":
    main()