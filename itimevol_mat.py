import numpy as np
import pandas as pd
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm, eigh

np.random.seed(9112001)

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

def get_approx_itimeop(ham, M, t):
    evop1 = expm(ham * t/M * 1j)
    evop2 = expm(ham * t/M * -1j)
    alpha_p = np.exp(3/4 * np.pi * 1j) / np.sqrt(2)
    alpha_m = np.exp(3/4 * np.pi * -1j) / np.sqrt(2)
    S_op = 2 * np.identity(ham.shape[0]) + alpha_p * evop1 + alpha_m * evop2

    return S_op

def get_exact_itimeop(ham, M, t):
    exp_op = expm(-ham * t/M)

    return exp_op
    
def get_random_H(N, bound):
    N_h = 2 ** N
    A = 0.5 * (2 * bound * np.random.rand(N_h, N_h) + 2 * bound * 1j * np.random.rand(N_h, N_h) - bound) 
    H = A + A.conj().T

    return H
        

def main():
    np.set_printoptions(precision=5)

    N = 2
    M = 8
    bound = 1
    t = 1

    # print general info
    print("Matrix simulation with random Hamiltonian")
    print(f"N = {N}, M = {M}, t = {t}, bound = {bound}")
    print("=============================================================")

    # setup Hamiltonian
    H = get_random_H(N, bound=bound)

    evs, w = eigh(H)
    w      = w.T
    gs     = w[0] / np.linalg.norm(w[0]) # we need to normalize it because later we calculate eps

    v_min = np.min(np.abs(evs))
    v_max = np.max(np.abs(evs))
    print(f"Smallest eigenvalue: {evs[0]:.5f}")
    print(f"Biggest eigenvalue: {evs[-1]:.5f}")
    print(f"Smallest eigenvalue in magnitude v_min = {v_min:.5f}")
    print(f"Biggest eigenvalue in magnitude v_max = {v_max:.5f}")

    # compare operators
    exact_op  = get_exact_itimeop(H, M, t)
    approx_op = get_approx_itimeop(H, M, t)

    diff = np.abs(exact_op - approx_op)

    print(f"v_min * t / M = {v_min * t / M:.5f}     (v_min * t / M) ^ 3 = {(v_min * t / M) ** 3:.5f}")
    print(f"v_max * t / M = {v_max * t / M:.5f}     (v_max * t / M) ^ 3 = {(v_max * t / M) ** 3:.5f}")
    print("Relative difference between approx and exact\n", np.matrix(diff / np.abs(exact_op)))

    # print theoretical ground state
    print("The theoretical ground state is")
    print_state(gs)

    print("The theoretical ground state final probability densities are")
    print_state_probs(gs)

    # approximate final state
    psi = np.ones(2 ** N)
    psi /= np.linalg.norm(psi)

    for _ in range(M):
        psi = approx_op @ psi
        psi /= np.linalg.norm(psi)

    print("The final state using the approximate operator is")
    print_state(psi)

    print("The final probability densities using the approximate operator are")
    print_state_probs(psi)
    psi_approx = psi

    # exact final state
    psi = np.ones(2 ** N)
    psi /= np.linalg.norm(psi)

    # for _ in range(M):
    #     psi = exact_op @ psi
    #     psi /= np.linalg.norm(psi)
    psi = expm(-H * t) @ psi
    psi /= np.linalg.norm(psi)

    print("The final state using the exact operator is")
    print_state(psi)

    print("The final probability densities using the exact operator are")
    print_state_probs(psi)

    # print success probability
    print(f"The success probability between approx and gs is {np.abs(np.vdot(gs, psi_approx)) ** 2:.5f}")
    print(f"The success probability between exact and gs is {np.abs(np.vdot(gs, psi)) ** 2:.5f}")

if __name__ == "__main__":
    main()