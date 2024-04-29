import numpy as np
from scipy.linalg import expm, eigh

import utils

np.random.seed(9112001)   

def main():
    np.set_printoptions(precision=5)

    N = 3
    M = 7
    bound = 3
    t = 1

    # print general info
    print("Matrix simulation with random Hamiltonian")
    print(f"N = {N}, M = {M}, t = {t}, bound = {bound}")
    print("=============================================================")

    # setup Hamiltonian
    H = utils.get_random_H(N, bound=bound)
    print("Hamiltonian:\n", np.matrix(H))

    evs, w = eigh(H)
    w      = w.T
    gs     = w[0] / np.linalg.norm(w[0]) # we need to normalize it because later we calculate eps
    w_max  = w[np.argmax(np.abs(evs))]

    v_min = np.min(np.abs(evs))
    v_max = np.max(np.abs(evs))
    print(f"Smallest eigenvalue: {evs[0]:.5f}")
    print(f"Biggest eigenvalue: {evs[-1]:.5f}")
    print(f"Smallest eigenvalue in magnitude v_min = {v_min:.5f}")
    print(f"Biggest eigenvalue in magnitude v_max = {v_max:.5f}")
    print(f"v_min * t / M = {v_min * t / M:.5f}     (v_min * t / M) ^ 3 = {(v_min * t / M) ** 3:.5f}")
    print(f"v_max * t / M = {v_max * t / M:.5f}     (v_max * t / M) ^ 3 = {(v_max * t / M) ** 3:.5f}")

    # compare operators
    exact_op  = utils.get_exact_itimeop(H, M, t)
    approx_op = utils.get_approx_itimeop(H, M, t)

    diff = np.abs(exact_op - approx_op)

    print("\n")
    print("Relative difference between approx and exact operators (%)\n", np.matrix(diff / np.abs(exact_op)) * 100)

    print("\n")
    print(f"Average absolute distance between approx and exact operators: {np.mean(diff):.5f}")


    # initial state
    psi0 = np.ones(2 ** N)
    psi0 /= np.linalg.norm(psi0)

    print("\n")
    print("The initial state is")
    utils.print_state(psi0)
    print(f"The success probability between initial state and w_max is {np.abs(np.vdot(w_max, psi0)) ** 2:.5f}")

    # approximate final state
    psi = psi0

    for _ in range(M):
        psi = approx_op @ psi
        psi /= np.linalg.norm(psi)

    print("\n")
    print("Approximate final state (Left) vs Ground state (Right)")
    utils.compare_states(psi, gs)
    print(f"The success probability between approx and gs is {np.abs(np.vdot(gs, psi)) ** 2:.5f}")

    # exact final state with trotterization
    # psi = psi0

    # for _ in range(M):
    #     psi = exact_op @ psi
    #     psi /= np.linalg.norm(psi)

    # print("\n")
    # print("Exact final state with trotterization (Left) vs Ground state (Right)")
    # utils.compare_states(psi, gs)
    # print(f"The success probability between exact with trotterization and gs is {np.abs(np.vdot(gs, psi)) ** 2:.5f}")

    # exact final state
    psi = psi0

    psi = expm(-H * t) @ psi
    psi /= np.linalg.norm(psi)

    print("\n")
    print("Exact final state (Left) vs Ground state (Right)")
    utils.compare_states(psi, gs)
    print(f"The success probability between exact and gs is {np.abs(np.vdot(gs, psi)) ** 2:.5f}")

if __name__ == "__main__":
    main()