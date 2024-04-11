import numpy as np
import scipy.sparse as sp


def get_hamiltonian(N, lambd, invert=False):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    if invert:
        sigma_x, sigma_z = sigma_z, sigma_x

    fmt = "csr"
    cross = sp.kron(sigma_x, sigma_x, fmt)
    total = sp.csr_matrix((2 ** N, 2 ** N))
    for i in range(N - 1):
        left = sp.kron(sp.identity(2 ** i, format=fmt), cross, format=fmt)
        total += sp.kron(left, sp.identity(2 ** (N - i - 2), format=fmt), format=fmt)

    if lambd:
        for i in range(N):
            left = sp.kron(sp.identity(2 ** i, format=fmt), sigma_z, format=fmt)
            total += lambd * sp.kron(left, sp.identity(2 ** (N - i - 1), format=fmt), format=fmt)

    return total


def real_it_op(H, tau):
    return np.diag(np.exp(np.diag(-H * tau)))


def unit_it_op(H, tau):
    alfa_p = np.exp(1j * 3 / 4 * np.pi) / np.sqrt(2)
    alfa_m = np.exp(-1j * 3 / 4 * np.pi) / np.sqrt(2)
    mat = np.diag(alfa_m * np.exp(np.diag(-1j * H * tau)) + alfa_p * np.exp(np.diag(1j * H * tau)) + 2)
    assert np.all(mat.imag == 0)
    return mat.real


N = 3
H = get_hamiltonian(N, 0, invert=True).toarray()
assert np.all(np.diag(np.diag(H)) == H)
psi_real = np.ones(2 ** N) / np.sqrt(2 ** N)
psi_unit = np.ones(2 ** N) / np.sqrt(2 ** N)

tau = 0.1
M = 1000

op_real = real_it_op(H, tau)
op_unit = unit_it_op(H, tau)

for i in range(M):
    psi_real = op_real @ psi_real
    psi_unit = op_unit @ psi_unit
    # psi = psi / psi[0]
    psi_real = psi_real / np.sqrt(np.sum(np.abs(psi_real) ** 2))
    psi_unit = psi_unit / np.sqrt(np.sum(np.abs(psi_unit) ** 2))
    print(psi_real, psi_unit)


