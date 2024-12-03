import numpy as np
from scipy.linalg import pinv, eig, cholesky, svd


def solve_orr_sommerfeld(alpha, beta, Re, U, Uy, Uyy, Dyy, Dyyyy, dy):
    n = len(U)
    k_sqrd = alpha**2 + beta**2
    Q = k_sqrd * np.eye(n) - Dyy + 0j

    # Orr Sommerfeld
    os_lhs = (
        -Q * U[:, None] * alpha * 1j
        - Uyy * np.eye(n) * alpha * 1j
        - 1 / Re * (Dyyyy - 2 * Dyy * k_sqrd + (k_sqrd**2) * np.eye(n))
    )

    os_rhs = Q + 0j

    # Squire
    s_lhs = -1 / Re * Q - alpha * (U[:, None] * np.eye(n)) * 1j
    s_rhs = np.eye(n) + 0j

    # Non normal term
    nn_lhs = 0 - Uy[:, None] * np.eye(n) * beta * 1j

    L = np.block([[os_lhs, np.zeros((n, n))], [nn_lhs, s_lhs]])
    M = np.block([[os_rhs, np.zeros((n, n))], [np.zeros((n, n)), s_rhs]])
    M_inv = pinv(M)
    L1 = np.dot(M_inv, L)
    I = np.eye(2 * n) + 0j

    # v = 0 at walls
    I[[0, n - 1], :] = 0 + 0j
    L1[[0, n - 1], :] = 0 + 0j
    L1[0, 0] = 1
    L1[n - 1, n - 1] = 1

    # du/dy = 0 at walls
    I[[1, n - 2]] = 0 + 0j
    L1[[1, n - 2]] = 0 + 0j
    L1[1, 0] = -11 / 6
    L1[1, 1] = 3
    L1[1, 2] = -3 / 2
    L1[1, 3] = 1 / 3
    L1[1, :] *= 1 / dy
    L1[n - 2, n - 1] = 11 / 6
    L1[n - 2, n - 2] = -3
    L1[n - 2, n - 3] = 3 / 2
    L1[n - 2, n - 4] = -1 / 3
    L1[n - 2, :] *= 1 / dy

    # vorticity = 0 at walls
    L1[n, :] = 0 + 0j
    L1[n, n] = 1 + 0j
    L1[n + n - 1, :] = 0 + 0j
    L1[n + n - 1, n + n - 1] = 1 + 0j
    I[n, :] = 0 + 0j
    I[n + n - 1, :] = 0 + 0j

    e_val, e_vec = eig(L1, I)

    omega = e_val * 1j

    return omega, e_val, e_vec, L1


def build_diff_operator(coeffs, n):
    operator = np.zeros((n, n))
    for coeffs_ in coeffs:
        n_coeffs = len(coeffs_)
        for i, coeff in enumerate(coeffs_):
            k = i - n_coeffs // 2

            col = np.arange(k, k + n)
            rows = np.arange(n)

            mask = (rows >= n_coeffs // 2) & (rows < n - (n_coeffs // 2))

            col = col[mask]
            rows = rows[mask]

            operator[rows, col] = coeff

    return operator


def calculate_transient_growth(alpha, beta, n, e_vec, e_val, coeffs_yy, t_array):
    F = calculate_F(alpha, beta, n, coeffs_yy)

    FV = np.dot(F, e_vec)
    FV_inv = pinv(FV)

    max_energy = [1]
    for t_ in t_array[1:]:
        G = np.dot(np.dot(FV, np.diag(np.exp(e_val * t_))), FV_inv)
        _, s, _ = svd(G)
        max_energy.append(s[0] ** 2)

    return max_energy


def calculate_resolvent(alpha, beta, n, e_vec, e_val, coeffs_yy, fr_array, fi_array):
    F = calculate_F(alpha, beta, n, coeffs_yy)

    FV = np.dot(F, e_vec)
    FV_inv = pinv(FV)

    out = []
    for fi in fi_array:
        row = []
        for fr in fr_array:
            freq = fr + fi * 1j
            A = np.dot(np.dot(FV, np.diag(1 / (freq - e_val * 1j))), FV_inv)
            _, s, _ = svd(A)
            row.append(s[0])
        out.append(row)

    out = np.array(out)
    return out


def calculate_F(alpha, beta, n, coeffs_yy):
    k_sqrd = alpha**2 + beta**2

    Dyy_ = np.zeros((n, n))
    middle_i = len(coeffs_yy[-1]) // 2
    for i, coeff in enumerate(coeffs_yy[-1]):
        j = i - middle_i
        Dyy_[np.eye(n, k=j, dtype="bool")] = coeffs_yy[-1][i]

    M = k_sqrd * np.eye(n) - Dyy_ + 0j

    Q = np.block(
        [
            [M, np.zeros((n, n))],
            [np.zeros((n, n)), np.eye(n, n)],
        ]
    )
    Q /= 2 * k_sqrd**2

    F = cholesky(Q)

    return F
