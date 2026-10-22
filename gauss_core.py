"""Ядро лабораторної роботи №1: метод Гаусса з вибором головного елемента,
визначник, обернена матриця, число обумовленості, матриця Гільберта."""
import math
import re


# ---------- допоміжні функції ----------
def parse_number(tok: str) -> float:
    tok = tok.strip().replace(",", ".")
    if "/" in tok:
        a, b = tok.split("/", 1)
        return float(a) / float(b)
    return float(tok)


def parse_vector(text: str):
    toks = [t for t in re.split(r"[\s;]+", text.strip()) if t]
    return [parse_number(t) for t in toks]


def parse_matrix(text: str):
    rows = []
    for line in text.strip().splitlines():
        if line.strip():
            rows.append(parse_vector(line))
    if not rows:
        raise ValueError("Матрицю не задано.")
    n = len(rows)
    for i, r in enumerate(rows):
        if len(r) != n:
            raise ValueError(
                f"Матриця має бути квадратною: рядок {i + 1} містить {len(r)} "
                f"елементів, а рядків {n}.")
    return rows


def norm_rows(M):
    """||A|| = max_i sum_j |a_ij|"""
    return max(sum(abs(v) for v in row) for row in M)


def norm_cols(M):
    """||A|| = max_j sum_i |a_ij|"""
    return max(sum(abs(M[i][j]) for i in range(len(M))) for j in range(len(M[0])))


def vec_norm_inf(v):
    return max(abs(x) for x in v)


def mat_vec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def mat_mul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


# ---------- метод Гаусса ----------
def gauss_solve(A, b, eps=1e-12):
    """Прямий хід з вибором головного елемента по стовпцю + зворотний хід.
    Визначник обчислюється одночасно: det A = (-1)^l * a11 * a22^(1) * ... * amm^(m-1),
    де l — кількість перестановок рядків.

    Повертає словник: status in {'unique','inconsistent','infinite'}, x, det, rank.
    """
    n = len(A)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    scale = max(abs(v) for row in A for v in row) or 1.0
    tol = eps * scale

    swaps = 0
    r = 0
    pivots = []
    for c in range(n):
        if r >= n:
            break
        p = max(range(r, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) <= tol:
            continue                      # у стовпці немає ненульового ведучого елемента
        if p != r:
            M[r], M[p] = M[p], M[r]
            swaps += 1
        pivots.append(M[r][c])
        for i in range(r + 1, n):
            f = M[i][c] / M[r][c]
            if f != 0.0:
                for j in range(c, n + 1):
                    M[i][j] -= f * M[r][j]
            M[i][c] = 0.0
        r += 1
    rank = r

    if rank < n:
        btol = eps * max(scale, max(abs(v) for v in b))
        incompatible = any(abs(M[i][n]) > btol for i in range(rank, n))
        return {"status": "inconsistent" if incompatible else "infinite",
                "x": None, "det": 0.0, "rank": rank, "swaps": swaps}

    det = (-1) ** swaps
    for p in pivots:
        det *= p

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return {"status": "unique", "x": x, "det": det, "rank": rank, "swaps": swaps}


def gauss_inverse(A, eps=1e-12):
    """Обернена матриця методом Гаусса–Жордана з вибором головного елемента."""
    n = len(A)
    scale = max(abs(v) for row in A for v in row) or 1.0
    tol = eps * scale
    M = [list(map(float, A[i])) + [1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) <= tol:
            return None
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for i in range(n):
            if i != c:
                f = M[i][c]
                if f != 0.0:
                    M[i] = [a - f * bv for a, bv in zip(M[i], M[c])]
    return [row[n:] for row in M]


def cond_number(A, norm="rows", eps=1e-12):
    inv = gauss_inverse(A, eps)
    if inv is None:
        return None, None, None, None
    nf = norm_rows if norm == "rows" else norm_cols
    na, ni = nf(A), nf(inv)
    return na * ni, na, ni, inv


# ---------- матриця Гільберта ----------
def hilbert_system(m):
    """a_ij = 1/(i+j-1), x*_i = i^2, f = A x*."""
    A = [[1.0 / (i + j - 1) for j in range(1, m + 1)] for i in range(1, m + 1)]
    x_exact = [float(i * i) for i in range(1, m + 1)]
    f = mat_vec(A, x_exact)
    return A, f, x_exact