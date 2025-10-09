import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from matplotlib import pyplot as plt

# Random objects generation

def generate_X(A, D, Model_X):
    if Model_X == True:
        x = np.random.randn(D, A)
        X = x.T @ x
        return X
    else:
        x = np.random.randint(-1, 1, size=(D, A))
        X = x.T @ x
        return X

def generate_Y(A, Model_Y):
    if Model_Y == True:
        off_diag = -np.abs(np.random.randn(A, A))
        Y = np.triu(off_diag, 1) + np.triu(off_diag, 1).T
        np.fill_diagonal(Y, -np.sum(Y, axis=1))
        return Y
    else:
        off_diag = -np.abs(np.random.randint(-1, 2, size=(A, A)))
        Y = np.triu(off_diag, 1) + np.triu(off_diag, 1).T
        np.fill_diagonal(Y, -np.sum(Y, axis=1))
        return Y

def generate_D(A):
    diagonal = abs(np.random.randn(A))
    return np.diag(diagonal)

def compute_M(X, Y):
    I = np.eye(X.shape[0])
    return np.linalg.inv(I + X @ Y) @ X

def compute_default_diag_dom(M, tol=10e-6):
    A = M.shape[0]
    N = np.zeros((A,A))
    np.fill_diagonal(N, 0)
    for a in range(A):
        for b in range(A):
            if a != b and M[a, b] > M[a, a] + tol:
                N[a, b] = M[a, b]
                N[b, a] = M[b, a]
                N[a, a] = M[a, a]
    return N

def generate_X_gaussian_good(A, D, Model_X):
    Y = A * np.eye(A) - np.ones((A, A))
    while True:
        X = generate_X(A, D, Model_X)
        MX = compute_M(X, Y)
        if is_diag_dominant(MX):
            return X

def generate_X_metagood(A, D, nb_iter, Model_X):
    while True:
        X = generate_X_gaussian_good(A, D, Model_X)
        count = 0
        for _ in range(nb_iter):
            Y = generate_Y(A, True)
            M = compute_M(X, Y)
            if is_diag_dominant(M):
                count += 1
        if count == nb_iter:
            return X

def generate_x_publisher(A, D):
    As = np.random.multinomial(A - D, np.ones(D) / D) + 1  # Assure A_d > 0
    x = np.zeros((D, A), dtype=int)
    start = 0
    for d in range(D):
        x[d, start:start + As[d]] = 1
        start += As[d]
    return x

def generate_block_diagonal_J(A, D):
    block_sizes = np.random.multinomial(A, [1 / D] * D)
    while np.any(block_sizes == 0):
        block_sizes = np.random.multinomial(A, [1 / D] * D)
    K = np.zeros((A, A))
    start = 0
    for size in block_sizes:
        K[start:start + size, start:start + size] = np.ones((size, size))
        start += size
    return K

def generate_block_diagonal_matrix(X1, X2):
    # Créer une matrice de blocs diagonaux
    X = np.block([[X1, np.zeros((X1.shape[0], X2.shape[1]))],
                   [np.zeros((X2.shape[0], X1.shape[1])), X2]])
    return X
# The tests:

def is_positive_definite(X):
    if not np.allclose(X, X.T):
        return False
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False

def is_gaussian_good(X):
    A = np.shape(X)[0]
    Y = generate_Y(A, True)
    M = compute_M(X, Y)
    if not is_diag_dominant(M):
        return False, M
    return True, None

def is_metagood(X, nb_iter, Model_Y=False):
    for _ in range(nb_iter):
        A = np.shape(X)[0]
        Y = generate_Y(A, Model_Y)
        M = compute_M(X, Y)
        if not is_diag_dominant(M):
            return False, Y, M
    return True, None, None

def is_diag_dominant(M, tol=1e-4):
    A = M.shape[0]
    for a in range(A):
        if np.any(M[a, :] > M[a, a] + tol):
            return False
    return True


def is_super_laplacian(matrix):
    n = matrix.shape[0]
    for i in range(n):
        diagonal_term = matrix[i, i]

        if diagonal_term < 0:
            print("diagonal term = ", diagonal_term)
            return False

        off_diagonal_sum = np.sum(np.abs(matrix[i])) - np.abs(diagonal_term)

        if diagonal_term <= off_diagonal_sum:
            print("diagonal_term - off_diagonal_sum = ", diagonal_term - off_diagonal_sum)
            return False

        for j in range(n):
            if i != j and matrix[i, j] > 0:
                print("matrix[i,j] = ", matrix[i,j])
                return False

    return True

def test_X1_plus_X2(A, D, nb_iter, Model_X, Model_Y=False):
    X1 = generate_X_metagood(A, D, nb_iter, Model_X)
    X2 = generate_X_metagood(A, D, nb_iter, Model_X)
    X = X1 + X2
    Y = generate_Y(A, Model_Y)
    MX = compute_M(X, Y)
    return X1, X2, is_diag_dominant(MX)

#Other

def block_diagonal(X, Y):
    Nx, Ny = X.shape[0], Y.shape[0]
    Zx = np.zeros((Nx, Ny))
    Zy = np.zeros((Ny, Nx))
    return np.block([[X, Zx],
                     [Zy, Y]])

def schur_complements(M, A_size):
    A = M[:A_size, :A_size]
    B = M[:A_size, A_size:]
    C = M[A_size:, :A_size]
    D = M[A_size:, A_size:]
    try:
        A_inv = np.linalg.inv(A)
        S2 = D - C @ A_inv @ B
    except np.linalg.LinAlgError:
        S2 = None
    try:
        D_inv = np.linalg.inv(D)
        S1 = A - B @ D_inv @ C
    except np.linalg.LinAlgError:
        S1 = None
    return S1, S2

# Cumulant functions and random variable generations

def discretize(comp_mat, k):
    return np.round((comp_mat + 1) / 2 * (k - 1)) / (k - 1) * 2 - 1

def phi_bin(theta):
    return np.log(np.cosh(theta))

def phi_kna(theta, k):
    val = theta / (k - 1)
    return np.log(np.sinh(k * val) / (k * np.sinh(val)))

def phi_poi(theta, lam):
    return lam * np.cosh(theta)

def phi_gau(theta, sigma = 1):
    return sigma**2 * theta**2

def phi_uni(theta):
    max_theta = 500
    theta = np.clip(theta, -max_theta, max_theta)
    return np.log(np.sinh(abs(theta)) / (abs(theta) + 1e-6))

def phi_uni_prime(theta):
    eps = 1e-6
    safe_theta = np.where(np.abs(theta) < eps, eps, theta)
    return np.where(np.abs(theta) < eps, safe_theta / 3, 1 / np.tanh(safe_theta) - 1 / safe_theta)

def phi_bin_prime(theta):
    return np.tanh(theta)

def phi_tri_prime(theta, p):
    return ((1 - p) * np.sinh(theta)) / (p + (1 - p) * np.cosh(theta))

def phi_gau_prime(theta, sigma=1.0):
    return theta / sigma ** 2

def generate_bin(theta_diff):
    k_values = np.linspace(-1, 1, 2)
    weights = np.exp(k_values * theta_diff)
    probabilities = weights / np.sum(weights)  # Normalisation
    return np.random.choice(k_values, p=probabilities)

def generate_knary(theta_diff, K):
    k_values = np.linspace(-1, 1, K)
    weights = np.exp(k_values * theta_diff)
    probabilities = weights / np.sum(weights)  # Normalisation
    return np.random.choice(k_values, p=probabilities)

def generate_poisson(lmbda, theta_diff): #Not sure at all
    k_pos = poisson.rvs(lmbda, 1)
    signs = np.random.choice([-1, 1])

    acceptance_probs = np.exp(k_pos * theta_diff)
    acceptance_probs /= np.max(acceptance_probs)
    accepted = np.random.uniform(0, 1) < acceptance_probs

    k_final = signs * k_pos
    return k_final[accepted] if len(k_final[accepted]) > 0 else generate_poisson(lmbda, theta_diff)

def generate_gaussian(theta_diff, sigma):
        return np.random.normal(loc=sigma * theta_diff, scale=sigma ** 2)

def generate_truncated_exp(theta_diff):
    if np.abs(theta_diff) < 1e-6:
        return np.random.uniform(-1, 1)
    else:
        sample = 3
        param = np.abs(theta_diff)
        while sample > 2:
            sample = np.random.exponential(1 / param)
        return -(sample - 1) * theta_diff / param

def error_metric(theta_star, theta_true):
    theta_star = np.asarray(theta_star).flatten()
    theta_true = np.asarray(theta_true).flatten()

    return np.linalg.norm(theta_true - theta_star)**2 / np.linalg.norm(theta_true)**2

def error_metric_shift_invariant(theta_star, theta_true):
    u = np.mean(theta_star - theta_true)
    theta_star_shifted = theta_star - u
    return np.linalg.norm(theta_true - theta_star_shifted)**2 / np.linalg.norm(theta_true)**2

# ------------------------ Data Generation ------------------------#

def generate_r_power(C, theta_true, eta, phi):
    A = theta_true.shape
    r = np.zeros((A, A))

    for a in range(A):
        for b in range(A):
            if a != b:
                if phi == phi_uni:
                    r[a, b] = generate_truncated_exp(theta_true[a] - theta_true[b])
                elif phi == phi_gau:
                    r[a, b] = generate_gaussian(theta_true[a] - theta_true[b], sigma =1)
                elif phi == phi_poi:
                    r[a, b] = generate_poisson(theta_true[a] - theta_true[b], lmbda = 1)
                elif phi == phi_kna:
                    r[a, b] = generate_knary(theta_true[a] - theta_true[b], K = 11)

    r_c, _ = choose_pairs(r, C, eta)

    return r_c, r

def choose_pairs(r, C, eta):
    A = r.shape[0]

    pairs = [(a, b) for a in range(A) for b in range(A) if a != b]
    weights = np.array([1 / ((a + 1) ** eta * (b + 1) ** eta) for a, b in pairs])
    probabilities = weights / np.sum(weights)

    chosen_indices = np.random.choice(len(pairs), size=C, p=probabilities)
    chosen_pairs = [pairs[i] for i in chosen_indices]

    r_c = np.full((A, A), np.nan)
    for a, b in chosen_pairs:
        r_c[a, b] = r[a, b]

    return r_c, chosen_pairs

def generate_data_gbt(x, Sigma_beta, C, L, phi):
    D, A = x.shape
    beta_true = np.random.multivariate_normal(mean=np.zeros(D), cov=Sigma_beta + x @ L @ x.T)
    theta_true = x.T @ beta_true
    r = np.zeros((A, A))
    for a in range(A):
        for b in range(A):
            if a != b:
                if phi == phi_uni:
                    r[a, b] = generate_truncated_exp(theta_true[a] - theta_true[b])
                elif phi == phi_gau:
                    r[a, b] = generate_gaussian(theta_true[a] - theta_true[b], sigma =1)
                elif phi == phi_poi:
                    r[a, b] = generate_poisson(theta_true[a] - theta_true[b], lmbda = 1)
                elif phi == phi_kna:
                    r[a, b] = generate_knary(theta_true[a] - theta_true[b], K = 11)
    r_c, _ = choose_pairs(r, C, 0)
    return r_c, r, beta_true, theta_true

def generate_data_gbt_eta(x, Sigma_beta, C, eta, phi):
    D, A = x.shape
    beta_true = np.random.multivariate_normal(mean=np.zeros(D), cov=Sigma_beta)
    theta_true = x.T @ beta_true
    r = np.zeros((A, A))
    for a in range(A):
        for b in range(A):
            if a != b:
                if phi == phi_uni:
                    r[a, b] = generate_truncated_exp(theta_true[a] - theta_true[b])
                elif phi == phi_gau:
                    r[a, b] = generate_gaussian(theta_true[a] - theta_true[b], sigma =1)
                elif phi == phi_poi:
                    r[a, b] = generate_poisson(theta_true[a] - theta_true[b], lmbda = 1)
                elif phi == phi_kna:
                    r[a, b] = generate_knary(theta_true[a] - theta_true[b], K = 11)
    r_c = choose_pairs(r, C, eta)
    return r_c, r, beta_true, theta_true

def loss_emb_gbt(beta, r, x, Sigma_beta, L, phi):
    beta = np.asarray(beta)
    theta = x.T @ beta
    reg = 0.5 * beta @ (Sigma_beta + x @ L @ x.T) @ beta

    if isinstance(r, tuple):
        r = r[0]
    a, b = np.where(~np.isnan(r))
    valid_indices = b > a
    a, b = a[valid_indices], b[valid_indices]
    r_ab = r[a, b]
    theta_ab = theta[a] - theta[b]
    fit = np.sum(phi(theta_ab) - r_ab * theta_ab)
    return reg + fit

def compute_scores(r, x, Sigma_beta, L, phi):
    #r = np.nan_to_num(r, nan=0.0)
    D, A = x.shape
    beta_star = np.random.normal(0, 1, D)
    args = (r, x, Sigma_beta, L, phi)
    res = minimize(loss_emb_gbt, beta_star, args=args, method="L-BFGS-B")
    beta_star = res.x
    theta_star = x.T @ beta_star
    return beta_star, theta_star

#Experiments

def compute_empiric_MSE(x_true, x_data, Sigma_true, Sigma_data, p_c, phi_true, phi_data, nb_seeds):
    errors = []
    errors_shift = []
    for seed in range(nb_seeds):
        r_c, r, theta_true = generate_data_gbt(x_true, Sigma_true, p_c, phi_true)
        beta_star, theta_star = compute_scores(r_c, x_data, Sigma_data, phi_data)
        error = error_metric(theta_star, theta_true)
        errors.append(error)
        error_shift = error_metric_shift_invariant(theta_star, theta_true)
        errors_shift.append(error_shift)
    return np.mean(errors), np.mean(errors_shift)
