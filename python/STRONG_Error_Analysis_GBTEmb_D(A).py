import numpy as np
import matplotlib.pyplot as plt
import Functions


def experiment(A, D_min, D_max, nb_seeds, N, L, phi):
    D_values = np.arange(D_min, D_max + 1)

    errors = []
    std_errors = []

    for D in D_values:
        errors_for_D = {'uni': []}
        k = 0
        print("D = ", D)
        for seed in range(nb_seeds):
            k += 1
            # print("D, k =", [int(D), k])

            x_D = np.random.randn(D, A)
            x_full = np.vstack((np.eye(A), x_D))

            r_c, r, beta_true, theta_true = Functions.generate_data_gbt(x_full, np.eye(A + D), N, L, phi)

            _, theta_star_GBTE = Functions.compute_scores(r_c, x_full, np.eye(A + D), L, phi)
            _, theta_star_GBT = Functions.compute_scores(r_c, np.eye(A), np.eye(A), L, phi)
            _, theta_star_Emb = Functions.compute_scores(r_c, x_D, np.eye(D), L, phi)

            error_GBTE = Functions.error_metric(theta_star_GBTE, theta_true)
            error_GBT = Functions.error_metric(theta_star_GBT, theta_true)
            error_Emb = Functions.error_metric(theta_star_Emb, theta_true)
            errors_for_D['uni'].append((error_GBTE, error_GBT, error_Emb))

        mean_errors = np.mean(errors_for_D['uni'], axis=0)
        std_errors_for_D = np.std(errors_for_D['uni'], axis=0)

        errors.append(mean_errors)
        std_errors.append(std_errors_for_D)

    errors = np.array(errors)
    std_errors = np.array(std_errors)

    return D_values, errors, std_errors


A = 25
D_min = 1
D_max = A
nb_seeds = 1000
N = 500
L = np.zeros((A, A))
phi = Functions.phi_uni

D_values, errors, std_errors = experiment(A, D_min, D_max, nb_seeds, N, L, phi)

plt.figure(figsize=(8, 5))

errors_GBTE = errors[:, 0]
errors_GBT = errors[:, 1]
errors_Emb = errors[:, 2]

std_errors_GBTE = std_errors[:, 0]
std_errors_GBT = std_errors[:, 1]
std_errors_Emb = std_errors[:, 2]

# Plot the curves
plt.plot(D_values, errors_GBTE, label="GBT & encoding", marker='o', color='blue')
plt.plot(D_values, errors_GBT, label="GBT", marker='s', color='orange')
plt.plot(D_values, errors_Emb, label="Encoding", marker='^', color='green')

plt.xlabel("D")
plt.ylabel("nMSE")
plt.legend()
# plt.title("rMSE of GBT models for various D")
plt.savefig('rMSE-GBT.png')
