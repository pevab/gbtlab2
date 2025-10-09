import numpy as np
from matplotlib import pyplot as plt
import Functions


def experiment(A, D, nb_seeds, C_values, L, phi, s):
    errors = []
    stds = []

    for C in C_values:
        errors_for_C = {'uni': []}
        print("C = ", C)

        for seed in range(nb_seeds):
            np.random.seed(seed)
            x_alpha = s * np.eye(A)
            x_beta = Functions.generate_x_publisher(A, D)
            x_full = np.vstack((x_alpha, x_beta))

            # Générer les données
            r_c, r, beta_true, theta_true = Functions.generate_data_gbt(x_full, np.eye(A + D), C, L, phi)

            # Calcul des scores pour phi
            _, theta_star_GBTE = Functions.compute_scores(r_c, x_full, np.eye(A + D), L, phi)
            _, theta_star_GBT = Functions.compute_scores(r_c, x_alpha, np.eye(A), L, phi)

            # Calcul des erreurs pour phi
            error_GBTE = Functions.error_metric(theta_star_GBTE, theta_true)
            error_GBT = Functions.error_metric(theta_star_GBT, theta_true)
            errors_for_C['uni'].append((error_GBTE, error_GBT))

        # Moyenne et écart-type des erreurs pour chaque p_c
        mean_errors = np.mean(errors_for_C['uni'], axis=0)
        std_errors = np.std(errors_for_C['uni'], axis=0)

        errors.append(mean_errors)
        stds.append(std_errors)

    # Convertir les erreurs en array pour faciliter les plots
    errors = np.array(errors)
    stds = np.array(stds)

    return errors, stds

A = 20
D = 10
nb_seeds = 500
C_values = range(20, 320, 40)
s = 1/2
L = np.zeros((A, A))
phi = Functions.phi_uni

errors, std_errors = experiment(A, D, nb_seeds, C_values, L, phi, s)

# Tracer les erreurs en fonction de p_c
plt.figure(figsize=(8, 5))

# Calcul des erreurs pour phi
errors_GBTE = errors[:, 0]
errors_GBT = errors[:, 1]

# Calcul des écarts-types pour phi
std_GBTE = std_errors[:, 0]
std_GBT = std_errors[:, 1]

# Plot des erreurs pour phi
plt.plot(C_values, errors_GBTE, label="GBT & encoding", marker='o', color='blue')
plt.plot(C_values, errors_GBT, label="GBT", marker='s', color='orange')

plt.xlabel("N")
plt.ylabel("nMSE")
plt.legend()
#plt.title("nMSE as a function of the number of comparisons C")
plt.savefig('rMSE-GBT-onehot.png')
