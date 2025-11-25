import numpy as np
import matplotlib.pyplot as plt
import Functions


def experiment(A_max, D_max, nb_seed, nb_iter, alpha):

    A_values = np.array(range(2, A_max + 1))
    D_values = np.array(range(1, D_max + 1))
    total_A = A_values.shape[0]
    total_D = D_values.shape[0]
    Probs_metagood = np.zeros((total_A, total_D))

    for A in A_values:
        for D in D_values:
            count_metagood = 0
            for seed in range(nb_seed):
                #Sigma_x = Functions.generate_X(D, D, True)
                Sigma_x = np.eye(D)
                x = np.random.multivariate_normal(mean=np.zeros(D), cov=Sigma_x, size=A).T
                y = np.eye(A)
                X = x.T @ x + alpha * y.T @ y
                is_X_metagood, _, _ = Functions.is_metagood(X, nb_iter, False)
                if is_X_metagood == True:
                    count_metagood += 1

            Probs_metagood[A-2, D-1] = count_metagood / nb_seed

    return Probs_metagood

def build_plot(Probs, figname):
    # Create a table to visualize the values of the Probs matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(Probs, cmap='viridis', vmin=0, vmax=1)  # Set vmin and vmax
    plt.colorbar(cax)  # Add a colorbar to indicate the scale

    # Set the ticks and labels
    ax.set_xticks(np.arange(D_max))  # Ticks for D (features)
    ax.set_yticks(np.arange(A_max))  # Ticks for A (alternatives)
    ax.set_xticklabels(np.arange(1, D_max + 1))  # Labels for D
    ax.set_yticklabels(np.arange(2, A_max + 2))  # Labels for A

    # Add gridlines
    plt.grid(False)
    plt.xlabel("D")
    plt.ylabel("A")
    plt.title("Probability of Goodness for Different A and D")

    # Invert the y-axis to have the first row at the top
    ax.invert_yaxis()

    # Adjust the y-axis limits to avoid the white band
    ax.set_ylim(A_max - 1.5, -0.5)  # Set limits to show from 2 to A_max

    plt.savefig(figname)
    return

A_max = 15
D_max = 15
nb_seed = 100
nb_iter = 100

## Experiment 1
alpha=0
probs = experiment(A_max, D_max, nb_seed, nb_iter, alpha)
build_plot(probs, 'density-a.pdf')

## Experiment 2
alpha=1/2
probs = experiment(A_max, D_max, nb_seed, nb_iter, alpha)
build_plot(probs, 'density-b.pdf')
