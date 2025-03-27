# %% [markdown]
# # Reverse SDE Simulation
# This script simulates a reverse stochastic differential equation (SDE) process and visualizes:
# 1. An objective function
# 2. Forward diffusion process
# 3. Backward (reverse) SDE process
#
# The visualization compares a reverse SDE approach with a Monte Carlo score ascent method.

# %% [markdown]
# ## Setup and Imports

# %%
# Import required libraries
from matplotlib import pyplot as plt
import numpy as np
import scienceplots

# Apply scientific plot style
plt.style.use("science")

# %% [markdown]
# ## Define Objective Function


# %%
def objective_function(x):
    """
    Define the objective function J(x) with multiple local minima.

    Args:
        x: Input values

    Returns:
        Objective function values
    """
    # Main objective component (double-well potential)
    y = 1.0 * ((x**4 - 0.5) ** 2)

    # Add sinusoidal perturbation for multiple local minima
    y = y + np.sin(2 * np.pi * 10.0 * x) * 0.04

    return y


# %% [markdown]
# ## Configure Diffusion Parameters

# %%
# Set up diffusion process parameters
num_timesteps = 200  # Total number of diffusion steps
num_samples = 10  # Number of samples for Monte Carlo estimation

# Define noise schedule (decreasing alphas)
alphas = 1.0 - np.linspace(0.0001, 0.01, num_timesteps + 1)

# Calculate cumulative products of alphas
alphas_bar = np.cumprod(alphas)

# Calculate variance schedule
sigmas_squared = 1 - alphas_bar
sigmas = np.sqrt(sigmas_squared)

print(f"Final alpha_bar: {alphas_bar[-1]}")

# Define color schemes for different plots
forward_colors = plt.cm.Reds(np.linspace(1.0, 0.1, num_timesteps + 1))
backward_colors = plt.cm.Blues(np.linspace(1.0, 0.1, num_timesteps))
density_colors = plt.cm.Reds(np.linspace(1.0, 0.3, 5))

# %% [markdown]
# ## Create Figure Layout

# %%
# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# %% [markdown]
# ## Generate Input Space and Initial Distribution

# %%
# Define input space
x_values = np.linspace(-2.0, 2.0, 1000)

# Calculate initial distribution (proportional to exp(-J/lambda))
# where lambda controls the temperature
temperature = 1 / 20.0  # lambda = 0.05
p0 = np.exp(-objective_function(x_values) / temperature)
p0 = p0 / np.sum(p0)  # Normalize to make a proper probability distribution

# Initialize lists to store distributions and means
distributions = [p0]
means = [np.sum(x_values * p0)]

# Initial point mass distribution (for visualization purposes)
mu0_idx = np.argmin(np.abs(x_values - means[0]))
q0 = np.zeros(len(x_values))
q0[mu0_idx] = 0.04
point_distributions = [q0]

# %% [markdown]
# ## Plot Objective Function (First Subplot)

# %%
# Plot the objective function
axes[0].plot(x_values, objective_function(x_values), label="$J(Y)$", color="black")
axes[0].set_xlim(-1, 1)
axes[0].set_ylim(-0.05, 0.3)
axes[0].set_ylabel("Objective", fontsize=16)
axes[0].set_xlabel("Y", fontsize=16)
axes[0].set_title("(a) Objective Function $J(Y)$", fontsize=16)

# %% [markdown]
# ## Simulate Forward Diffusion Process (Second Subplot)

# %%
# Plot initial distribution
axes[1].plot(x_values, p0, "--", label=r"$p_0=e^{-\frac{J}{\lambda}}$", color="black")

# Simulate forward diffusion process
for i in range(num_timesteps):
    current_dist = distributions[-1]

    # Scale distribution according to current noise level
    scaled_x = x_values * np.sqrt(alphas[i])

    # Resample with linear interpolation
    rescaled_dist = np.interp(x_values, scaled_x, current_dist)

    # Convolve with Gaussian noise
    noise_variance = 1 - alphas[i]
    gaussian_kernel = np.exp(-0.5 * x_values**2 / noise_variance)

    # Apply convolution to simulate noise addition
    next_dist = np.convolve(rescaled_dist, gaussian_kernel, mode="same")
    next_dist = next_dist / np.sum(next_dist)  # Normalize

    # Calculate mean of the distribution
    mean = np.sum(x_values * next_dist)
    means.append(mean)

    # Store the distribution
    distributions.append(next_dist)

    # Plot selected timesteps to show progression
    if i in [1, num_timesteps // 10, num_timesteps // 2, num_timesteps - 1]:
        k = [1, num_timesteps // 10, num_timesteps // 2, num_timesteps - 1].index(i)
        axes[1].plot(
            x_values,
            next_dist,
            color=density_colors[k],
            label=f"$p_{{{i}}}$",
            alpha=0.75,
            linewidth=2.0,
        )

# Configure second subplot
axes[1].set_ylabel("Density", fontsize=16)
axes[1].set_xlabel("Y", fontsize=16)
axes[1].set_title("(b) Forward Density", fontsize=16)
axes[1].set_xlim(-1.4, 1.4)
axes[1].legend(fontsize=14)

# %% [markdown]
# ## Simulate Reverse SDE and Score Ascent Processes (Third Subplot)

# %%
# Convert list of distributions to numpy array for heatmap
dist_array = np.array(distributions)

# Plot reverse trajectories
ax = axes[2]

# Run multiple simulations of reverse process
for i in range(8):
    # Initialize starting point (pure noise)
    current_x = 0.0
    idx_x = np.argmin(np.abs(x_values - current_x))
    sde_trajectory = [current_x]

    # Initialize Monte Carlo trajectory
    mc_mean = 0.0
    mc_trajectory = [mc_mean]

    # Simulate reverse diffusion from t=T to t=1
    for t in range(num_timesteps, 0, -1):
        # Monte Carlo score estimation
        # Calculate energy landscape
        energy = (
            objective_function(x_values)
            + 1 / sigmas_squared[t] * (x_values - mc_mean) ** 2
        )

        # Generate samples around current mean
        samples = np.clip(
            np.random.normal(mc_mean, sigmas[t], num_samples), -0.99, 0.99
        )
        sample_indices = np.argmin(np.abs(x_values[:, None] - samples[None, :]), axis=0)

        # Weight samples by objective function
        weights = np.exp(-objective_function(x_values[sample_indices]) / temperature)
        weights = weights / np.sum(weights)

        # Update mean estimate
        mc_mean = np.sum(samples * weights)
        mc_trajectory.append(mc_mean * np.sqrt(alphas_bar[t]))

        # Reverse SDE update
        # Estimate score (gradient of log density)
        grad_pt = (dist_array[t, idx_x + 1] - dist_array[t, idx_x - 1]) / (
            x_values[idx_x + 1] - x_values[idx_x - 1]
        )
        score = grad_pt / dist_array[t, idx_x]

        # Update using reverse SDE formula
        drift_term = (
            1 / np.sqrt(alphas[t]) * (current_x + 0.5 * (1 - alphas[t]) * score)
        )
        diffusion_term = np.sqrt(1 - alphas[t - 1]) * np.random.randn()

        # Update x and clip to valid range
        next_x = drift_term + diffusion_term
        current_x = np.clip(next_x, -2, 2)

        # Find closest index in discretized space
        idx_x = np.clip(np.argmin(np.abs(x_values - current_x)), 1, len(x_values) - 2)

        # Store trajectory
        sde_trajectory.append(current_x)

    # Plot trajectories with different colors for first iteration
    if i == 0:
        ax.plot(
            np.arange(num_timesteps + 1, 0, -1) - 0.5,
            sde_trajectory,
            color="red",
            alpha=0.5,
            label="Reverse SDE",
        )
        ax.plot(
            np.arange(num_timesteps + 1, 0, -1) - 0.5,
            mc_trajectory,
            color="blue",
            alpha=0.5,
            label="MC Score Ascent",
        )
    else:
        # Plot additional trajectories without labels
        ax.plot(
            np.arange(num_timesteps + 1, 0, -1) - 0.5,
            sde_trajectory,
            color="red",
            alpha=0.5,
        )
        ax.plot(
            np.arange(num_timesteps + 1, 0, -1) - 0.5,
            mc_trajectory,
            color="blue",
            alpha=0.5,
        )

# %% [markdown]
# ## Create Background Heatmap and Finalize Plot

# %%
# Create heatmap of log density over time
ax.imshow(
    np.log(dist_array.T[::-1] + 0.001),  # Add small constant to avoid log(0)
    aspect="auto",
    cmap="rainbow",
    extent=[0, num_timesteps, -2, 2],
    alpha=0.4,
)

# Add colorbar
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, orientation="vertical"
)

# Configure third subplot
ax.set_xlabel("Diffusion Step", fontsize=16)
ax.set_ylabel("Y", fontsize=16)
ax.set_title("(c) Backward Process", fontsize=16)
ax.legend(fontsize=12)

# %% [markdown]
# ## Show Final Plot

# %%
# Adjust layout and display plot
plt.tight_layout()
# plt.savefig("reverse_sde_visualization.pdf", bbox_inches="tight")
plt.show()
