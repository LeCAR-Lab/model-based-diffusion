import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import random
from matplotlib import pyplot as plt

rng = random.PRNGKey(0)

Ntar = 2
Y0s_tar = jnp.array([-0.5, 0.5])
log_weights_tar = jnp.array([0.0, 0.0])
sigma_tar = 0.2

Nsample = 128
Y0s_hat = random.normal(rng, (Nsample,))
log_weights_hat = 1 / Nsample * jnp.ones(Nsample)
sigma = sigma_tar*jnp.sqrt(Ntar / Nsample) 

def sample_GMM(means, log_weights, sigma, num_samples, key):
    components = random.categorical(key, log_weights, shape=(num_samples,))
    samples = means[components] + sigma * random.normal(key, (num_samples,))
    return samples

def get_logp_GMM(means, log_weights, sigma, x):
    log_probs = jnp.log(jax.nn.softmax(log_weights))
    log_probs = log_probs + norm.logpdf(x, means, sigma)
    return jax.scipy.special.logsumexp(log_probs)

get_logp_GMM_vmap = jax.vmap(get_logp_GMM, in_axes=(None, None, None, 0))

def update_once(Y0s_hat, logp_Y0s_hat, key):
    key, subkey = random.split(key)
    Y0s_hat = sample_GMM(Y0s_hat, logp_Y0s_hat, sigma, Nsample, subkey)
    logp_Y0s_hat = get_logp_GMM_vmap(Y0s_tar, log_weights_tar, sigma_tar, Y0s_hat) - get_logp_GMM_vmap(Y0s_hat, logp_Y0s_hat, sigma, Y0s_hat)
    logp_Y0s_hat = logp_Y0s_hat - jnp.max(logp_Y0s_hat)
    return Y0s_hat, logp_Y0s_hat

def plot_gaussian_mixture(Y0s_hat, logp_Y0s_hat, sigma):
    x_values = jnp.linspace(-1.5, 1.5, 100)
    weights = jax.nn.softmax(logp_Y0s_hat)

    # Compute the mixture density
    mixture_density = jnp.zeros_like(x_values)
    for mean, weight in zip(Y0s_hat, weights):
        mixture_density += weight * norm.pdf(x_values, mean, sigma)
    # normalize
    scale = 1 / jnp.sum(mixture_density) / (x_values[1] - x_values[0])
    mixture_density = mixture_density * scale

    # Target distribution: Gaussian mixture with two components
    target_density = 0.5 * norm.pdf(x_values, -0.5, sigma_tar) + 0.5 * norm.pdf(x_values, 0.5, sigma_tar)

    plt.cla()

    # Plot each component
    for mean, weight in zip(Y0s_hat[:16], weights[:16]):
        plt.plot(x_values, norm.pdf(x_values, mean, sigma) * weight * scale)

    # Plot the mixture density
    plt.plot(x_values, mixture_density, 'k--', linewidth=2, label='Estimated Mixture')
    plt.plot(x_values, target_density, 'r:', linewidth=2, label='Target Distribution')
    plt.fill_between(x_values, 0, mixture_density, color='gray', alpha=0.3)
    plt.fill_between(x_values, 0, target_density, color='red', alpha=0.2)
    plt.title('Gaussian Mixture Approximation vs Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 2.0)
    plt.pause(0.3)

for i in range(10):
    rng, key = random.split(rng)
    Y0s_hat, logp_Y0s_hat = update_once(Y0s_hat, log_weights_hat, key)
    plot_gaussian_mixture(Y0s_hat, logp_Y0s_hat, sigma)
    # plt.savefig(f"../figure/{i}.png")