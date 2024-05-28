import jax
from jax import numpy as jnp
from tqdm import tqdm
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
import numpy as np
import os, struct, array, gzip
import urllib.request
from os import path

# global variables
rng = jax.random.PRNGKey(0)
Nsample = 512
Nsample = 256
Ndiffuse = 500
temp_sample = 0.3
betas = jnp.linspace(3e-5, 1e-3, Ndiffuse)
alphas = 1.0 - betas
alphas_bar = jnp.cumprod(alphas)
sigmas = jnp.sqrt(1 - alphas_bar)
batch_size = 128

# download data
_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


# load dataset
train_images, train_labels, test_images, test_labels = mnist()
train_images = jnp.array(train_images)
train_labels = jnp.array(train_labels)
Ndata = train_images.shape[0]

# create network
layer_size = jnp.array([28**2, 32, 32, 10])
weight_size = jnp.array(
    [layer_size[i] * layer_size[i + 1] for i in range(len(layer_size) - 1)]
)
bias_size = layer_size[1:]
init_random_params, predict = stax.serial(
    Dense(32), Relu, Dense(32), Relu, Dense(10), LogSoftmax
)
_, params = init_random_params(rng, (-1, 28 * 28))

params_shape = [[p.shape for p in layer] for layer in params]
params_batch_shape = []
for layer in params_shape:
    layer_shape = []
    for param in layer:
        layer_shape.append((Nsample,) + param)
    params_batch_shape.append(layer_shape)


def add_noise_batch_to_params(params, sigma, rng):
    noisy_params_batch = []

    # Iterate over each layer's parameters (weights and biases) and their shapes
    for i, (param_layer, shape_layer) in enumerate(zip(params, params_batch_shape)):
        noisy_layer = []
        for j, (param, shape) in enumerate(zip(param_layer, shape_layer)):
            # Generate Gaussian noise based on the cached shape
            rng, rng_noise = jax.random.split(rng)
            noise = jax.random.normal(rng_noise, shape) * sigma
            if i == 0 and j == 0:
                noise = noise * 0.1  # NOTE: first layer is too large, limit the noise
            rng, rng_update = jax.random.split(rng)
            # a simple implementation of Gibbs sampling
            update_weight = jax.random.bernoulli(rng_update, 0.2, shape)
            noisy_param = param + noise * update_weight
            noisy_layer.append(noisy_param)
        noisy_params_batch.append(noisy_layer)

    return noisy_params_batch


def get_params_batch_weighted_sum(params_batch, weights):
    new_params = []

    for layer in params_batch:
        new_layer = []
        for param in layer:
            new_param = jnp.tensordot(weights, param, axes=[0, 0])
            new_layer.append(new_param)
        new_params.append(tuple(new_layer))

    return new_params


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


def eval_fn(params):
    return loss(params, (train_images, train_labels))


def add_noise_to_params(params, sigma, rng):
    noisy_params = []

    # Iterate over each layer's parameters (weights and biases)
    for param_layer in params:
        noisy_layer = []
        for param in param_layer:
            # Add Gaussian noise to the parameter
            noise = jax.random.normal(rng, param.shape) * sigma
            noisy_param = param + noise
            noisy_layer.append(noisy_param)
            # Update the RNG key
            rng, _ = jax.random.split(rng)
        noisy_params.append(tuple(noisy_layer))

    return noisy_params


@jax.jit
def reverse_once(carry, unused):
    t, rng, Y0 = carry

    # sample from q_i
    rng, Y0_rng = jax.random.split(rng)
    Y0s = add_noise_batch_to_params(Y0, sigmas[t], Y0_rng)

    # esitimate mu_0tm1
    rng, batch_rng = jax.random.split(rng)
    batch_idx = jax.random.choice(batch_rng, Ndata, (Nsample,), replace=False)
    train_images_batch = train_images[batch_idx]
    train_labels_batch = train_labels[batch_idx]
    l = jax.vmap(loss, in_axes=(0, None))(Y0s, (train_images_batch, train_labels_batch))
    Js = -l
    # Js = -jax.vmap(eval_fn)(Y0s)
    logp0 = (Js - Js.mean()) / Js.std() / temp_sample
    weights = jax.nn.softmax(logp0)
    Y0 = get_params_batch_weighted_sum(Y0s, weights)

    return (t - 1, rng, Y0), Js.mean()


Y0 = params
_, _ = reverse_once((0, rng, Y0), None)  # to compile
Yt = Y0
with tqdm(range(Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
    for t in pbar:
        carry_once = (t, rng, Yt)
        (t, rng, Yt), J = reverse_once(carry_once, None)
        train_acc = accuracy(Yt, (train_images, train_labels))
        test_acc = accuracy(Yt, (test_images, test_labels))
        pbar.set_postfix(
            J=f"{J:.2f}", train_acc=f"{train_acc:.3f}", test_acc=f"{test_acc:.3f}"
        )
