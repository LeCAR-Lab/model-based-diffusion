from jax import numpy as jnp

def get_A(x: jnp.ndarray, env_params: EnvParams) -> jnp.ndarray:
    return jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ) * env_params.dt + jnp.eye(n_state)


def get_B(x: jnp.ndarray, env_params: EnvParams) -> jnp.ndarray:
    return (
        jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        * env_params.dt
    )
