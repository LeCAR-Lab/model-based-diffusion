def get_logp_dynamics(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    mdb_params: MBDParams,
    env_params: EnvParams,
) -> jnp.ndarray:
    # tau: (n_step, n_dim)
    x_hat = x_traj[0]
    cov_hat = jnp.eye(6) * 0.0
    logp_dynamics = 0.0
    for t in range(1, 50):
        # parse state and action
        u_prev = u_traj[t - 1]
        # get state prediction
        A = get_A(x_hat, env_params)  # NOTE: A is discrete time
        B = get_B(x_hat, env_params)
        Q = B @ B.T * mdb_params.noise_std**2
        R = jnp.eye(6) * mdb_params.noise_std**2
        x_pred = A @ x_hat + B @ u_prev
        # get prediction covariance
        cov_pred = A @ cov_hat @ A.T + Q
        # get optimal prediction feedback matrix
        K = cov_pred @ jnp.linalg.inv(cov_pred + R)
        # update state and covariance
        x_hat = x_pred + K @ (x_traj[t] - x_pred)
        cov_hat = (jnp.eye(6) - K) @ cov_pred
        # update logp_dynamics
        y_cov_pred = (
            cov_pred + R
        )  # NOTE: y_cov_pred is the covariance of the observation
        logp_dynamics += -0.5 * (
            jnp.log(2 * jnp.pi) * 6
            + jnp.linalg.slogdet(y_cov_pred)[1]
            + (x_traj[t] - x_pred).T @ jnp.linalg.inv(y_cov_pred) @ (x_traj[t] - x_pred)
        )
    return logp_dynamics