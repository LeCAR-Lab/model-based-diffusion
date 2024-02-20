def test_kalman_filter():
    # generate feasible initial trajectory
    x_traj_real = jnp.zeros((env_params.horizon, env_params.n_state))
    u_traj_real = jax.random.normal(rng_u, (env_params.horizon, env_params.n_action))
    x_traj_real = x_traj_real.at[0].set(env_params.init_state)
    for t in range(1, env_params.horizon):
        x_traj_real = x_traj_real.at[t].set(
            get_A(x_traj_real[t-1], env_params) @ x_traj_real[t-1]
            + get_B(x_traj_real[t-1], env_params) @ u_traj_real[t-1]
        )
    # plot the trajectory
    plot_traj(x_traj_real, "init_traj")

    # add noise to the initial trajectory
    x_traj = x_traj_real + jax.random.normal(rng_x, x_traj_real.shape) * 1.0
    x_traj = x_traj.at[0].set(env_params.init_state)
    u_traj = u_traj_real + jax.random.normal(rng_u, u_traj_real.shape) * 1.0
    # use kalman filter to estimate the initial state
    mdb_params = mdb_params.replace(
        noise_std=1.0,
    )
    logp_dynamics, x_traj_filtered = get_logp_dynamics(x_traj, u_traj, mdb_params, env_params)
    # plot the trajectory
    plot_traj(x_traj_filtered, "init_traj_filtered")
    plot_traj(x_traj, "init_traj_noisy")
    jax.debug.print('logp_dynamics = {x}', x=logp_dynamics)
    jax.debug.print('grad logp_dynamics = {x}', x=jax.grad(get_logp_dynamics_scan, argnums=[0])(x_traj, u_traj, mdb_params, env_params))