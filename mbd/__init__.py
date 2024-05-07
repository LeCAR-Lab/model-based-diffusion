from brax import envs as brax_envs

from mbd import envs, planners


def get_env(env_name: str):
    if env_name == "pushT":
        return envs.PushT()
    elif env_name in ["ant", "halfcheetah"]: 
        return brax_envs.get_environment(env_name=env_name, backend="positional")
    elif env_name == "hopper":
        return envs.Hopper()
    else:
        raise ValueError(f"Unknown environment: {env_name}")