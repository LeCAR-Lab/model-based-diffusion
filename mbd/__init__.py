from mbd import envs, planners

def get_env(env_name: str):
    if env_name == "pushT":
        return envs.PushT()
    else:
        raise ValueError(f"Unknown environment: {env_name}")