from brax import envs as brax_envs

from mbd.envs.pushT import PushT
from mbd.envs.hopper import Hopper
from mbd.envs.humanoidstandup import HumanoidStandup
from mbd.envs.humanoidrun import HumanoidRun
from mbd.envs.walker2d import Walker2d

def get_env(env_name: str):
    if env_name == "pushT":
        return PushT()
    elif env_name == "hopper":
        return Hopper()
    elif env_name == "humanoidstandup":
        return HumanoidStandup()
    elif env_name == "humanoidrun":
        return HumanoidRun()
    elif env_name == "walker2d":
        return Walker2d()
    elif env_name in ["ant", "halfcheetah"]: 
        return brax_envs.get_environment(env_name=env_name, backend="positional")
    else:
        raise ValueError(f"Unknown environment: {env_name}")