from brax import envs as brax_envs

from .pushT import PushT
from .hopper import Hopper
from .humanoidstandup import HumanoidStandup
from .humanoidtrack import HumanoidTrack
from .humanoidrun import HumanoidRun
from .walker2d import Walker2d
from .cartpole import Cartpole
from .car2d import Car2d

def get_env(env_name: str):
    if env_name == "pushT":
        return PushT()
    elif env_name == "hopper":
        return Hopper()
    elif env_name == "humanoidstandup":
        return HumanoidStandup()
    elif env_name == "humanoidrun":
        return HumanoidRun()
    elif env_name == "humanoidtrack":
        return HumanoidTrack()
    elif env_name == "walker2d":
        return Walker2d()
    elif env_name == "cartpole":
        return Cartpole()
    elif env_name == "car2d":
        return Car2d()
    elif env_name in ["ant", "halfcheetah"]:
        return brax_envs.get_environment(env_name=env_name, backend="positional")
    else:
        raise ValueError(f"Unknown environment: {env_name}")