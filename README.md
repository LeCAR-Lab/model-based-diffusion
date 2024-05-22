# Model-Based Diffusion

<div align="center">

[[Website]](https://model-based-diffusion.github.io/)
[[PDF(Coming Soon)]]()
[[Arxiv(Coming Soon)]]()

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- insert figure -->
<img src="pics/joint.gif" width="600px"/>

</div>

This repository contains the code for the paper "Model-based Diffusion for Trajectory Optimization".

## Installation

To install the required packages, run the following command:

```bash
git clone --depth 1 git@github.com:LeCAR-Lab/model-based-diffusion.git
pip install -e .
```

## Usage

### Model-based Diffusion for Trajectory Optimization

To run model-based diffusion to optimize a trajectory, run the following command:

```bash
cd mbd/planners
python mbd_planner.py --env_name $ENV_NAME
```

where `$ENV_NAME` is the name of the environment, you can choose from `hopper`, `halfcheetah`, `walker2d`, `ant`, `humanoidrun`, `humanoidstandup`, `humanoidtrack`, `car2d`, `pushT`.

To run model-based diffusion combined with demonstrations, run the following command:

```bash
cd mbd/planners
python mbd_planner.py --env_name $ENV_NAME --enable_demos
```

Currently, only the `humanoidtrack`, `car2d` support demonstrations.

To run multiple seeds, run the following command:

```bash
cd mbd/scripts
python run_mbd.py --env_name $ENV_NAME
```

To visualize the diffusion process, run the following command:

```bash
cd mbd/scripts
python visualize_mbd.py --env_name $ENV_NAME
```

Please make sure you have run the planner first to generate the data.

### Model-based Diffusion for Black-box Optimization

To run model-based diffusion for black-box optimization, run the following command:

```bash
cd mbd/blackbox
python mbd_opt.py
```

### Other Baselines

To run RL-based baselines, run the following command:

```bash
cd mbd/rl
python train_brax.py --env_name $ENV_NAME
```

To run other zeroth order trajectory optimization baselines, run the following command:

```bash
cd mbd/planners
python path_integral.py --env_name $ENV_NAME --mode $MODE
```

where `$MODE` is the mode of the planner, you can choose from `mppi`, `cem`, `cma-es`.

## Acknowledgements

* This codebase's environment and RL implementation is built on top of [Brax](https://github.com/google/brax).
