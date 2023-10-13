# `nocturne_lab`: fast driving simulator 🧪 + 🚗

`nocturne_lab` is a maintained fork of [Nocturne](https://github.com/facebookresearch/nocturne); a 2D, partially observed, driving simulator built in C++. Currently, `nocturne_lab` is used internally at the Emerge lab. You can get started with the intro examples 🏎️💨 [here](https://github.com/Emerge-Lab/nocturne_lab/tree/feature/nocturne_fork_cleanup/examples).

## Basic usage

```python
from nocturne.envs.base_env import BaseEnv

# Initialize an environment
env = BaseEnv(config=env_config)

# Reset
obs_dict = env.reset()

# Get info
agent_ids = [agent_id for agent_id in obs_dict.keys()]
dead_agent_ids = []

for step in range(1000):

    # Sample actions
    action_dict = {
        agent_id: env.action_space.sample() 
        for agent_id in agent_ids
        if agent_id not in dead_agent_ids
    }
    
    # Step in env
    obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

    # Update dead agents
    for agent_id, is_done in done_dict.items():
        if is_done and agent_id not in dead_agent_ids:
            dead_agent_ids.append(agent_id)

    # Reset if all agents are done
    if done_dict["__all__"]:
        obs_dict = env.reset()
        dead_agent_ids = []

# Close environment
env.close()
```

## Implemented algorithms

| Algorithm                              | Reference                                                  | Code  | Compatible with    | Notes                                                                                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------- | ----- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PPO **single-agent** control | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) | [ppo_with_sb3.ipynb](https://github.com/Emerge-Lab/nocturne_lab/blob/feature/nocturne_fork_cleanup/examples/04_ppo_with_sb3.ipynb) | Stable baselines 3 |                                                                                                                                                                        |
| PPO **multi-agent** control  | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) | `#TODO` | Stable baselines 3 | SB3 doesn't support multi-agent environments. Using the `VecEnv`class to treat observations from multiple agents as a set of vectorized single-agent environments. |
|                                        |                                                            |       |                    |                                                                                                                                                                        |
|                                        |                                                            |       |                    |                                                                                                                                                                        |

## Installation
The instructions for installing Nocturne locally are provided below. To use the package on a HPC (e.g. HPC Greene), follow the instructions in [./hpc/hpc_setup.md](./hpc/hpc_setup.md).

### Requirements

* Python (>=3.10)

### Virtual environment
Below different options for setting up a virtual environment are described. Either option works although `pyenv` is recommended.

> _Note:_ The virtual environment needs to be **activated each time** before you start working.

#### Option 1: `pyenv`
Create a virtual environment by running:

```shell
pyenv virtualenv 3.10.12 nocturne_lab
```

The virtual environment should be activated every time you start a new shell session before running subsequent commands:

```shell
pyenv shell nocturne_lab
```

Fortunately, `pyenv` provides a way to assign a virtual environment to a directory. To set it for this project, run:
```shell
pyenv local nocturne_lab
```

#### Option 2: `conda`
Create a conda environment by running:

```shell
conda env create -f ./environment.yml
```

This creates a conda environment using Python 3.10 called `nocturne_lab`.

To activate the virtual environment, run:

```shell
conda activate nocturne_lab
```

#### Option 3: `venv`
Create a virtual environment by running:

```shell
python -m venv .venv
```

The virtual environment should be activated every time you start a new shell session before running the subsequent command:

```shell
source .venv/bin/activate
```

### Dependencies

`poetry` is used to manage the project and its dependencies. Start by installing `poetry` in your virtual environment:

```shell
pip install poetry
```

Before installing the package, you first need to synchronise and update the git submodules by running:

```shell
# Synchronise and update git submodules
git submodule sync
git submodule update --init --recursive
```

You also need to have SFML installed, which you can do by running:
```shell
brew install sfml 
```
on a mac or 
```shell
sudo apt-get install libsfml-dev
```
on Linux.

Now install the package by running:

```shell
poetry install
```
> _Note:_ If it fails to build `nocturne`, try running `poetry build` to obtain a more descriptive error message.

> Under the hood the `nocturne` package uses the `nocturne_cpp` Python package that wraps the Nocturne C++ code base and provides bindings for Python to interact with the C++ code using `pybind11`.


### Development setup
To configure the development setup, run:
```shell
# Install poetry dev dependencies
poetry install --only=dev

# Install pre-commit (for flake8, isort, black, etc.)
pre-commit install

# Optional: Install poetry docs dependencies
poetry install --only=docs
```

## Ongoing work

Here is a list of features that we are developing:

- @Daphne: Support for SB3's PPO algorithm with multi-agent control
- @Alex: Logging and unit testing
- @Tiyas: Random resets
