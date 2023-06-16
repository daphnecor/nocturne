from pathlib import Path
from dataclasses import dataclass

@dataclass
class PPOExperimentConfig:
    experiment_name: str
    num_epochs: int
    num_steps_per_epoch: int
    batch_size: int
    env_name: str
    ppo: "PPOConfig"
    data: "DataSettings"
    wandb: "WandBSettings"
    # Additional experiment-specific settings

@dataclass
class PPOConfig:
    env_name: str = 'Nocturne-v0'
    has_cont_action_space: bool = False
    max_ep_len: int = 80                    # max timesteps in one episode
    max_iters: int = 300                    # break training loop if timeteps > max_training_timesteps
    update_timestep: int = max_ep_len * 4
    gamma: float = 0.99
    K_epochs: int = 40                      # update policy for K epochs (optimization epochs)
    eps_clip: float = 0.2                   # clip parameter for PPO
    value_loss_coef: float = 0.1            # discount factor
    lr_actor: float = 0.0003                # learning rate for actor network
    lr_critic: float = 0.001                # learning rate for critic network
    random_seed: int = 0                    # set random seed if required (0 = no random seed)
    entropy_coef: float = 0.3               #TODO
    max_grad_norm: float = 1                #TODO

@dataclass
class DataSettings:
    base_dir: str = None
    nocturne_env_dir = Path('/Users/Daphne/Github Repositories/nocturne-research/cfgs/config.yaml')
    train_data_dir: str = None
    valid_data_dir: str = None
    test_split: float = None

@dataclass
class HumanPolicyConfig:
    batch_size: int = 1
    hidden_layers = [1025, 256, 128],  # Model used in paper
    actions_discretizations = [15, 42],
    actions_bounds = [[-6, 6], [-0.7, 0.7]],  # Bounds for (acc, steering)
    lr: float = 1e-4

@dataclass
class WandBSettings:
    enabled: bool = False
    project: str = 'nocturne_rl'
    group: str = 'testing'
