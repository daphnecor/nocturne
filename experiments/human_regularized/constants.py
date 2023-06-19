import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PPOExperimentConfig:
    seed: int = 1                     # seed of the experiment
    torch_deterministic = True        # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda = False                      # if toggled, cuda will be enabled by default
    total_iters: int = 5000           # total iterations of the experiments
    num_policy_rollouts: int = 20     # determines the batch size, the amount of experience to collect before doing an optim step
    num_steps: int = 80               # the number of steps to run in each environment per policy rollout
    learning_rate: float = 2.5e-4     # the learning rate of the optimizer 
    anneal_lr: float = True           # toggle learning rate annealing for policy and value networks
    gamma: float = 0.99               # the discount factor gamma
    gae_lambda: float = 0.95          # the lambda for the general advantage estimation
    update_epochs: int = 4            # the K epochs to update the policy
    norm_adv: bool = True             # toggles advantages normalization
    clip_coef: float = 0.2            # the surrogate clipping coefficient
    clip_vloss: bool = True           # toggles whether or not to use a clipped loss for the value function, as per the paper
    ent_coef: float = 0.01            # coefficient of the entropy
    vf_coef: float = 0.5              # coefficient of the value function
    max_grad_norm: float = 0.5        # the maximum norm for the gradient clipping
    lam: float = 0                    # coefficient of kl_div to human anchor policy
    target_kl: float = None           # the target KL divergence threshold

@dataclass
class NocturneConfig:
    env_id: str = 'Nocturne-v0'
    nocturne_rl_cfg: str = '/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml'
    train_data_dir: str = None
    valid_data_dir: str = None
    test_split: float = None

@dataclass
class HumanPolicyConfig:
    batch_size: int = 1
    hidden_layers = [1025, 256, 128]  # Model used in paper
    actions_discretizations = [15, 42]
    actions_bounds = [[-6, 6], [-0.7, 0.7]]  # Bounds for (acc, steering)
    lr: float = 1e-4
    pretrained_model_path: str = '/scratch/dc4971/nocturne/experiments/human_regularized/human_anchor_policy_AS.pth'

@dataclass
class WandBSettings:
    track: bool = False
    project_name: str = 'human_regularized_rl'
    group: str = 'nocturne'
    exp_name: str = 'ppo_large_batch'

