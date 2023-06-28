import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PPOExperimentConfig:
    # fmt: off
    seed: int = 12                    # seed of the experiment
    torch_deterministic = True        # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda = True                       # if toggled, cuda will be enabled by default
<<<<<<< HEAD
    total_iters: int = 3000           # total iterations of the experiments
    num_policy_rollouts: int = 90     # determines the batch size, the amount of experience to collect before doing an optim step
    num_steps: int = 90               # the number of steps to run in each environment per policy rollout
    learning_rate: float = 1e-4       # the learning rate of the optimizer 
=======
    total_iters: int = 100            # total iterations of the experiments
    num_policy_rollouts: int = 70     # determines the batch size, the amount of experience to collect before doing an optim step
    num_steps: int = 80               # the number of steps to run in each environment per policy rollout
    learning_rate: float = 2.5e-4     # the learning rate of the optimizer 
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
    anneal_lr: float = True           # toggle learning rate annealing for policy and value networks
    gamma: float = 0.99               # the discount factor gamma
    gae_lambda: float = 0.95          # the lambda for the general advantage estimation
    update_epochs: int = 5            # the K epochs to update the policy
    norm_adv: bool = True             # toggles advantages normalization
    clip_coef: float = 0.2            # the surrogate clipping coefficient
    clip_vloss: bool = True           # toggles whether or not to use a clipped loss for the value function, as per the paper
    ent_coef: float = 0.0             # coefficient of the entropy
<<<<<<< HEAD
    vf_coef: float = 0.5              # coefficient of the value function
=======
    vf_coef: float = 0.2              # coefficient of the value function
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
    max_grad_norm: float = 0.5        # the maximum norm for the gradient clipping
    human_kl_lam: float = 0           # coefficient of kl_div to human anchor policy
    target_kl: float = None           # the target KL divergence threshold
    save_model: bool = True           # save policy and value networks
    save_path: str = 'experiments/human_regularized/ppo_agent_models' # Define the path where you want to save the model
    # fmt: on


@dataclass
class NocturneConfig:
    env_id: str = "Nocturne-v0"
    nocturne_rl_cfg: str = "experiments/human_regularized/rl_config.yaml"
    view_dist: float = 80
    view_angle: float = 3.14
    train_data_dir: str = None
    valid_data_dir: str = None
    test_split: float = None


@dataclass
class WandBSettings:
    track: bool = True
    record_video: bool = False
<<<<<<< HEAD
    render_mode: str = "whole_scene"  # options: whole_scene / agent_view
    log_every_t_iters: int = 20
    log_every_t_steps: int = 5  # Take a snapshot every T steps
    render_fps: int = 5
    window_size:int = 1000
    draw_target: bool = True
    project_name: str = "test_ppo"
    group: str = "sweep"
    exp_name: str = "ppo_mem_5_penal_coll_5"

=======
    render_mode: str = 'whole_scene' # options: whole_scene / agent_view
    log_every_t_iters: int = 20
    log_every_t_steps: int = 5 # Take a snapshot every T steps
    render_fps: int = 5
    window_size: int = 1000
    draw_target: bool = True
    project_name: str = 'test_ppo'
    group: str = 'nocturne'
    exp_name: str = 'debug'
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8

@dataclass
class HumanPolicyConfig:
    batch_size: int = 1
    hidden_layers = [1025, 256, 128]  # Model used in paper
    actions_discretizations = [15, 42]
    actions_bounds = [[-6, 6], [-0.7, 0.7]]  # Bounds for (acc, steering)
    lr: float = 1e-4
<<<<<<< HEAD
    pretrained_model_path: str = (
        "experiments/human_regularized/models/human_anchor_policy_AS.pth"
    )
=======
    pretrained_model_path: str = 'experiments/human_regularized/human_anchor_policy_AS.pth'
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
