import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PPOExperimentConfig:
    # fmt: off
    seed: int = 12                    # seed of the experiment
    torch_deterministic = True        # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda = True                       # if toggled, cuda will be enabled by default
    total_iters: int = 2 #500            # total iterations of the experiments
    num_policy_rollouts: int = 20 # 90  # determines the batch size, the amount of experience to collect before doing an optim step
    num_steps: int = 90               # the number of steps to run in each environment per policy rollout
    learning_rate: float = 1e-4       # the learning rate of the optimizer 
    anneal_lr: float = True           # toggle learning rate annealing for policy and value networks
    gamma: float = 0.99               # the discount factor gamma
    gae_lambda: float = 0.95          # the lambda for the general advantage estimation
    update_epochs: int = 4            # the K epochs to update the policy
    norm_adv: bool = True             # toggles advantages normalization
    clip_coef: float = 0.2            # the surrogate clipping coefficient
    clip_vloss: bool = True           # toggles whether or not to use a clipped loss for the value function, as per the paper
    ent_coef: float = 0.0             # coefficient of the entropy
    vf_coef: float = 0.5              # coefficient of the value function
    max_grad_norm: float = 0.5        # the maximum norm for the gradient clipping
    human_kl_lam: float = 0           # coefficient of kl_div to human anchor policy
    target_kl: float = None           # the target KL divergence threshold
    save_model: bool = True           # save policy and value networks
    device: str = None
    max_agents: int = 2
    num_processes: int = 2
    # fmt: on


@dataclass
class WandBSettings:
    track: bool = False
    record_video: bool = False
    render_mode: str = "whole_scene"  # options: whole_scene / agent_view
    log_every_t_iters: int = 20
    log_every_t_steps: int = 5  # Take a snapshot every T steps
    render_fps: int = 5
    window_size:int = 1000
    draw_target: bool = True
    project_name: str = "test_ppo"
    group: str = "debug"
    exp_name: str = "test_multiprocessing" #"ppo_mem_5_penal_coll_5"


@dataclass
class HumanPolicyConfig:
    batch_size: int = 1
    hidden_layers = [1025, 256, 128]  # Model used in paper
    actions_discretizations = [15, 42]
    actions_bounds = [[-6, 6], [-0.7, 0.7]]  # Bounds for (acc, steering)
    lr: float = 1e-4
    pretrained_model_path: str = (
        "experiments/human_regularized/models/human_anchor_policy_AS.pth"
    )
