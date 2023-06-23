import datetime
import glob
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
#from torch.profiler import profile, schedule, tensorboard_trace_handler

import wandb
import yaml

from base_env import BaseEnv
from cfgs.config import set_display_window
from constants import (
    PPOExperimentConfig,
    NocturneConfig,
    WandBSettings,
    HumanPolicyConfig,
)
from dataclasses import asdict, dataclass
from imit_models import BehavioralCloningAgentJoint
from nocturne import Action

logging.basicConfig(level=logging.CRITICAL)


def make_env(config_path, seed, run_name):
    """Make nocturne environment."""
    with open(config_path, "r") as stream:
        base_env_config = yaml.safe_load(stream)
    env = BaseEnv(base_env_config)
    env.seed(seed)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def dict_to_tensor(my_dict):
    tensor_list = []
    for agent_id, tensor in my_dict.items():
        tensor_list.append(torch.Tensor(tensor))
    stacked_tensor = torch.stack(tensor_list, dim=1)
    return stacked_tensor.squeeze()

class Agent(nn.Module):
    def __init__(self, envs, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(device)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_policy(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs


class RolloutBuffer:
    """
    Shared rollout buffer to store collected trajectories for every agent we control.
    Must be reset after the policy network is updated.
    """

    def __init__(
        self, controlled_agents, num_steps, obs_space_dim, act_space_dim, device
    ):
        """
        Args:
            controlled_vehicles (list[nocturne vehicles])
        """
        self.observations = self.create_tensor_dict(
            controlled_agents, num_steps, device, obs_space_dim
        )
        self.actions = self.create_tensor_dict(
            controlled_agents,
            num_steps,
            device,
        )
        self.logprobs = self.create_tensor_dict(controlled_agents, num_steps, device)
        self.rewards = self.create_tensor_dict(controlled_agents, num_steps, device)
        self.dones = self.create_tensor_dict(controlled_agents, num_steps, device)
        self.values = self.create_tensor_dict(controlled_agents, num_steps, device)

    def create_tensor_dict(self, controlled_agents, num_steps, device="cpu", dim=None):
        tensor_dict = {}
        for agent in controlled_agents:
            key = agent
            if dim is not None:
                tensor = torch.zeros((num_steps, dim))
            else:
                tensor = torch.zeros((num_steps,))
            tensor_dict[key] = tensor.to(device)
        return tensor_dict

    def clear(self):
        for key in self.observations.keys():
            self.observations[key].zero_()
            self.actions[key].zero_()
            self.logprobs[key].zero_()
            self.rewards[key].zero_()
            self.dones[key].zero_()
            self.values[key].zero_()

if __name__ == "__main__":

    # Configs
    args_exp = PPOExperimentConfig()
    args_wandb = WandBSettings()
    args_env = NocturneConfig()
    args_human_pol = HumanPolicyConfig()

    # Logging
    now = datetime.datetime.now()
    formatted_time = now.strftime("%D%H%M")
    run_name = f"{args_env.env_id}__{args_wandb.exp_name}_{formatted_time}"
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_exp).items()])),
    )

    if args_wandb.track:
        wandb.init(
            project=args_wandb.project_name,
            sync_tensorboard=True,
            config=asdict(args_wandb),
            group="nocturne",
            name=run_name,
            save_code=True,
        )

    # Seeding
    random.seed(args_exp.seed)
    np.random.seed(args_exp.seed)
    torch.manual_seed(args_exp.seed)
    torch.backends.cudnn.deterministic = args_exp.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args_exp.cuda else "cpu"
    )

    logging.critical(f'DEVICE: {device}')

    # Env setup
    set_display_window()
    env = make_env(args_env.nocturne_rl_cfg, args_exp.seed, run_name)

    # State and action space dimension
    observation_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    agent = Agent(env, observation_space_dim, action_space_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args_exp.learning_rate, eps=1e-5)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # Load human anchor policy
    human_anchor_policy = BehavioralCloningAgentJoint(
        num_inputs=observation_space_dim,
        config=args_human_pol,
        device=device,
    ).to(device)

    human_anchor_policy.load_state_dict(
        torch.load(args_human_pol.pretrained_model_path)
    )

    global_step = 0
    start_time = time.time()
    MAX_AGENTS = 2

    for iter in range(1, args_exp.total_iters + 1):

        logging.critical(f'iter: {iter}')

        # Storage
        obs_tensor = torch.zeros(
            (
                args_exp.num_policy_rollouts,
                args_exp.num_steps,
                MAX_AGENTS,
                observation_space_dim,
            )
        ).to(device)
        rew_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        act_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        done_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        logprob_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        value_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        veh_coll_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(device)
        edge_coll_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(device)
        goal_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(device)

        #TODO: Do rollouts on the GPU
        # # # #  Collect experience with current policy  # # # #
        start_rollout = time.time()
        for rollout_step in range(args_exp.num_policy_rollouts):

            # Reset environment
            # NOTE: this can either be the same env or a new traffic scene
            # currently using the same scene for debugging purposes
            start_reset = time.time()

            next_obs_dict = env.reset()

            controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
            num_agents = len(controlled_agents)
            dict_next_done = {agent_id: False for agent_id in controlled_agents}
            already_done_ids = env.done_ids.copy()
            current_ep_reward = 0

            # Set data buffer for within scene logging
            buffer = RolloutBuffer(
                controlled_agents,
                args_exp.num_steps,
                observation_space_dim,
                action_space_dim,
                device,
            )

            # Adapt learning rate based on how far we are in the learning process
            if args_exp.anneal_lr:
                frac = 1.0 - (iter - 1.0) / args_exp.total_iters
                lrnow = frac * args_exp.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            frames = []

            # # # #  Interact with environment  # # # #
            start_env = time.time()
            for step in range(0, args_exp.num_steps):
                if args_wandb.record_video:
                    # Render env every zeroth rollout
                    if step % 5 == 0 and rollout_step == 0:
                        if args_wandb.render_mode == "whole_scene":
                            render_scene = env.scenario.getImage(
                                img_width=args_wandb.window_size,
                                img_height=args_wandb.window_size,
                                padding=0,
                                draw_target_positions=args_wandb.draw_target,
                            )
                            
                        elif args_wandb.render_mode == "agent_view":
                            render_scene = env.scenario.getConeImage(
                                # Select one of the vehicles we are controlling
                                source=env.controlled_vehicles[0], 
                                view_dist=args_env.view_dist,
                                view_angle=args_env.view_angle,
                                head_angle=0,
                                img_width=args_wandb.window_size,
                                img_height=args_wandb.window_size,
                                padding=10.0,
                                draw_target_position=args_wandb.draw_target,
                            )

                        frames.append(render_scene.T)
                            

                global_step += 1

                for agent_id in controlled_agents:
                    buffer.dones[agent_id][step] = dict_next_done[agent_id] * 1
                    if agent_id not in already_done_ids:
                        buffer.observations[agent_id][step, :] = torch.Tensor(
                            next_obs_dict[agent_id]
                        )
                    else:
                        continue

                # Action logic
                #TODO: multiprocess actions
                with torch.no_grad():
                    # Nocturne expects a dictionary with actions, we create an item
                    # for every agent that is still active (i.e. not done)
                    action_dict = {
                        agent_id: None
                        for agent_id in controlled_agents
                        if agent_id not in already_done_ids
                    }

                    for agent_id in action_dict.keys():
                        # Take an action
                        action, logprob, _, value = agent.get_action_and_value(
                            torch.Tensor(next_obs_dict[agent_id]).to(device)
                        )
                        # Store in buffer
                        buffer.values[agent_id][step] = value.flatten()
                        buffer.actions[agent_id][step] = action
                        buffer.logprobs[agent_id][step] = logprob
                        # Store in action_dict
                        action_dict[agent_id] = action.item()

                moving_vehs = env.scenario.getObjectsThatMoved()

                if args_env.take_random_actions:
                    # Sanity check A: If activated, agents take random actions
                    # accel_interval = np.linspace(-6, 6, num=15)
                    # steer_interval = np.linspace(-0.7, 0.7, num=21)
                    # OR take very simple intervals
                    accel_interval = np.array([-6, 6])
                    steer_interval = np.array([-1, 1])
                    action_dict = {
                        agent_id: Action(
                            acceleration=np.random.choice(accel_interval, size=1),
                            steering=np.random.choice(steer_interval, size=1),
                        )
                        for agent_id in controlled_agents
                        if agent_id not in already_done_ids
                    }

                    # Sanity check B: take fixed actions
                    # action_dict = {
                    #     veh.id: Action(acceleration=2.0, steering=1.0)
                    #     for veh in moving_vehs
                    # }

                # Take simultaneous action in env
                next_obs_dict, reward_dict, next_done_dict, info_dict = env.step(
                    action_dict
                )

                # Sanity check C: Check (x, y) coordinates for both vehicles
                if args_wandb.track:
                    for veh in env.controlled_vehicles:
                        wandb.log(
                            {
                                "global_step": step,
                                f"veh_{veh.id}_x_pos": veh.position.x,
                                f"veh_{veh.id}_y_pos": veh.position.y,
                            }
                        )

                # Store rewards
                for agent_id in next_obs_dict.keys():
                    buffer.rewards[agent_id][step] = torch.from_numpy(
                        np.asarray(reward_dict[agent_id])
                    ).to(device)

                # Update collisions and whether goal is achieved
                veh_coll_tensor[rollout_step] += sum(
                    info["veh_veh_collision"] for info in info_dict.values()
                )
                edge_coll_tensor[rollout_step] += sum(
                    info["veh_edge_collision"] for info in info_dict.values()
                )
                goal_tensor[rollout_step] += sum(
                    info["goal_achieved"] for info in info_dict.values()
                )

                # Update episodal rewards
                current_ep_reward += sum(reward_dict.values())

                # Update done agents
                for agent_id in next_obs_dict.keys():
                    if next_done_dict[agent_id]:
                        dict_next_done[agent_id] = True
                        # Fill dones with ones from step where terminal state was reached
                        buffer.dones[agent_id][step:] = 1

                already_done_ids = [
                    agent_id for agent_id, value in dict_next_done.items() if value
                ]

                # End the game early if all agents are done
                if len(already_done_ids) == num_agents or step == args_exp.num_steps:
                    last_step = step
                    logging.info(f'complete_episode: {time.time() - start_env} | time_per_step: {(time.time() - start_env)/step} \n')
                    break

    
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
            # Store rollout scene experience
            obs_tensor[rollout_step, :, :num_agents, :] = dict_to_tensor(
                buffer.observations
            ).to(device)
            rew_tensor[rollout_step, :, :num_agents] = dict_to_tensor(
                buffer.rewards
            ).to(device)
            act_tensor[rollout_step, :, :num_agents] = dict_to_tensor(
                buffer.actions
            ).to(device)
            done_tensor[rollout_step, :, :num_agents] = dict_to_tensor(buffer.dones).to(
                device
            )
            logprob_tensor[rollout_step, :, :num_agents] = dict_to_tensor(
                buffer.logprobs
            ).to(device)
            value_tensor[rollout_step, :, :num_agents] = dict_to_tensor(
                buffer.values
            ).to(device)

            # Normalize counts by agents in scene
            veh_coll_tensor[rollout_step] /= num_agents
            edge_coll_tensor[rollout_step] /= num_agents
            goal_tensor[rollout_step] /= num_agents

            # Clear buffer for next rollout
            buffer.clear()

            logging.info(f"Episodic_return: {current_ep_reward}")

            if args_wandb.track:
                wandb.log({
                    "global_step": global_step,
                    "charts/num_agents_in_scene": num_agents,
                    "charts/episodic_return": current_ep_reward,
                    "charts/episodic_length": step,
                    "charts/goal_achieved_rate": goal_tensor[rollout_step],
                    "charts/veh_veh_collision_rate": veh_coll_tensor[rollout_step],
                    "charts/veh_edge_collision_rate": edge_coll_tensor[rollout_step],
                })

            if args_wandb.track:
                if rollout_step == 0 and args_wandb.record_video:
                    movie_frames = np.array(frames, dtype=np.uint8)
                    wandb.log(
                        {
                            "iter": iter,
                            "scene_videos": wandb.Video(
                                movie_frames,
                                fps=args_wandb.render_fps,
                                caption=f"Training iter: {iter}",
                            ),
                        }
                    )
                    del movie_frames

        rollouts_done = time.time()
        logging.info(f'total_rollout_time: {rollouts_done - start_rollout} | per rollout: {(rollouts_done - start_rollout)/args_exp.num_policy_rollouts}')

        wandb.log({"global_iter": iter})

        # # # # Compute advantage estimate via GAE on collected experience # # # #
        # Select the last observation for every policy rollout
        #NOTE: Check, I need the last observation for every agent & rollout
        next_obs = obs_tensor[:, last_step, :, :].reshape(
            -1, observation_space_dim
        )  # (N_steps * N_rollouts, D_obs)
        next_done = done_tensor[:, last_step, :].reshape(-1)  # (N_steps * N_rollouts)

        # Flatten over rollout x agent dimension
        dones = done_tensor.reshape(
            (args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts)
        )
        values = value_tensor.reshape(
            (args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts)
        )
        rewards = rew_tensor.reshape(
            (args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts)
        )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros(
                (args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts)
            ).to(device)
            lastgaelam = 0
            for t in reversed(range(args_exp.num_steps)):
                if t == args_exp.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # Compute TD-error
                delta = (
                    rewards[t]
                    + args_exp.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                # Update advantage for timestep
                advantages[t] = lastgaelam = (
                    delta
                    + args_exp.gamma
                    * args_exp.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # # # #  Optimization   # # # #
        b_obs = obs_tensor.reshape(-1, observation_space_dim)
        b_logprobs = logprob_tensor.reshape(-1)
        b_actions = act_tensor.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # Since in our multi-agent env some agents finsh earlier than others,
        # there will be entries without observations. We filter these out, as we
        # only want to train on valid sequences
        valid_b_inds = torch.nonzero(torch.any(b_obs != 0, dim=1), as_tuple=False)[:, 0]
        
        logging.info(f'batch_size: {len(valid_b_inds)}')

        clipfracs = []
        batch_size = len(valid_b_inds)
        minibatch_size = batch_size // num_agents
        
        start_optim = time.time()
        
        for epoch in range(args_exp.update_epochs):

            # Shuffle batch indices
            indices = torch.randperm(valid_b_inds.size(0))
            valid_b_inds = valid_b_inds[indices]

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = valid_b_inds[start:end]

                # COMPUTE KL DIV TO HUMAN ANCHOR POLICY
                action, log_prob, tau_dist = human_anchor_policy(b_obs[mb_inds])
                actor_dist = agent.get_policy(b_obs[mb_inds])
                # kl_div = kl_loss(tau_dist.probs, actor_dist.probs)

                # Check if the minibatch has at least two elements
                if len(mb_inds) < 2:
                    continue  # Skip this minibatch and move to the next one

                # Compute new logprobs of state dist
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )

                # Compute r(theta) between old and new policy
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args_exp.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args_exp.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args_exp.clip_coef, 1 + args_exp.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args_exp.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args_exp.clip_coef,
                        args_exp.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    - args_exp.ent_coef * entropy_loss
                    + args_exp.vf_coef * v_loss
                    # - args_exp.lam * kl_div 
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args_exp.max_grad_norm)
                optimizer.step() 

            if args_exp.target_kl is not None:
                if approx_kl > args_exp.target_kl:
                    break
        
        # # # #     End of iteration     # # # #
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        #TODO: Check what goes wrong, explained var is negative which can't be the case
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args_wandb.track:
            wandb.log({
                "global_step": global_step,
                "charts/b_advantages": wandb.Histogram(b_advantages.cpu().numpy()),
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/kl_policy_update": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
            })

        end_optim = time.time()
        logging.info(f'optim_step_time: {end_optim - start_optim}')

    # # # #    End of run    # # # #
    env.close()
    writer.close()
