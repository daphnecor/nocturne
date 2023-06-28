import numpy as np
import random
import torch

def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def find_nearest_grid_idx(action_grids, actions):
    """
    Convert a batch of actions values to a batch of the nearest action indexes (for discrete actions only).
    credits https://stackoverflow.com/a/46184652/16207351
    Args:
        actions_grids (List[Tensor]): A list of one-dimensional tensors representing the grid of possible
            actions for each dimension of the actions tensor.
        actions (Tensor): A two-dimensional tensor of size (batch_size, num_actions).
    Returns:
        Tensor: A tensor of size (batch_size, num_actions) with the indices of the nearest action in the action grids.
    """
    output = torch.zeros_like(actions)
    for i, action_grid in enumerate(action_grids):
        action = actions[:, i]

        # get indexes where actions would be inserted in action_grid to keep it sorted
        idxs = torch.searchsorted(action_grid, action)

        # if it would be inserted at the end, we're looking at the last action
        idxs[idxs == len(action_grid)] -= 1

        # find indexes where previous index is closer (simple grid has constant sampling intervals)
        idxs[action_grid[idxs] - action > torch.diff(action_grid).mean() * 0.5] -= 1

        # write indexes in output
        output[:, i] = idxs
    return output


def compute_log_prob(action_dists, ground_truth_action, action_grids, reduction='mean', return_indexes=False):
    """Compute the log probability of the expert action for a number of action distributions.
        Losses are averaged over observations for each batch by default.

    Args:
        action_dists (List[Categorical]): Distributions over model actions.
        ground_truth_action (Tensor): Action taken by the expert.
    Returns:
        Tensor of size (num_actions, batch_size) if reduction == 'none' else (num_actions)
    """

    # Find indexes in actions grids whose values are the closest to the ground truth actions
    expert_action_idx = find_nearest_grid_idx(
        action_grids=action_grids, 
        actions=ground_truth_action,
    )

    # Stack log probs of actions indexes wrt. Categorial variables for each action dimension
    log_probs = torch.stack([dist.log_prob(expert_action_idx[:, i]) for i, dist in enumerate(action_dists)])

    if reduction == 'none':
        return (log_probs, expert_action_idx) if return_indexes else log_probs    

    elif reduction == 'sum': 
        agg_log_prob = log_probs.sum(axis=1)
        
    elif reduction == 'mean': 
        agg_log_prob = log_probs.mean(axis=1)
    
    return (agg_log_prob, expert_action_idx) if return_indexes else agg_log_prob


def get_total_norm(model):
    """
    Compute the total norm of the gradients of a model's parameters.

    This function calculates the L2 norm of the gradients for each parameter in the model,
    and then computes the total norm by summing the squared values of the individual norms
    and taking the square root of the sum.

    Args:
        model (nn.Module): The model for which to compute the total gradient norm.

    Returns:
        float: The total gradient norm.

    Note:
        - If a parameter's gradient is `None`, it will be excluded from the computation.
        - The gradients are detached from the computational graph before computing the norm.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm