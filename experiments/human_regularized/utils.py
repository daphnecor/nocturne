import torch

<<<<<<< HEAD

def render_scene(env, render_mode, window_size, ego_vehicle=None, view_dist=None, view_angle=None, draw_target=True, padding=10.0):
    """
    Renders a nocturne scene.

    Args:
        env (object): The environment object.
        render_mode (str): The rendering mode ("whole_scene" or "agent_view").
        window_size (int): The size of the rendering window.
        ego_vehicle (object, optional): The ego vehicle object. Defaults to None.
        view_dist (float, optional): The viewing distance. Defaults to None.
        view_angle (float, optional): The viewing angle. Defaults to None.
        draw_target (bool, optional): Flag to indicate whether to draw the target. Defaults to True.
        padding (float, optional): The padding value. Defaults to 10.0.

    Returns:
        torch.Tensor: The rendered scene.
    """

    if render_mode == "whole_scene":
=======
def render_scene(env, render_mode, window_size, ego_vehicle=None, view_dist=None, 
                 view_angle=None, draw_target=True, padding=10.0):

    if render_mode == "whole_scene":

>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
        render_scene = env.scenario.getImage(
            img_width=1200,
            img_height=1200,
            padding=10.0,
            draw_target_positions=True,
        )
        
    elif render_mode == "agent_view":
<<<<<<< HEAD
        render_scene = env.scenario.getConeImage(
            source=ego_vehicle,
=======
         
         render_scene = env.scenario.getConeImage(
            # Select one of the vehicles we are controlling
            source=ego_vehicle, #env.controlled_vehicles[1], 
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
            view_dist=view_dist,
            view_angle=view_angle,
            head_angle=0,
            img_width=window_size,
            img_height=window_size,
            padding=padding,
            draw_target_position=draw_target,
        )
         
    return render_scene.T


def find_last_zero_index(tensor):
<<<<<<< HEAD
    """
    Finds the index of the last occurrence of zero in each row of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The indices of the last occurrence of zero in each row.
    """

=======
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
    num_items, num_steps = tensor.shape
    last_zero_indices = torch.zeros(num_items, dtype=torch.long)
    
    for i in range(num_items):
        found_zero = False
        for j in range(num_steps-1, -1, -1):
            if tensor[i, j] == 0:
                last_zero_indices[i] = j
                found_zero = True
                break
        if not found_zero:
            last_zero_indices[i] = -1
    
<<<<<<< HEAD
    return last_zero_indices


def dict_to_tensor(my_dict):
    """
    Converts a dictionary of tensors to a stacked tensor.

    Args:
        my_dict (dict): The input dictionary.

    Returns:
        torch.Tensor: The stacked tensor.
    """

    tensor_list = []
    for agent_id, tensor in my_dict.items():
        tensor_list.append(torch.Tensor(tensor))
    stacked_tensor = torch.stack(tensor_list, dim=1)
    return stacked_tensor.squeeze()


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
=======
    return last_zero_indices
>>>>>>> 1e9848cf4d866bac10e66aea39c9d7bfb28848a8
