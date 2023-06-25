import torch

def render_scene(env, render_mode, window_size, ego_vehicle=None, view_dist=None, 
                 view_angle=None, draw_target=True, padding=10.0):

    if render_mode == "whole_scene":

        render_scene = env.scenario.getImage(
            img_width=1200,
            img_height=1200,
            padding=10.0,
            draw_target_positions=True,
        )
        
    elif render_mode == "agent_view":
         
         render_scene = env.scenario.getConeImage(
            # Select one of the vehicles we are controlling
            source=ego_vehicle, #env.controlled_vehicles[1], 
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
    
    return last_zero_indices