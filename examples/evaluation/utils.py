from collections import defaultdict
from nocturne import Simulation
import numpy as np
import torch


def construct_state(scenario, vehicle, view_dist=80, view_angle=np.radians(180)):
    """Construct the full state for a vehicle.
    Args:
        scenario (nocturne_cpp.Scenario): Simulation at a particular timepoint.
        vehicle (nocturne_cpp.Vehicle): A vehicle object in the simulation.
        view_dist (int): Viewing distance of the vehicle.
        view_angle (int): The view cone angle in radians.
    Returns:
        state (ndarray): The vehicle state.
    """
    ego_state = scenario.ego_state(
        vehicle
    )
    visible_state = scenario.flattened_visible_state(
        vehicle, 
        view_dist=view_dist, 
        view_angle=view_angle
    )
    return np.concatenate((ego_state, visible_state))


def evaluate_agent_in_traffic_scene_(
        sim, scenario_config, num_stacked_states, model,
        num_steps=90, step_size=0.1, invalid_pos=-1e4, warmup_period=10,
        allowed_goal_distance=0.5,
    ):
    """Compute the collision and/or goal rate for a file (traffic scene).
    Args:
        sim (str): The traffic scenario.
        scenario_config (Dict): Initialization parameters.
        num_stacked_states (int): The memory of an agent.
        model (BehavioralCloningAgent): The model to be evaluated.  
        num_steps (int, optional): Number of steps to take in traffic scenario.
        step_size (float, optional): Size of the steps.
        invalid_pos (int, optional): Check if vehicle is in an invalid location.
        warmup_period (int, optional): We start every episode one second in.
        allowed_goal_distance (float, optional): The distance to the goal position that 
            is considered as successfully reaching the goal.

    Returns: 
        collision_rate_vehicles (ndarray): Ratio of vehicles that collided with another vehicle.
        collision_rate_edges (ndarray): Ratio of vehicles that collided with a road edge.
        reached_goal_rate (ndarray): Ratio of vehicles that reached their goal.
    """
    stacked_state = defaultdict(lambda: None)
    
    # Create simulation from file
    #sim = Simulation(str(path_to_file), scenario_config)
    scenario = sim.getScenario()
    vehicles = scenario.getVehicles()
    objects_that_moved = scenario.getObjectsThatMoved()

    # Set all vehicles to expert control mode
    for obj in scenario.getVehicles():
        obj.expert_control = True

    # If a model is given, model will control vehicles that moved
    controlled_vehicles = [obj for obj in vehicles if obj in objects_that_moved]
    for veh in controlled_vehicles: veh.expert_control = False

    # Vehicles to check for collisions on
    objects_to_check = [
        obj for obj in controlled_vehicles if (obj.target_position - obj.position).norm() > 0.5
    ]

    collided_with_vehicle = {obj.id: False for obj in objects_to_check}
    collided_with_edge = {obj.id: False for obj in objects_to_check}
    reached_goal = {obj.id: False for obj in objects_to_check}
    
    # Step through the simulation 
    for time in range(num_steps):      
        for veh in controlled_vehicles:
            
            # Get the state for vehicle at timepoint
            state = construct_state(scenario, veh)

            # Stack state
            if stacked_state[veh.getID()] is None: 
                stacked_state[veh.getID()] = np.zeros(len(state) * num_stacked_states, dtype=state.dtype)
            # Add state to the end and convert to tensor
            stacked_state[veh.getID()] = np.roll(stacked_state[veh.getID()], len(state))
            stacked_state[veh.getID()][:len(state)] = state
            state_tensor = torch.Tensor(stacked_state[veh.getID()]).unsqueeze(0)

            # Pred actions
            actions, _ , _ = model(state_tensor)

            # Set vehicle actions (assuming we don't have head tilt)
            veh.acceleration = actions[0]
            veh.steering = actions[1]
            
        # Step the simulator and check for collision
        sim.step(step_size)

        # Once the warmup period is over                    
        if time > warmup_period:            
            for obj in objects_to_check:
                # Check for collisions
                if not np.isclose(obj.position.x, invalid_pos) and obj.collided: 
                    if int(obj.collision_type) == 1:
                        collided_with_vehicle[obj.id] = True
                    if int(obj.collision_type) == 2:
                        collided_with_edge[obj.id] = True   

                # Check if goal has been reached
                if (obj.target_position - obj.position).norm() < allowed_goal_distance:
                    reached_goal[obj.id] = True
        
    # Average
    collisions_with_vehicles = list(collided_with_vehicle.values())
    collisions_with_edges = list(collided_with_edge.values())
    collision_rate_vehicles = collisions_with_vehicles.count(True) / len(collisions_with_vehicles)
    collision_rate_edges = collisions_with_edges.count(True) / len(collisions_with_edges)

    reached_goal_values = list(reached_goal.values())
    reached_goal_rate = reached_goal_values.count(True) / len(reached_goal_values)

    return collision_rate_vehicles, collision_rate_edges, reached_goal_rate