import pdb
import yaml
from base_env import BaseEnv
from cfgs.config import get_scenario_dict
from nocturne import Action, Simulation
import os

if __name__ == "__main__":

    # test
    config_path = "experiments/human_regularized/rl_config.yaml"

    file = '/scratch/dc4971/nocturne/data/formatted_json_v2_no_tl_train/example_scenario.json'
    #file = 'data/formatted_json_v2_no_tl_train/tfrecord-00007-of-01000_237.json'

    with open(config_path, "r") as stream:
        base_env_config = yaml.safe_load(stream)
    
    print(os.path.isfile(file))

    scenario_config = {
        'start_time': 0, # When to start the simulation
        'allow_non_vehicles': True, # Whether to include cyclists and pedestrians 
        'max_visible_road_points': 10, # Maximum number of road points for a vehicle
        'max_visible_objects': 10, # Maximum number of road objects for a vehicle
        'max_visible_traffic_lights': 10, # Maximum number of traffic lights in constructed view
        'max_visible_stop_signs': 10, # Maximum number of stop signs in constructed view
    }
    
    print('create simulation:')
    # Make simulation
    sim = Simulation(
        file,
        scenario_config
    )

    print('create rl environment:')
    env = BaseEnv(base_env_config)
    
    
    #env.seed(seed)