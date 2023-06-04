from torch.utils.data import Dataset
from nocturne import Simulation

class SimulationDataset(Dataset):
    """
    This dataset loads a traffic scene from a given file path and converts 
    it into a Nocturne simulation object.
    
    Args:
        paths (list): A list of file paths to the traffic scene files.
    Returns:
        Simulation: A Nocturne simulation object representing the loaded traffic scene.
    """
    def __init__(self, paths, scenario_config):
        self.paths = paths
        self.scenario_config = scenario_config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Retrieves the simulation object at the specified index.
        Args:
            idx (int): The index of the simulation object to retrieve.
        Returns:
            Simulation: A Nocturne simulation object representing the loaded traffic scene.
        """
        path = self.paths[idx]
        sim = Simulation(str(path), self.scenario_config)
        return sim
    

def simulation_collate_fn(batch):
    """
    Collate function for handling batches of Simulation objects: returns the 
    batch of Simulation objects as-is.
    Args:
        batch (list): List of Simulation objects.
    Returns:
        list: The batch of Simulation objects.
    """
    return batch  
