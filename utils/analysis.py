# Utilities functions to conduct model analysis
# TODO: complete functions to visualize layers and add typing hints

# Imports ----------------------------------------------------------------------

# Torch imports
import torch as th


# Parameters counter -----------------------------------------------------------
def count_parameters(model: th.nn.Module) -> int:
    """Function to count the total number of parameters of a model
    
    Parameters
    ----------
    model: th.nn.Module
        Model to analyze
        
    Returns
    -------
    int
        Number of parameters of the model
    """
    return sum(p.numel() for p in model.parameters())