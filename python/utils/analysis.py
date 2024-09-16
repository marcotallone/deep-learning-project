# Utilities functions to conduct model analysis
# TODO: complete functions to visualize layers and add typing hints

# Imports ----------------------------------------------------------------------

# Parameters counter -----------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())