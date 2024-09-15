# Test script to collect NVIDIA GPU information using PyTorch

import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
        print(f"  CUDA Cores: {torch.cuda.get_device_properties(i).multi_processor_count * 64}") # Assuming 64 cores per SM
else:
    print("CUDA is not available.")

# Data on ORFEO:
# Device 0: Tesla V100-PCIE-32GB
#   CUDA Capability: (7, 0)
#   Total Memory: 31.73 GB
#   CUDA Cores: 5120
