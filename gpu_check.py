import torch

# torch gpu check
print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
torch.cuda.set_device(2)
print("CUDA device name: ", torch.cuda.get_device_name())
print("CUDA current device: ", torch.cuda.current_device())
