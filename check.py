import torch

data=torch.load(
    "context_irony_model_epoch5.pth",
    map_location="cpu"
)

print(type(data))

if isinstance(data,dict):

    print(data.keys())