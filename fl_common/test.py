import torch
entrypoints = torch.hub.list('pytorch/vision', force_reload=True, skip_validation=False, trust_repo=True)
