import timm

# Get the list of all available models in timm
all_models = timm.list_models(pretrained=True)

# Print the names of models pretrained on ImageNet
print("Models pretrained:")
print(len(all_models))