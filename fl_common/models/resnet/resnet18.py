from torchvision.models.resnet import resnet18, ResNet18_Weights

# Initialize model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
