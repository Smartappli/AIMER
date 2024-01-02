from torchvision.models.resnet import resnet34, ResNet34_Weights

# Initialize model
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
