from torchvision.models.resnet import resnet101, ResNet101_Weights

# Initialize model
weights = ResNet101_Weights.DEFAULT
model = resnet101(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
