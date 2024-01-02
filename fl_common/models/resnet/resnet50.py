from torchvision.models.resnet import resnet50, ResNet50_Weights

# Initialize model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
