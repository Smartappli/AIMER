from torchvision.models.resnet import resnet152, ResNet152_Weights

# Initialize model
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)