from torchvision.models.resnet import resnext101_32x8d, ResNeXt101_32X8D_Weights

# Initialize model
weights = ResNeXt101_32X8D_Weights.DEFAULT
model = resnext101_32x8d(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
