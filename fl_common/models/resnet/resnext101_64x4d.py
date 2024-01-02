from torchvision.models.resnet import resnext101_64x4d, ResNeXt101_64X4D_Weights

# Initialize model
weights = ResNeXt101_64X4D_Weights.DEFAULT
model = resnext101_64x4d(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
