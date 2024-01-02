from torchvision.models.resnet import wide_resnet101_2, Wide_ResNet101_2_Weights

# Initialize model
weights = Wide_ResNet101_2_Weights.DEFAULT
model = wide_resnet101_2(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)