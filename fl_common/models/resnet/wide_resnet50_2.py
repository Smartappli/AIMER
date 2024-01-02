from torchvision.models.resnet import wide_resnet50_2, Wide_ResNet50_2_Weights

# Initialize model
weights = Wide_ResNet50_2_Weights.DEFAULT
model = wide_resnet50_2(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
