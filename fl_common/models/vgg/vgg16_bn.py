from torchvision.models.vgg import vgg16_bn, VGG16_BN_Weights

# Initialize model
weights = VGG16_BN_Weights.DEFAULT
model = vgg16_bn(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
