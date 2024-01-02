from torchvision.models.vgg import vgg16, VGG16_Weights

# Initialize model
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
