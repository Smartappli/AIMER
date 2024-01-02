from torchvision.models.vgg import vgg11, VGG11_Weights

# Initialize model
weights = VGG11_Weights.DEFAULT
model = vgg11(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)

