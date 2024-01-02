from torchvision.models.vgg import vgg13, VGG13_Weights

# Initialize model
weights = VGG13_Weights.DEFAULT
model = vgg13(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)