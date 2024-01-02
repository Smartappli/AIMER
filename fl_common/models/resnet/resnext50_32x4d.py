import torch
from torchvision.models.resnet import resnext50_32x4d, ResNeXt50_32X4D_Weights

# Initialize model
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
# img_transformed = preprocess(img)

# Set model to train mode
model.train(True)
