import tensorflow as tf
import keras as k
import sklearn as sk
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import platform
import torch as tc

numgpu = len(tf.config.list_physical_devices('GPU'))
print("GPUs Available: " + str(
    numgpu) + " - Python: " + platform.python_version() + " - PyTorch: " + tc.__version__ + " - TensorFlow: " + tf.__version__ + " - Keras: " + k.__version__ + " - Numpy: " + np.version.version + " - Pandas: " + pd.__version__ + " - Sklearn: " + sk.__version__ + " - Seaborn: " + sns.__version__ + "  - Matplotlib: " + mpl.__version__)

model_id = 1
match model_id:
    case 1:  # Xception
        print("Xception")
        from tensorflow.keras.applications.xception import Xception

    case 2:  # VGG 11
        print("VGG 11")

    case 3:  # VGG 13
        print("VGG 13")

    case 4:  # VGG 16
        print("VGG 16")
        from tensorflow.keras.applications.vgg16 import VGG16

    case 5:  # VGG 19
        print("VGG 19")
        from tensorflow.keras.applications.vgg19 import VGG19

    case 6:  # ResNet 18
        print("ResNet 18")

    case 7:
        print("ResNet 34")

    case 8:
        print("ResNet 50")
        from tensorflow.keras.applications.resnet50 import ResNet50

    case 9:
        print("ResNet 50 V2")
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2

    case 10:
        print("ResNet RS 50")
        from tensorflow.keras.applications.resnet_rs import ResNetRS50

    case 11:
        print("ResNet 101")
        from tensorflow.keras.applications.resnet import ResNet101

    case 12:
        print("ResNet 101 V2")
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2

    case 13:
        print("ResNet RS 101")
        from tensorflow.keras.applications.resnet_rs import ResNetRS101

    case 14:
        print("ResNet 152")
        from tensorflow.keras.applications.resnet import ResNet152

    case 15:
        print("ResNet 152 V2")
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2

    case 16:
        print("ResNet RS 152")
        from tensorflow.keras.applications.resnet_rs import ResNetRS152

    case 17:
        print("ResNet RS 200")
        from tensorflow.keras.applications.resnet_rs import ResNetRS200

    case 18:
        print("ResNet RS 270")
        from tensorflow.keras.applications.resnet_rs import ResNetRS270

    case 19:
        print("ResNet RS 350")
        from tensorflow.keras.applications.resnet_rs import ResNetRS350

    case 20:
        print("ResNet RS 420")
        from tensorflow.keras.applications.resnet_rs import ResNetRS420

    case 21:
        print("Inception V3")
        from tensorflow.keras.applications.inception_v3 import InceptionV3

    case 22:
        print("Inception ResNet V2")
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

    case 23:
        print("MobileNet")
        from tensorflow.keras.applications.mobilenet import MobileNet

    case 24:
        print("MobileNet V2")
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    case 25:
        print("MobileNet V3 Small")
        from tensorflow.keras.applications import MobileNetV3Small

    case 26:
        print("MobileNet V3 Large")
        from tensorflow.keras.applications import MobileNetV3Large

    case 27:
        print("DenseNet 121")
        from tensorflow.keras.applications.densenet import DenseNet121

    case 28:
        print("DenseNet 169")
        from tensorflow.keras.applications.densenet import DenseNet169

    case 29:
        print("DenseNet 201")
        from tensorflow.keras.applications.densenet import DenseNet201

    case 30:
        print("NASNet Mobile")
        from tensorflow.keras.applications.nasnet import NASNetMobile

    case 31:
        print("NASNet Large")
        from tensorflow.keras.applications.nasnet import NASNetLarge

    case 32:
        print("EfficientNet B0")
        from tensorflow.keras.applications.efficientnet import EfficientNetB0

    case 33:
        print("EfficientNet B0 V2")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0

    case 34:
        print("EfficientNet B1")
        from tensorflow.keras.applications.efficientnet import EfficientNetB1

    case 35:
        print("EfficientNet B1 V2")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B1

    case 36:
        print("EfficientNet B2")
        from tensorflow.keras.applications.efficientnet import EfficientNetB2

    case 37:
        print("EfficientNet B2 V2")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2

    case 38:
        print("EfficientNet B3")
        from tensorflow.keras.applications.efficientnet import EfficientNetB3

    case 39:
        print("EfficientNet B3 V2")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3

    case 40:
        print("EfficientNet B4")
        from tensorflow.keras.applications.efficientnet import EfficientNetB4

    case 41:
        print("EfficientNet B5")
        from tensorflow.keras.applications.efficientnet import EfficientNetB5

    case 42:
        print("EfficientNet B6")
        from tensorflow.keras.applications.efficientnet import EfficientNetB6

    case 43:
        print("EfficientNet B7")
        from tensorflow.keras.applications.efficientnet import EfficientNetB7

    case 44:
        print("EfficientNet V2 Small")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S

    case 45:
        print("EfficientNet V2 Medium")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M

    case 46:
        print("EfficientNet V2 Large")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L

    case 47:
        print("ConvNeXt Tiny")
        from tensorflow.keras.applications.convnext import ConvNeXtTiny

    case 48:
        print("ConvNeXt Small")
        from tensorflow.keras.applications.convnext import ConvNeXtSmall

    case 49:
        print("ConvNeXt Medium")
        from tensorflow.keras.applications.convnext import ConvNeXtBase

    case 50:
        print("ConvNeXt Large")
        from tensorflow.keras.applications.convnext import ConvNeXtLarge

    case 51:
        print("ConvNeXt XLarge")
        from tensorflow.keras.applications.convnext import ConvNeXtXLarge

    case 52:
        print("RegNetX 002")
        from tensorflow.keras.applications.regnet import RegNetX002

    case 53:
        print("RegNetY 002")
        from tensorflow.keras.applications.regnet import RegNetY002

    case 54:
        print("RegNetX 004")
        from tensorflow.keras.applications.regnet import RegNetX004

    case 55:
        print("RegNetY 004")
        from tensorflow.keras.applications.regnet import RegNetY004

    case 56:
        print("RegNetX 006")
        from tensorflow.keras.applications.regnet import RegNetX006

    case 57:
        print("RegNetY 006")
        from tensorflow.keras.applications.regnet import RegNetY006

    case 58:
        print("RegNetX 008")
        from tensorflow.keras.applications.regnet import RegNetX008

    case 59:
        print("RegNetY 008")
        from tensorflow.keras.applications.regnet import RegNetY008

    case 60:
        print("RegNetX 016")
        from tensorflow.keras.applications.regnet import RegNetX016

    case 61:
        print("RegNetY 016")
        from tensorflow.keras.applications.regnet import RegNetY016

    case 62:
        print("RegNetX 032")
        from tensorflow.keras.applications.regnet import RegNetX032

    case 63:
        print("RegNetY 032")
        from tensorflow.keras.applications.regnet import RegNetY032

    case 64:
        print("RegNetX 040")
        from tensorflow.keras.applications.regnet import RegNetX040

    case 65:
        print("RegNetY 040")
        from tensorflow.keras.applications.regnet import RegNetY040

    case 66:
        print("RegNetX 064")
        from tensorflow.keras.applications.regnet import RegNetX064

    case 67:
        print("RegNetY 064")
        from tensorflow.keras.applications.regnet import RegNetY064

    case 68:
        print("RegNetX 080")
        from tensorflow.keras.applications.regnet import RegNetX080

    case 69:
        print("RegNetY 080")
        from tensorflow.keras.applications.regnet import RegNetY080

    case 70:
        print("RegNetX 120")
        from tensorflow.keras.applications.regnet import RegNetX120

    case 71:
        print("RegNetY 120")
        from tensorflow.keras.applications.regnet import RegNetY120

    case 72:
        print("RegNetX 160")
        from tensorflow.keras.applications.regnet import RegNetX160

    case 73:
        print("RegNetY 160")
        from tensorflow.keras.applications.regnet import RegNetY160

    case 74:
        print("RegNetX 320")
        from tensorflow.keras.applications.regnet import RegNetX320

    case 75:
        print("RegNetY 320")
        from tensorflow.keras.applications.regnet import RegNetY320

    case _:
        print("Error")
