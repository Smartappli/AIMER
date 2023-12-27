"""
Copyright (C) 2024  Olivier DEBAUCHE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import platform
import tensorflow as tf
import keras as k
import keras_tuner as kt
import sklearn as sk
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import torch as tc
from tensorflow.keras import layers

from fl_client.models import Queue
# from fl_client.models import Model, Model_File
# from fl_client.models import Dataset, Dataset_File
# from fl_client.models import Dataset_Central_Data, Dataset_Local_Data, Dataset_Remote_Data

numgpu = len(tf.config.list_physical_devices('GPU'))
print("GPUs Available: " + str(numgpu)
      + " - Python: " + platform.python_version()
      + " - PyTorch: " + tc.__version__
      + " - TensorFlow: " + tf.__version__
      + " - Keras: " + k.__version__
      + " - Numpy: " + np.version.version
      + " - Pandas: " + pd.__version__
      + " - Sklearn: " + sk.__version__
      + " - Seaborn: " + sns.__version__
      + "  - Matplotlib: " + mpl.__version__)

tasks = Queue.objects.get(queue_state='CR', queue_model_type='DLCL')
for task in tasks:
    model_id = task.queue_model_id
    dataset_id = task.queue_dataset_id
    params = task.queue_model.params

    # Begin temporary
    img_height = 224
    img_width = 224
    channel = 3
    classes = 4
    # End temporary

    match model_id:
        case 1:  # Xception
            print("Xception")
            from tensorflow.keras.applications.xception import Xception

            base_model = Xception(input_shape=(img_height, img_width, channel),
                                  include_top=False,
                                  weights='imagenet')

        case 2:  # VGG 11
            print("VGG 11")

        case 3:  # VGG 13
            print("VGG 13")

        case 4:  # VGG 16
            print("VGG 16")
            from tensorflow.keras.applications.vgg16 import VGG16
            base_model = VGG16(input_shape=(img_height, img_width, channel),
                               include_top=False,
                               weights='imagenet')

        case 5:  # VGG 19
            print("VGG 19")
            from tensorflow.keras.applications.vgg19 import VGG19
            base_model = VGG19(input_shape=(img_height, img_width, channel),
                               include_top=False,
                               weights='imagenet')

        case 6:  # ResNet 18
            print("ResNet 18")

        case 7:
            print("ResNet 34")

        case 8:
            print("ResNet 50")
            from tensorflow.keras.applications.resnet50 import ResNet50
            base_model = ResNet50(input_shape=(img_height, img_width, channel),
                                  include_top=False,
                                  weights='imagenet')

        case 9:
            print("ResNet 50 V2")
            from tensorflow.keras.applications.resnet_v2 import ResNet50V2
            base_model = ResNet50V2(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 10:
            print("ResNet RS 50")
            from tensorflow.keras.applications.resnet_rs import ResNetRS50
            base_model = ResNetRS50(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 11:
            print("ResNet 101")
            from tensorflow.keras.applications.resnet import ResNet101
            base_model = ResNet101(input_shape=(img_height, img_width, channel),
                                   include_top=False,
                                   weights='imagenet')

        case 12:
            print("ResNet 101 V2")
            from tensorflow.keras.applications.resnet_v2 import ResNet101V2
            base_model = ResNet101V2(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 13:
            print("ResNet RS 101")
            from tensorflow.keras.applications.resnet_rs import ResNetRS101
            base_model = ResNetRS101(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')
        case 14:
            print("ResNet 152")
            from tensorflow.keras.applications.resnet import ResNet152
            base_model = ResNet152(input_shape=(img_height, img_width, channel),
                                   include_top=False,
                                   weights='imagenet')

        case 15:
            print("ResNet 152 V2")
            from tensorflow.keras.applications.resnet_v2 import ResNet152V2
            base_model = ResNet152V2(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 16:
            print("ResNet RS 152")
            from tensorflow.keras.applications.resnet_rs import ResNetRS152
            base_model = ResNetRS152(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 17:
            print("ResNet RS 200")
            from tensorflow.keras.applications.resnet_rs import ResNetRS200
            base_model = ResNetRS200(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 18:
            print("ResNet RS 270")
            from tensorflow.keras.applications.resnet_rs import ResNetRS270
            base_model = ResNetRS270(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 19:
            print("ResNet RS 350")
            from tensorflow.keras.applications.resnet_rs import ResNetRS350
            base_model = ResNetRS350(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 20:
            print("ResNet RS 420")
            from tensorflow.keras.applications.resnet_rs import ResNetRS420
            base_model = ResNetRS420(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 21:
            print("Inception V3")
            from tensorflow.keras.applications.inception_v3 import InceptionV3
            base_model = InceptionV3(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')
        case 22:
            print("Inception ResNet V2")
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
            base_model = InceptionResNetV2(input_shape=(img_height, img_width, channel),
                                           include_top=False,
                                           weights='imagenet')

        case 23:
            print("MobileNet")
            from tensorflow.keras.applications.mobilenet import MobileNet
            base_model = MobileNet(input_shape=(img_height, img_width, channel),
                                   include_top=False,
                                   weights='imagenet')

        case 24:
            print("MobileNet V2")
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            base_model = MobileNetV2(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 25:
            print("MobileNet V3 Small")
            from tensorflow.keras.applications import MobileNetV3Small
            base_model = MobileNetV3Small(input_shape=(img_height, img_width, channel),
                                          include_top=False,
                                          weights='imagenet')

        case 26:
            print("MobileNet V3 Large")
            from tensorflow.keras.applications import MobileNetV3Large
            base_model = MobileNetV3Large(input_shape=(img_height, img_width, channel),
                                          include_top=False,
                                          weights='imagenet')

        case 27:
            print("DenseNet 121")
            from tensorflow.keras.applications.densenet import DenseNet121
            base_model = DenseNet121(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 28:
            print("DenseNet 169")
            from tensorflow.keras.applications.densenet import DenseNet169
            base_model = DenseNet169(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 29:
            print("DenseNet 201")
            from tensorflow.keras.applications.densenet import DenseNet201
            base_model = DenseNet201(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 30:
            print("NASNet Mobile")
            from tensorflow.keras.applications.nasnet import NASNetMobile
            base_model = NASNetMobile(input_shape=(img_height, img_width, channel),
                                      include_top=False,
                                      weights='imagenet')

        case 31:
            print("NASNet Large")
            from tensorflow.keras.applications.nasnet import NASNetLarge
            base_model = NASNetLarge(input_shape=(img_height, img_width, channel),
                                     include_top=False,
                                     weights='imagenet')

        case 32:
            print("EfficientNet B0")
            from tensorflow.keras.applications.efficientnet import EfficientNetB0
            base_model = EfficientNetB0(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 33:
            print("EfficientNet B0 V2")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
            base_model = EfficientNetV2B0(input_shape=(img_height, img_width, channel),
                                          include_top=False,
                                          weights='imagenet')

        case 34:
            print("EfficientNet B1")
            from tensorflow.keras.applications.efficientnet import EfficientNetB1
            base_model = EfficientNetB1(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 35:
            print("EfficientNet B1 V2")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B1
            base_model = EfficientNetV2B1(input_shape=(img_height, img_width, channel),
                                          include_top=False,
                                          weights='imagenet')

        case 36:
            print("EfficientNet B2")
            from tensorflow.keras.applications.efficientnet import EfficientNetB2
            base_model = EfficientNetB2(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 37:
            print("EfficientNet B2 V2")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2
            base_model = EfficientNetV2B2(input_shape=(img_height, img_width, channel),
                                          include_top=False,
                                          weights='imagenet')

        case 38:
            print("EfficientNet B3")
            from tensorflow.keras.applications.efficientnet import EfficientNetB3
            base_model = EfficientNetB3(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 39:
            print("EfficientNet B3 V2")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
            base_model = EfficientNetV2B3(input_shape=(img_height, img_width, channel),
                                          include_top=False,
                                          weights='imagenet')

        case 40:
            print("EfficientNet B4")
            from tensorflow.keras.applications.efficientnet import EfficientNetB4
            base_model = EfficientNetB4(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 41:
            print("EfficientNet B5")
            from tensorflow.keras.applications.efficientnet import EfficientNetB5
            base_model = EfficientNetB5(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')
        case 42:
            print("EfficientNet B6")
            from tensorflow.keras.applications.efficientnet import EfficientNetB6
            base_model = EfficientNetB6(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 43:
            print("EfficientNet B7")
            from tensorflow.keras.applications.efficientnet import EfficientNetB7
            base_model = EfficientNetB7(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 44:
            print("EfficientNet V2 Small")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
            base_model = EfficientNetV2S(input_shape=(img_height, img_width, channel),
                                         include_top=False,
                                         weights='imagenet')

        case 45:
            print("EfficientNet V2 Medium")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
            base_model = EfficientNetV2M(input_shape=(img_height, img_width, channel),
                                         include_top=False,
                                         weights='imagenet')

        case 46:
            print("EfficientNet V2 Large")
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
            base_model = EfficientNetV2L(input_shape=(img_height, img_width, channel),
                                         include_top=False,
                                         weights='imagenet')

        case 47:
            print("ConvNeXt Tiny")
            from tensorflow.keras.applications.convnext import ConvNeXtTiny
            base_model = ConvNeXtTiny(input_shape=(img_height, img_width, channel),
                                      include_top=False,
                                      weights='imagenet')

        case 48:
            print("ConvNeXt Small")
            from tensorflow.keras.applications.convnext import ConvNeXtSmall
            base_model = ConvNeXtSmall(input_shape=(img_height, img_width, channel),
                                       include_top=False,
                                       weights='imagenet')

        case 49:
            print("ConvNeXt Medium")
            from tensorflow.keras.applications.convnext import ConvNeXtBase
            base_model = ConvNeXtBase(input_shape=(img_height, img_width, channel),
                                      include_top=False,
                                      weights='imagenet')

        case 50:
            print("ConvNeXt Large")
            from tensorflow.keras.applications.convnext import ConvNeXtLarge
            base_model = ConvNeXtLarge(input_shape=(img_height, img_width, channel),
                                       include_top=False,
                                       weights='imagenet')

        case 51:
            print("ConvNeXt XLarge")
            from tensorflow.keras.applications.convnext import ConvNeXtXLarge
            base_model = ConvNeXtXLarge(input_shape=(img_height, img_width, channel),
                                        include_top=False,
                                        weights='imagenet')

        case 52:
            print("RegNetX 002")
            from tensorflow.keras.applications.regnet import RegNetX002
            base_model = RegNetX002(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 53:
            print("RegNetY 002")
            from tensorflow.keras.applications.regnet import RegNetY002
            base_model = RegNetY002(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 54:
            print("RegNetX 004")
            from tensorflow.keras.applications.regnet import RegNetX004
            base_model = RegNetX004(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 55:
            print("RegNetY 004")
            from tensorflow.keras.applications.regnet import RegNetY004
            base_model = RegNetY004(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 56:
            print("RegNetX 006")
            from tensorflow.keras.applications.regnet import RegNetX006
            base_model = RegNetX006(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')
        case 57:
            print("RegNetY 006")
            from tensorflow.keras.applications.regnet import RegNetY006
            base_model = RegNetY006(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 58:
            print("RegNetX 008")
            from tensorflow.keras.applications.regnet import RegNetX008
            base_model = RegNetX008(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 59:
            print("RegNetY 008")
            from tensorflow.keras.applications.regnet import RegNetY008
            base_model = RegNetY008(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 60:
            print("RegNetX 016")
            from tensorflow.keras.applications.regnet import RegNetX016
            base_model = RegNetX016(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 61:
            print("RegNetY 016")
            from tensorflow.keras.applications.regnet import RegNetY016
            base_model = RegNetY016(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 62:
            print("RegNetX 032")
            from tensorflow.keras.applications.regnet import RegNetX032
            base_model = RegNetX032(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 63:
            print("RegNetY 032")
            from tensorflow.keras.applications.regnet import RegNetY032
            base_model = RegNetY032(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')
        case 64:
            print("RegNetX 040")
            from tensorflow.keras.applications.regnet import RegNetX040
            base_model = RegNetX040(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 65:
            print("RegNetY 040")
            from tensorflow.keras.applications.regnet import RegNetY040
            base_model = RegNetY040(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 66:
            print("RegNetX 064")
            from tensorflow.keras.applications.regnet import RegNetX064
            base_model = RegNetX064(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 67:
            print("RegNetY 064")
            from tensorflow.keras.applications.regnet import RegNetY064
            base_model = RegNetY064(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 68:
            print("RegNetX 080")
            from tensorflow.keras.applications.regnet import RegNetX080
            base_model = RegNetX080(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 69:
            print("RegNetY 080")
            from tensorflow.keras.applications.regnet import RegNetY080
            base_model = RegNetY080(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 70:
            print("RegNetX 120")
            from tensorflow.keras.applications.regnet import RegNetX120
            base_model = RegNetX120(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 71:
            print("RegNetY 120")
            from tensorflow.keras.applications.regnet import RegNetY120
            base_model = RegNetY120(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 72:
            print("RegNetX 160")
            from tensorflow.keras.applications.regnet import RegNetX160
            base_model = RegNetX160(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 73:
            print("RegNetY 160")
            from tensorflow.keras.applications.regnet import RegNetY160
            base_model = RegNetY160(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 74:
            print("RegNetX 320")
            from tensorflow.keras.applications.regnet import RegNetX320
            base_model = RegNetX320(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case 75:
            print("RegNetY 320")
            from tensorflow.keras.applications.regnet import RegNetY320
            base_model = RegNetY320(input_shape=(img_height, img_width, channel),
                                    include_top=False,
                                    weights='imagenet')

        case _:
            print("Error")

    base_model.trainable = False

    hp = kt.HyperParameters()

    inputs = tf.keras.Input(shape=(img_height, img_width, channel))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(classes)(x)
    model = tf.keras.Model(inputs, outputs)
