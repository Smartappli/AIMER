# import keras_tuner as kt
from keras.applications.xception import Xception


class ModelXception():
    """Class representing a xception model"""

    def __init__(self):
        """Initialize the xception model"""
        self.model = "xception"
        self.input_shape = (224, 224, 3)
        self.num_classes = 3
        self.dropout_rate = 0.5
        self.weight_decay = 1e-4
        self.epochs = 20
        self.batch_size = 32
        self.optimizer = "adam"
        self.learning_rate = 0.001
        self.momentum = 0.9

        base_model = Xception(input_shape=self.input_shape,
                              include_top=False,
                              weights='imagenet')


if __name__ == '__main__':
    return_val = ModelXception()
    # try:
    #    arg = sys.argv[1]
    # except IndexError:
    #    arg = None

    # return_val = ModelXception(arg)
