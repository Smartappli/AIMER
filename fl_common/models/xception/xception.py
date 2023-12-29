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
