from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Activation

import tensorflow as tf


class SimpleCNN(tf.keras.models.Model):
    def __init__(self, numclasses, chanDim=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(5, 5), padding='Same')
        self.act1 = Activation("relu")
        self.conv2 = Conv2D(filters=32, kernel_size=(5, 5), padding='Same')
        self.act2 = Activation("relu")
        self.mpool1 = MaxPool2D(pool_size=(2, 2))
        self.dpt1 = Dropout(0.5)

        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='Same')
        self.act3 = Activation("relu")
        self.conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='Same')
        self.act4 = Activation("relu")
        self.mpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.dpt2 = Dropout(0.5)

        self.flat = Flatten()
        self.dense1 = Dense(512)
        self.act5 = Activation("relu")
        self.dpt3 = Dropout(0.5)
        self.dense2 = Dense(numclasses)
        self.softmax = Activation("softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.mpool1(x)
        x = self.dpt1(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.mpool2(x)
        x = self.dpt2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.act5(x)
        x = self.dpt3(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x
