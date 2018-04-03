from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Dropout, \
    BatchNormalization, Conv2DTranspose, LeakyReLU, Concatenate, multiply, add
from keras.activations import tanh
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np


class LSTM_layer:

    def __init__(self, size, hidden, state, input):

        merged = Concatenate()([hidden, input])
        forget = Dense(size, activation='sigmoid')(merged)
        i = Dense(size, activation='sigmoid')(merged)
        C = Dense(size, activation='tanh')(merged)
        o = Dense(size, activation='sigmoid')(merged)
        ct = multiply([state, forget])
        ci = multiply([i, C])
        ct = add([ci, ct])
        ht = multiply([o, tanh(ct)])

        self.state_out = ct
        self.hidden_out = ht





