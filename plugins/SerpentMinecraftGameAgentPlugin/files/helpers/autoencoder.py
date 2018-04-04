import matplotlib.colors
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Dropout, \
    BatchNormalization, Conv2DTranspose, LeakyReLU, Flatten, Lambda, Reshape, regularizers
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K

import numpy as np


def build_variational(latent, inlayer):
    epsilon_std = 1.0
    x = Flatten()(inlayer)
    mu = Dense(latent, kernel_regularizer=regularizers.l2(0.01))(x)
    sigma = Dense(latent, kernel_regularizer=regularizers.l2(0.01))(x)

    # taken from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent,))([mu, sigma])
    encoded = z
    return encoded


def build_encoder(shape, drop, input_img):
    x = Conv2D(32, 4, padding='valid', strides=2, activation='elu')(input_img)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, 4, padding='valid', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2D(128, 4, padding='valid', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2D(256, 4, padding='valid', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    return x



def build_decoder(shape, drop, encoded):

    x = Dense(1024)(encoded)

    x = Reshape((1, 1, -1))(x)
    x = Conv2DTranspose(128, 5, padding='same', strides=4, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2DTranspose(64, 5, padding='same', strides=4, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2DTranspose(32, 6, padding='same', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    decoded = Conv2DTranspose(3, 6, padding='same', strides=2, activation='sigmoid')(x)
    return decoded


def build_autoencoder(shape, drop=0.5):
    latent_dim = 1024
    input_img = Input(shape=shape)
    encoded = build_encoder(shape, drop, input_img)
    variational = build_variational(latent_dim, encoded)
    decoded = build_decoder(shape, drop, variational)
    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    opt = Adam(lr=0.0001)
    # opt = SGD(lr=0.5, momentum=.9, clipvalue=0.5)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')

    return autoencoder

def evaluate(model, data:np.ndarray)->np.ndarray:
    data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
    # data = matplotlib.colors.rgb_to_hsv(data)

    out = model.predict(np.asarray([data]))
    # out = matplotlib.colors.hsv_to_rgb(out)
    out = (out * 256).astype('uint8')
    out = np.reshape(out, (128, 256, 3))
    return out

def train(model, data):
    data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
    data = np.asarray([data])
    # data = matplotlib.colors.rgb_to_hsv(data)
    output = model.train_on_batch(data, data)
    print(output)

class ae_model:
    def __init__(self, shape):
        self.shape = shape
        self.model = None

    def evaluate(self, data:np.ndarray)->np.ndarray:
        """

        Args:
            data:

        Returns:

        """
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        # data = matplotlib.colors.rgb_to_hsv(data)

        out = self.model.predict(np.asarray([data]))
        # out = matplotlib.colors.hsv_to_rgb(out)
        out = (out * 255).astype('uint8')
        out = np.reshape(out, self.shape)
        return out


    def train(self, data):
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        data = np.asarray([data])
        # data = matplotlib.colors.rgb_to_hsv(data)
        output = self.model.train_on_batch(data, data)
        print(output)

    def train_on_batch(self, data):
        data = np.asarray(data)
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        output = self.model.train_on_batch(data, data)
        print(output)


class their_model(ae_model):

    def __init__(self, shape, drop=0.5):
        super(their_model, self).__init__(shape)
        self.model = build_autoencoder(shape, drop)



class my_model(ae_model):

    def __init__(self, shape, drop=0.5):
        super().__init__(shape)
        input_img = Input(shape=shape)
        x = Conv2D(32, (3, 3), padding='same')(input_img)
        x = LeakyReLU()(x)
        x = AveragePooling2D((2, 2), padding='same')(x)
        # x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = AveragePooling2D((2, 2), padding='same')(x)
        # x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        variational = build_variational(1024, encoded)

        # at this point the representation is (16, 8, 8) i.e. 128-dimensional
        x = Dense(1024)(variational)

        reshaped = Reshape((1, 1, -1))(x)


        reshaped = Conv2DTranspose(8, (8, 8), strides=8, padding='same', name="upsampling")(reshaped)


        # x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(reshaped)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = UpSampling2D(size=2)(x)
        # x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        # x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3),activation='sigmoid', padding='same')(x)
        # x = Conv2D(3, (3, 3), padding='same')(x)
        # decoded = LeakyReLU()(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()

        self.opt = Adam()
        # self.opt = SGD(lr=0.25, momentum=.9, clipvalue=0.5)
        autoencoder.compile(optimizer=self.opt, loss='mean_squared_error')
        self.model = autoencoder




