import matplotlib.colors
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Dropout, \
    BatchNormalization, Conv2DTranspose, LeakyReLU, Flatten, Lambda, Reshape, regularizers
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K

import numpy as np


def build_variational(latent, in_layer):
    epsilon_std = 1.0
    x = Flatten()(in_layer)
    mu = Dense(latent)(x)
    sigma = Dense(latent)(x)

    # taken from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent,))([mu, sigma])
    return z, mu, sigma


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



def build_decoder(shape, drop, encoded, latent_size=1024):

    x = Dense(max(latent_size, 1024))(encoded)

    x = Reshape((1, 1, -1))(x)
    x = Conv2DTranspose(128, 5, padding='valid', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2DTranspose(64, 5, padding='valid', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2DTranspose(32, 6, padding='valid', strides=2, activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    decoded = Conv2DTranspose(3, 6, name="Decoded", padding='valid', strides=2, activation='sigmoid')(x)
    return decoded


def build_autoencoder(shape, latent=1024, drop=0.5):
    input_img = Input(shape=shape)
    encoded = build_encoder(shape, drop, input_img)
    variational = build_variational(latent, encoded)[0]
    decoded = build_decoder(shape, drop, variational)
    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    opt = Adam(lr=0.0001)
    # opt = SGD(lr=0.5, momentum=.9, clipvalue=0.5)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')

    return autoencoder

class Autoencoder:
    def __init__(self, shape):
        self.shape = shape
        self.model = None

    def evaluate(self, data:np.ndarray)->(np.ndarray, float):
        """

        Args:
            data:

        Returns:

        """
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        # data = matplotlib.colors.rgb_to_hsv(data)

        out = self.model.predict(np.asarray([data]))
        out = np.reshape(out, self.shape)
        loss = np.square(np.subtract(data, out)).mean()
        # out = matplotlib.colors.hsv_to_rgb(out)
        out = (out * 255).astype('uint8')
        return out, loss


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


class TheirAutoencoder(Autoencoder):

    def __init__(self, shape, drop=0.5):
        super(TheirAutoencoder, self).__init__(shape)
        self.model = build_autoencoder(shape, 32, drop)



class MyAutoencoder(Autoencoder):

    @staticmethod
    def build_decoder(shape, drop, encoded, latent_size=1024):

        x = Dense(max(latent_size, 1024))(encoded)
        reshaped = Reshape((1, 1, -1))(x)
        reshaped = Conv2DTranspose(128, 8, strides=1, padding='valid', name="upsampling")(reshaped)

        x = Conv2D(128, (3, 3), padding='same')(reshaped)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        x = UpSampling2D(size=2)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        x = UpSampling2D(size=(2, 2))(x)
        decoded = Conv2D(3, (3, 3),activation='sigmoid', padding='same')(x)

        return decoded

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

        variational = build_variational(1024, encoded)[0]

        # at this point the representation is (16, 8, 8) i.e. 128-dimensional
        x = Dense(1024)(variational)

        reshaped = Reshape((1, 1, -1))(x)


        reshaped = Conv2DTranspose(128, 8, strides=1, padding='valid', name="upsampling")(reshaped)


        # x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(reshaped)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        # x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
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
        # autoencoder.summary()
        self.model = autoencoder




