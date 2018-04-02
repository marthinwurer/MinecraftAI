import matplotlib.colors
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Dropout, \
    BatchNormalization, Conv2DTranspose, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np

class my_model:

    def __init__(self):
        drop = 0.5
        input_img = Input(shape=(128, 256, 3))
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

        # at this point the representation is (16, 8, 8) i.e. 128-dimensional

        # x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(encoded)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(drop)(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        decoded = LeakyReLU()(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()

        self.opt = Adam()
        # self.opt = SGD(lr=0.25, momentum=.9, clipvalue=0.5)
        autoencoder.compile(optimizer=self.opt, loss='mean_squared_error')
        self.model = autoencoder

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
        out = (out * 256).astype('uint8')
        out = np.reshape(out, (128, 256, 3))
        return out


    def train(self, data):
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        data = np.asarray([data])
        # data = matplotlib.colors.rgb_to_hsv(data)
        output = self.model.train_on_batch(data, data)
        print(output)



