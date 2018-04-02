from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Dropout, \
    BatchNormalization, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np

class my_model:

    def __init__(self):
        input_img = Input(shape=(128, 256, 3))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = AveragePooling2D((2, 2), padding='same')(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = AveragePooling2D((2, 2), padding='same')(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (16, 8, 8) i.e. 128-dimensional

        # x = BatchNormalization()(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = Dropout(0.1)(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(x)
        # x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()

        # opt = Adam()
        self.opt = SGD(lr=0.01, momentum=.9, clipvalue=0.5)
        autoencoder.compile(optimizer=self.opt, loss='binary_crossentropy')
        self.model = autoencoder

    def evaluate(self, data:np.ndarray)->np.ndarray:
        """

        Args:
            data:

        Returns:

        """
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1

        out = (self.model.predict(np.asarray([data])) * 256).astype('uint8')
        out = np.reshape(out, (128, 256, 3))
        return out


    def train(self, data):
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        data = np.asarray([data])
        self.model.train_on_batch(data, data)



