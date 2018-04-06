from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as autoencoder


class model:
    def __init__(self, shape, control_shape):
        self.ae = None
        self.full = None
        self.c_train = None
        self.shape = shape
        self.control_shape = control_shape

    def evaluate(self, data):
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        out = self.full.predict(np.asarray([data]))
        decoded = out[0]
        encoded = out[1]
        control = out[2]
        # print(type(out), out)
        decoded = np.reshape(decoded, self.shape)
        encoded = np.reshape(encoded, encoded.shape[1:])
        control = np.reshape(control, control.shape[1:])
        loss = np.square(np.subtract(data, decoded)).mean()
        decoded = (decoded * 255).astype('uint8')
        # for i in out:
        #     print(type(i), i)
        return decoded, encoded, control, loss

    def train_autoencoder(self, data):
        data = np.asarray(data)
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        output = self.ae.train_on_batch(data, data)
        return output

    def train_controller(self, latent, actions):
        latent = np.asarray(latent)
        output = self.c_train.train_on_batch(latent, np.asarray(actions))
        return output


class my_model(model):
    def __init__(self, shape, control_shape, latent):
        super().__init__(shape, control_shape)

        # define the VAE layers
        input_img = Input(shape=shape)
        encoded = autoencoder.build_encoder(shape, 0.1, input_img)
        variational = autoencoder.build_variational(latent, encoded)[0]
        decoded = autoencoder.build_decoder(shape, 0.1, variational)

        # define the ae model
        ae = Model(input_img, decoded)
        ae.summary()
        opt = Adam(lr=0.0001)
        # opt = SGD(lr=0.5, momentum=.9, clipvalue=0.5)
        ae.compile(optimizer=opt, loss='mean_squared_error')
        self.ae = ae

        # define the model layers
        # to be done later

        # define the control layers
        control = Dense(1024, activation='elu', name="Control")
        control_outputs = Dense(control_shape, name="Actions")

        # define the full model
        control_through = control(variational)
        co_through = control_outputs(control_through)
        full = Model(input_img, [decoded, variational, co_through])
        full.summary()
        full.compile(opt, loss='mean_squared_error')
        self.full = full

        # define the controller training model
        control_input = Input((latent,))
        control_train = control(control_input)
        co_train = control_outputs(control_train)
        c_train = Model(control_input, co_train)
        c_train.summary()
        c_train.compile(opt, loss='mean_squared_error')
        self.c_train = c_train

