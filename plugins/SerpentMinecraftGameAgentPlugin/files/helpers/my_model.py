from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam

import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as autoencoder


class model:
    def __init__(self, shape, control_shape):
        self.shape = shape
        self.control_shape = control_shape

    def evaluate(self, data, previous_control, previous_state):
        raise NotImplemented()

    def train_autoencoder(self, data):
        raise NotImplemented()

    def train_controller(self, hiddens, previous):
        raise NotImplemented()


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
        control = Dense(1024, name="Control")
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

