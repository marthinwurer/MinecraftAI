from keras import Input, Model
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam
import numpy as np

import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as autoencoder

def wrap(arr):
    return np.asarray([arr])

class model:
    def __init__(self, shape, control_shape):
        self.ae = None
        self.full = None
        self.c_train = None
        self.shape = shape
        self.control_shape = control_shape

    def evaluate(self, frame, prev_action):

        frame = frame.astype('float32') / 255. # Make sure that the data is between 0 and 1
        frame = wrap(frame)
        prev_action = wrap(prev_action)
        out = self.full.predict([frame, prev_action])
        decoded = out[0]
        encoded = out[1]
        control = out[2]
        # print(type(out), out)
        decoded = np.reshape(decoded, self.shape)
        encoded = np.reshape(encoded, encoded.shape[1:])
        control = np.reshape(control, control.shape[1:])
        loss = np.square(np.subtract(frame, decoded)).mean()
        decoded = (decoded * 255).astype('uint8')
        # for i in out:
        #     print(type(i), i)
        return decoded, encoded, control, loss

    def train_autoencoder(self, data):
        data = np.asarray(data)
        data = data.astype('float32')/255. # Make sure that the data is between 0 and 1
        output = self.ae.train_on_batch(data, data)
        return output

    def train_controller(self, latent, prev_action, actions):
        latent = wrap(latent)
        prev_action = wrap(prev_action)
        actions = wrap(actions)
        output = self.c_train.train_on_batch([prev_action, latent], actions)
        return output


class my_model(model):
    def __init__(self, shape, action_shape, latent):
        super().__init__(shape, action_shape)

        # define the VAE layers
        input_img = Input(shape=shape)
        encoded = autoencoder.build_encoder(shape, 0.1, input_img)
        variational = autoencoder.build_variational(latent, encoded)[0]
        decoded = autoencoder.MyAutoencoder.build_decoder(shape, 0.1, variational, latent)
        # decoded = autoencoder.build_decoder(shape, 0.1, variational)

        # define the ae model
        ae = Model(input_img, decoded)
        ae.summary()
        opt = Adam(lr=0.0001)
        # opt = SGD(lr=0.5, momentum=.9, clipvalue=0.5)
        ae.compile(optimizer=opt, loss='mean_squared_error')
        self.ae = ae

        # define the model layers
        prev_action = Input((action_shape,), name="Prev_Action")

        # define the control layers
        control_input = Concatenate()
        control = Dense(1024, activation='elu', name="Control")
        actions = Dense(action_shape, name="Actions")

        # define the full model
        full_input = control_input([prev_action, variational])
        full_control = control(full_input)
        full_actions = actions(full_control)
        full = Model([input_img, prev_action], [decoded, variational, full_actions])
        full.summary()
        full.compile(opt, loss='mean_squared_error')
        self.full = full

        # define the controller training model
        latent_input = Input((latent,))
        train_input = control_input([prev_action, latent_input])
        control_train = control(train_input)
        train_actions = actions(control_train)
        c_train = Model([prev_action, latent_input], train_actions)
        c_train.summary()
        c_train.compile(opt, loss='mean_squared_error')
        self.c_train = c_train

