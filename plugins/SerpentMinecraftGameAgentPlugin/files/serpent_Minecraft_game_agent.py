import random
from pathlib import Path

import h5py
import matplotlib.image
import numpy as np
import os

import re

import scipy.ndimage.filters
import skimage.transform
from keras.losses import mean_squared_error
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey, MouseButton
import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae
import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.my_model as model
import traceback

shape = (64, 64, 3)
NUM_DIRS = 16
NUM_FRAMES = 16
BATCH_SIZE = 32
LATENT_SIZE = 1024
ACTION_SIZE = 5
MOVEMENT_SIZE = 128

class SerpentMinecraftGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_pause
        self.model = model.my_model(shape, ACTION_SIZE, LATENT_SIZE)
        cwd = os.getcwd()
        print(cwd)

        self.pause_debounce = False
        self.paused = False
        self.count = 0
        self.my_datasets = Path(cwd + "/mydata/")

        # set up dataset creation
        dire = self.my_datasets / "data"
        dire.mkdir(parents=True, exist_ok=True)
        max_img = 0
        for file in [x for x in dire.glob('*.png')]:
            num = int(re.search('data(\d*)', str(file.parts[-1])).group(1))
            max_img = num if num > max_img else max_img

        self.image_num = max_img

        print(self.count)

        # set up the in-memory dataset
        self.dataset = [[None for _ in range(NUM_FRAMES) ] for _ in range(NUM_DIRS)]

        # set up previous frame data
        self.prev_latent = np.random.rand(LATENT_SIZE)
        self.prev_choice = 0
        self.prev_loss = 0.5
        self.prev_control = np.random.rand(ACTION_SIZE)
        self.prev_action = np.zeros(ACTION_SIZE)
        self.prev_action[self.prev_choice] = 1.0

        print("init finished")
        # try:
        #     raise Exception("lolol")
        # except:
        #     traceback.print_stack()
        #     # exit(0)

    def save_data(self):
        # make the directory if it doesn't exist
        dir = self.my_datasets / "data"
        dir.mkdir(parents=True, exist_ok=True)
    #
    #     # get the highest file number
    #     max = 0
    #     for file in [x for x in dir.glob('*.png')]:
    #         num = int(re.search('data(\d*)', str(file.parts[-1])).group(1))
    #         max = num if num > max else max
    #
    #     max += 1
    #     for set in self.dataset:
    #         for image in set:
    #             # if image doesn't exist, don't write it
    #             if image is None:
    #                 break
    #             filename = dir / ("data" + str(max) + ".png")
    #             matplotlib.image.imsave(filename, image)
    #             print("saving %s" % max)
    #             max += 1
    #
    #     f = h5py.File(filename, 'w')



    def setup_play(self):
        print("in setup")
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        np.set_printoptions(precision=4)

    def handle_pause(self):
        # if first paused in a row, save the current dataset
        if self.pause_debounce:
            return
        else:
            self.pause_debounce = True

        print("in paused")
        comm = input("command> ")

        if comm == 'p':
            print("Paused control and training")
            self.paused = not self.paused
        elif comm == "s":
            dire = self.my_datasets/"model"
            dire.mkdir(parents=True, exist_ok=True)

            print("Saving weights")
            self.model.save_weights(dire)
        elif comm == "l":
            dire = self.my_datasets/"data"
            dire.mkdir(parents=True, exist_ok=True)

            print("loading weights")
            self.model.load_weights(dire)
        elif comm == 'q':
            exit(0)






        # self.save_data()

    def handle_play(self, game_frame):
        """

        Args:
            game_frame (serpent.game_frame.GameFrame):

        Returns:

        """
        # serpent play Minecraft SerpentMinecraftGameAgent

        # if the play is paused, don't do anything
        if self.paused:
            return

        self.pause_debounce = False
        count = self.count
        self.count += 1
        self.image_num += 1

        frame = game_frame.frame

        resized = skimage.transform.resize(frame, shape[:-1], mode="reflect", order=1)
        # resized = scipy.ndimage.filters.gaussian_filter(resized, (1,1,0))
        rgb = np.array(resized * 255, dtype="uint8")

        # save the frame
        current_dir = (count // NUM_FRAMES) % NUM_DIRS
        current_frame = count % NUM_FRAMES
        self.dataset[current_dir][current_frame] = resized
        # print(current_frame, current_dir)

        filename = self.my_datasets/ "data" / ("data" + str(self.image_num) + ".png")
        matplotlib.image.imsave(filename, resized)


        # if we're done with the batch, train
        if current_frame == 0:# and count > 255:
            # pause
            self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
            batch = []
            for dir in range(NUM_DIRS):
                for in_dir in range(BATCH_SIZE//NUM_DIRS):
                    # the 0th should always have something in it

                    data = random.choice(self.dataset[dir])
                    if data is None:
                        data = self.dataset[0][0]
                    batch.append(data)
            self.model.train_autoencoder(batch)

            # save the dataset every so often
            # if current_dir == 0:
            #     self.save_data()

            self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)





        # evaluate the neural net
        prev_action = np.zeros(ACTION_SIZE)
        prev_action[self.prev_choice] = 1.0
        outframe, encoded, control, loss = self.model.evaluate(resized, prev_action)


        # online training of the controller
        # set up actions
        actions = np.copy(self.prev_control)
        actions[self.prev_choice] = loss
        self.model.train_controller(self.prev_latent, self.prev_action, actions)

        if random.random() < 0.25:
            choice = random.randint(0, ACTION_SIZE-1)
        else:
            choice = np.argmax(control)


        print(rgb.shape, rgb.dtype)
        self.visual_debugger.store_image_data(
            game_frame.frame,
            game_frame.frame.shape,
            "1"
        )
        self.visual_debugger.store_image_data(outframe, outframe.shape, "3")
        self.visual_debugger.store_image_data(rgb, rgb.shape, "2")


        # print status
        print("%7s %2s %12s %s" % (count, choice, loss, actions))

        # do outputs
        if choice == 0:
            self.input_controller.tap_key(KeyboardKey.KEY_W)
        elif choice == 1:
            self.input_controller.move(x=0, y=MOVEMENT_SIZE, duration=0.05, absolute=False)
        elif choice == 2:
            self.input_controller.move(x=MOVEMENT_SIZE, y=0, duration=0.05, absolute=False)
        elif choice == 3:
            self.input_controller.move(x=0, y=-MOVEMENT_SIZE, duration=0.05, absolute=False)
        elif choice == 4:
            self.input_controller.move(x=-MOVEMENT_SIZE, y=0, duration=0.05, absolute=False)

        # store values for the next iteration's training
        self.prev_choice = choice
        self.prev_control = control
        self.prev_latent = encoded
        self.prev_action = prev_action

