import random
from pathlib import Path

import h5py
import matplotlib.image
import numpy as np
import os

import re
import skimage.transform
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey, MouseButton
import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae
import traceback

shape = (64, 64, 3)
NUM_DIRS = 16
NUM_FRAMES = 16
BATCH_SIZE = 16

class SerpentMinecraftGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_pause
        self.model = ae.their_model(shape, 0.5)
        cwd = os.getcwd()
        print(cwd)

        self.paused = False
        self.count = 0
        self.my_datasets = Path(cwd + "/mydata/")
        p = self.my_datasets
        if p.exists():
            # get the highest count in the main directory
            all_imgs = p / 'all'
            max = 0
            for file in [x for x in all_imgs.glob('*.png')]:
                num = int(re.search('(\d*)', file).group(1))  # assuming filename is "filexxx.txt"
                # compare num to previous max, e.g.
                max = num if num > max else max  # set max = 0 before for-loop

            self.count = max
        else:
            # make all the paths
            print("creating directories")
            for ii in range(NUM_DIRS):
                dir = p / str(ii)
                dir.mkdir(parents=True, exist_ok=True)
        print(self.count)

        # set up the in-memory dataset
        self.dataset = [[None for _ in range(NUM_FRAMES) ] for _ in range(NUM_DIRS)]


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

        # get the highest file number
        max = 0
        for file in [x for x in dir.glob('*.png')]:
            num = int(re.search('data(\d*)', str(file.parts[-1])).group(1))
            max = num if num > max else max

        max += 1
        for set in self.dataset:
            for image in set:
                # if image doesn't exist, don't write it
                if image is None:
                    break
                filename = dir / ("data" + str(max) + ".png")
                matplotlib.image.imsave(filename, image)
                print("saving %s" % max)
                max += 1

        # f = h5py.File(filename, 'w')



    def setup_play(self):
        print("in setup")
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

    def handle_pause(self):
        # if first paused in a row, save the current dataset
        if self.paused:
            return
        print("in paused")

        self.paused = True
        # self.save_data()

    def handle_play(self, game_frame):
        """

        Args:
            game_frame (serpent.game_frame.GameFrame):

        Returns:

        """
        # serpent play Minecraft SerpentMinecraftGameAgent

        self.paused = False
        count = self.count
        self.count += 1

        frame = game_frame.frame

        resized = np.array(skimage.transform.resize(frame, shape[:-1], mode="reflect", order=1) * 255, dtype="uint8")

        # save the frame
        current_dir = (count // NUM_FRAMES) % NUM_DIRS
        current_frame = count % NUM_FRAMES
        self.dataset[current_dir][current_frame] = resized
        print(current_frame, current_dir)

        # if we're done with the batch, train
        if current_frame == 0:
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
            self.model.train_on_batch(batch)

            # save the dataset every so often
            # if current_dir == 0:
            #     self.save_data()

            self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)



        resized = np.array(skimage.transform.resize(frame, shape[:-1], mode="reflect", order=1) * 255, dtype="uint8")
        print(resized.shape)
        # self.model.train(resized)
        output = self.model.evaluate(resized)
        print(count)

        choice = random.randint(0, 5)


        self.visual_debugger.store_image_data(
            game_frame.frame,
            game_frame.frame.shape,
            "1"
        )
        self.visual_debugger.store_image_data(output, output.shape, "3")
        self.visual_debugger.store_image_data(resized, resized.shape, "2")

        # do outputs
        if choice == 0:
            self.input_controller.tap_key(KeyboardKey.KEY_W)
        elif choice == 1:
            self.input_controller.move(x=64, y=0, duration=0.05, absolute=False)
        # elif choice == 2:
        #     self.input_controller.move(x=0, y=64, duration=0.05, absolute=False)
        elif choice == 3:
            self.input_controller.move(x=64, y=0, duration=0.05, absolute=False)
        # elif choice == 4:
        #     self.input_controller.move(x=0, y=-64, duration=0.05, absolute=False)
        elif choice == 5:
            self.input_controller.move(x=-64, y=0, duration=0.05, absolute=False)

