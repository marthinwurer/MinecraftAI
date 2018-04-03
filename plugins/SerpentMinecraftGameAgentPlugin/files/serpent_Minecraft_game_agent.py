import random

import numpy as np
import os
import skimage.transform
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey, MouseButton
import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae
import traceback

shape = (64, 64, 3)

class SerpentMinecraftGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.model = ae.my_model(shape, 0.5)
        cwd = os.getcwd()
        print(cwd)
        self.count = 0
        print("init finished")
        # try:
        #     raise Exception("lolol")
        # except:
        #     traceback.print_stack()
        #     # exit(0)


    def setup_play(self):
        print("in setup")
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

    def handle_play(self, game_frame):
        """

        Args:
            game_frame (serpent.game_frame.GameFrame):

        Returns:

        """
        # serpent play Minecraft SerpentMinecraftGameAgent
        # print(type(game_frame))
        # print(game_frame.frame.shape)
        # print(self.game.window_geometry["x_offset"])
        # print(self.game.window_geometry["y_offset"])
        frame = game_frame.frame

        resized = np.array(skimage.transform.resize(frame, shape[:-1], mode="reflect", order=1) * 255, dtype="uint8")
        print(resized.shape)
        # print(resized.shape)
        self.model.train(resized)
        output = self.model.evaluate(resized)
        self.count += 1
        print(self.count)

        choice = random.randint(0, 5)


        self.visual_debugger.store_image_data(
            game_frame.frame,
            game_frame.frame.shape,
            "1"
        )
        self.visual_debugger.store_image_data(output, output.shape, "2")
        self.visual_debugger.store_image_data(resized, resized.shape, "3")

        # do outputs
        if choice == 0:
            self.input_controller.tap_key(KeyboardKey.KEY_W)
        elif choice == 1:
            self.input_controller.tap_key(KeyboardKey.KEY_W)
        elif choice == 2:
            self.input_controller.move(x=0, y=32, duration=0.05, absolute=False)
        elif choice == 3:
            self.input_controller.move(x=32, y=0, duration=0.05, absolute=False)
        elif choice == 4:
            self.input_controller.move(x=0, y=-32, duration=0.05, absolute=False)
        elif choice == 5:
            self.input_controller.move(x=-32, y=0, duration=0.05, absolute=False)

