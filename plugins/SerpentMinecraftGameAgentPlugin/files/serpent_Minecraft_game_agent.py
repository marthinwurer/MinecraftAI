from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey


class SerpentMinecraftGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

    def handle_play(self, game_frame):
        """

        Args:
            game_frame (serpent.game_frame.GameFrame):

        Returns:

        """
        # serpent play Minecraft SerpentMinecraftGameAgent
        print(type(game_frame))


        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )

        self.input_controller.tap_key(KeyboardKey.KEY_W)
