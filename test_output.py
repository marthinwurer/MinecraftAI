import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae
import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.my_model as model
import numpy as np

# shape = (64, 128, 3)
shape = (64, 64, 3)
actions = 5
# ae.build_autoencoder(shape, 1024, 0.5)
# ae.MyAutoencoder(shape)
m = model.my_model(shape, actions, 1024)

prev_action = np.zeros(actions)
prev_action[0] = 1.0
print(m.evaluate(np.zeros(shape, 'uint8'), prev_action))


