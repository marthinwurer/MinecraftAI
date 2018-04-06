import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae
import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.my_model as model
import numpy as np

# shape = (64, 128, 3)
shape = (64, 64, 3)
# ae.build_autoencoder(shape, 0.5)
# ae.my_autoencoder(shape)
m = model.my_model(shape, 5, 1024)

print(m.evaluate(np.zeros(shape, 'uint8')))


