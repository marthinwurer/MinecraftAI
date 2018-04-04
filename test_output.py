import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae

# shape = (64, 128, 3)
shape = (64, 64, 3)
# ae.build_autoencoder(shape, 0.5)
ae.my_model(shape)

