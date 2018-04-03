import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.autoencoder as ae

shape = (64, 128, 3)
ae.build_autoencoder(shape, 0.5)

