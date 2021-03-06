import argparse
import os
import random
import sys
from pathlib import Path

from keras import metrics

from data_generator import *

import scipy.ndimage
import skimage.transform

import plugins.SerpentMinecraftGameAgentPlugin.files.helpers.my_model as model
import numpy as np


BATCH_SIZE = 32

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="base directory where dataset data is found")
    parser.add_argument("--reset", action='store_true', help="Create new weights for the network")
    args = parser.parse_args()

    # shape = (64, 128, 3)

    shape = (64, 64, 3)
    actions = 5
    # ae.build_autoencoder(shape, 1024, 0.5)
    # ae.MyAutoencoder(shape)
    m = model.my_model(shape, actions, 1024)

    prev_action = np.zeros(actions)
    prev_action[0] = 1.0
    # print(m.evaluate(np.zeros(shape, 'uint8'), prev_action))

    # open the directory
    p = Path(args.base)
    print(p)
    # dataset = [x for x in p.iterdir() if x.suffix == ".png"]

    try:
        if not args.reset:
            print("Loading previous weights")
            m.load_weights(p)
    except:
        print("No weights found, training from scratch")

    # main loop
    # grab 32 frames, train them in a batch
    # t = trange(500)
    # for ii in t:
    #     batch = []
    #     for in_dir in range(BATCH_SIZE):
    #         # the 0th should always have something in it
    #
    #         data = random.choice(dataset)
    #         data = scipy.ndimage.imread(data)
    #         data = data[:,:,:3] # remove alpha channel
    #         data = skimage.transform.resize(data, shape[:-1], mode="reflect", order=1)
    #         # data = np.array(data * 255, dtype="uint8")
    #         batch.append(data)
    #     loss = m.train_autoencoder(batch)
    #     t.set_postfix(loss=loss)

    generator = AutoencoderDataGenerator(p, batch_size=BATCH_SIZE)

    m.ae.fit_generator(generator, epochs=10, verbose=1,
                        use_multiprocessing=True,
                        workers=6)

    # model.fit_generator(generator=training_generator,
    #                     validation_data=validation_generator,
    #                     use_multiprocessing=True,
    #                     workers=6)

    m.save_weights(p)




if __name__ == "__main__":
    main(sys.argv)
