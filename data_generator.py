import numpy as np
import keras

# taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html and modified
import scipy.ndimage
import skimage.transform


class AutoencoderDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, path, batch_size=32, shape=(64,64), n_channels=3,
                 shuffle=True):
        """Initialization"""
        self.path = path
        self.shape = shape
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.filenames = [x for x in path.iterdir() if x.suffix == ".png"]
        self.indexes = None

        self.on_epoch_end() # sets the indexes

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_ids = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        """Generates data containing batch_size samples
            X : (n_samples, *dim, n_channels)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.shape, self.n_channels))
        # y = np.empty((self.batch_size,), dtype=int)

        # Generate data
        for i, ID in enumerate(batch_ids):
            # Store sample
            try:
                data = scipy.ndimage.imread(ID)
                data = data[:,:,:self.n_channels] # remove alpha channel
                data = skimage.transform.resize(data, self.shape, mode="reflect", order=1)
                X[i,] = data
            except:
                pass

            # Store class
            # y[i] = self.labels[ID]

        return X, None
