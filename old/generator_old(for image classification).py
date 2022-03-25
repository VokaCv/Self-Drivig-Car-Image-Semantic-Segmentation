import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential

from skimage.io import imread
from skimage.transform import resize


# exemple
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

''' tf.Documentation: 
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    Every Sequence must implement the 
        __init__
        __getitem__ 
        __len__ methods
    If you want to modify your dataset between epochs you may implement 
        on_epoch_end()
    The method 
        __getitem__
    should return a complete batch.
'''

class DataGenerator(Sequence):
    
    def __init__(self, list_IDs, labels, batch_size=32, 
                 dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):

        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        '''
            Here, the method on_epoch_end is triggered once at the very beginning 
            as well as at the end of each epoch. If the shuffle parameter is set 
            to True, we will get a new order of exploration at each pass 
            (or just keep a linear exploration scheme otherwise).
        '''

    def __len__(self):
        '''
        Denotes the number of batches per epoch.
        A common practice is to set this value to:
            len(sample) /  batch size
        so that the model sees the training samples at most once per epoch.
        '''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data. When the batch corresponding to a given index
        is called, the generator executes the __getitem__ method to generate it
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        '''
        Shuffling the order in which examples are fed to the classifier is helpful 
        so that batches between epochs do not look alike. Doing so will 
        eventually make our model more robust
        '''

    # def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples'
        # # X : (n_samples, *dim, n_channels)
        
        # # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')

        #     # Store class
        #     y[i] = self.labels[ID]

        # return X, to_categorical(y, num_classes=self.n_classes)
    
    def __data_generation(self, idx):
        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)



'''
Now, we have to modify our Keras script accordingly so that 
it accepts the generator that we just created.
'''
# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = 0 # IDs
labels = 0 # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
'''
As you can see, we called from model the fit_generator method instead of fit,
where we just had to give our training generator as one of the arguments. 
Keras takes care of the rest!
'''                    


### It is also worth noting that Keras also provide builtin 
# data generator that can be used for common cases. 
# For instance with ImageDataGenerator one can easily load images 
# from a directory and apply some basic transformations:
# datagen = ImageDataGenerator(
#   rescale=1./255,
#   shear_range=0.2,
#   zoom_range=0.2,
#   horizontal_flip=True
# )

