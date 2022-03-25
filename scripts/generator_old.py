import os
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence

import tensorflow.keras as kr
import tensorflow as tf

from scripts.helper_functions import form_2D_label, get_augmented_images

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

    For Data input
    https://keras.io/getting_started/intro_to_keras_for_engineers/

    Keras models accept three types of inputs:
        1. NumPy arrays, just like Scikit-Learn and many other 
            Python-based libraries. This is a good option if your 
            data fits in memory.
        2. TensorFlow Dataset objects. This is a high-performance 
            option that is more suitable for datasets that do 
            not fit in memory and that are streamed from disk 
            or from a distributed filesystem.
        3. Python generators that yield batches of data 
            (such as custom subclasses of the keras.utils.Sequence class).
'''

class DataGenerator(Sequence):
    
    def __init__(self, data_dir, data, 
                xPathLabel, yPathLabel, 
                classes,
                batchSize=2, shuffle=True,
                targetSize=(128,128),
                augment = None,
                nbChannels = 3):
        ''' 
            data_dir : racine de dossier source 
            data : nos données dans un dataframe Pandas 
                   [xLabel = path vers images de X, yLabel = path vers masques]
            xLabel : nom de colonne du df contenant nos données X
            yLabel : nom de colonne du df contenant nos données Y
            batchSize : taille d’un mini lot de données
            shuffle : booléen si on souhaite envoyer des données de 
                      façon aléatoire, ou dans l’ordre de l’index du dataframe
            targetSize : afin de resize nos images 
            augment : réaliser data augmentation à la volée
            nbChannels : RGB=3 or Grayscale=1
        '''
        self.data_dir = data_dir
        self.xData = data[xPathLabel]
        self.yData = data[yPathLabel]
        self.classes = classes
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.targetSize = targetSize
        self.augment = augment
        self.nbChannels = nbChannels
        self.on_epoch_end()
        '''
            Here, the method on_epoch_end is triggered once at the very beginning 
            as well as at the end of each epoch. If the shuffle parameter is set 
            to True, we will get a new order of exploration at each pass 
            (or just keep a linear exploration scheme otherwise).
        '''
        self.generate_dataset()

    def __len__(self):
        '''
        Denotes the number of batches per epoch.
        a common practice is to set this value to:
            len(sample) /  batch size
        so that the model sees the training samples at most once per epoch.
        '''
        return int(np.floor(len(self.xData) / self.batchSize))
    

    def __iter__(self):
        # On itère les batchs
        for i in range(len(self)):
            yield self[i]


    def read_resize_img(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, (self.targetSize[1],self.targetSize[0]), 
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image

    def normalize_transform_img(self, image_path, label_path):
        # On charge les images et on les met à l'échelle
        image = self.read_resize_img(image_path)
        label = self.read_resize_img(label_path)

        if self.augment is not None:
            # calls Albumentation, returns dict with 'image' and 'mask' keys as arrays
            transformed = get_augmented_images(image.numpy(), label.numpy())
            image = transformed['image']
            # do not transfor labels
            # label = transformed['mask']

        else:
            image = image.numpy()
    
        label_2D = form_2D_label(label.numpy(), self.classes[0])

        label_cat = to_categorical(label_2D, num_classes=8)
        label = label_cat

        image = tf.convert_to_tensor(image, tf.uint8)
        label = tf.convert_to_tensor(label, tf.uint8)

        return image, label

    def get_data(self, batch):
        xBatch = []
        yBatch = []

        # Traitement de vos données
        for i, rowId in enumerate(batch):
            image_path = self.data_dir + str(self.xData.iloc[rowId])
            label_path = self.data_dir + str(self.yData.iloc[rowId])

            image, label = self.normalize_transform_img(image_path, 
                                                        label_path)

            xBatch.append(image)
            yBatch.append(label)

        xBatch = tf.convert_to_tensor(xBatch, tf.uint8)
        yBatch = tf.convert_to_tensor(yBatch, tf.uint8)

        return xBatch, yBatch

    def __getitem__(self, index):
        '''
        Generate one batch of data. When the batch corresponding to a given index
        is called, the generator executes the __getitem__ method to generate it
        '''
        # Genere nombre d'ID de la taille de batchSize
        # sur de row de DATA (batchSize=2, [0,1])
        currentBatchIdsRow = self.indexes[index * self.batchSize : (index+1) * self.batchSize]
        
        x,y = self.get_data(currentBatchIdsRow)

        return x,y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        # [0,1,2,3,4... nb_image]
        self.indexes = np.arange(len(self.xData))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        '''
        Shuffling the order in which examples are fed to the classifier is helpful 
        so that batches between epochs do not look alike. Doing so will 
        eventually make our model more robust
        '''

    def generate_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((
            self.xData.apply(lambda x: self.data_dir + str(x)),
            self.yData.apply(lambda x: self.data_dir + str(x)),
        ))


        if self.shuffle:
            self.dataset = self.dataset.shuffle(len(self.dataset), seed=42, reshuffle_each_iteration=True)

        def tf_normalize_image_and_label(image_path, label_path):
            image, label = tf.py_function(
                func=self.normalize_transform_img,
                inp=[image_path, label_path],
                Tout=[tf.uint8, tf.uint8]
            )

            image.set_shape(tf.TensorShape([self.targetSize[1],
                                            self.targetSize[0], 
                                            3]))

            label.set_shape(tf.TensorShape([self.targetSize[1],
                                            self.targetSize[0], 
                                            8]))

            return image, label

        self.dataset = self.dataset.map(
            tf_normalize_image_and_label,
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(self.batchSize).prefetch(tf.data.AUTOTUNE)