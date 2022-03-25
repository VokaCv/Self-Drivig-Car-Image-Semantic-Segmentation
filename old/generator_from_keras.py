import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3 # 1 = "b&w" / 3 = 'rbg'

'''
for small datasets we can load the whole dataset into numpy array
   
   X_train = np.zeros((nb_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                    dtype=np.uint8)
   
   and fill it through loop with 
       keras img_to_array or PIL.open(path) -> np.asarray(img)
   
   save it np.save(path, "fname.npy", X_train)
   load: X_Train = np.load(path, "fname.npy"))

but for larger datasets, in order to avoid OUT OF MEMORY
we must create a generator:
'''

# 1. we can use tf.keras.preprocessing.image.ImageDataGenerator
image_datagen = ImageDataGenerator(rescale=1./255.,
                                    #rotation_range=0.2,
                                    #width_shift_range=0.05,
                                    #height_shift_range=0.05,
                                    #shear_range=0.05,
                                    #zoom_range=[0.05,0.2],
                                    #horizontal_flip=True,
                                    # brightness_range=[0.2,1.2])
                                    )

mask_datagen = ImageDataGenerator(rescale=1./255.)

'''
if has ImageDataGenerator().flow() and flow_from_directory() and
flow_from_dataframe() methods for reading data directly on the fly
'''
# 2. Create generator
home_path = ''
batch_size = 16
train_generator = image_datagen.flow_from_directory(
                  directory=home_path + r'/train/',
                  target_size=(IMG_HEIGHT, IMG_WIDTH), # resize to this size
                  color_mode="rgb", # for coloured images
                  batch_size=batch_size, # number of images to extract from folder for every batch
                #   class_mode="binary", # classes to predict
                  class_mode="categorical", # classes to predict
                  seed=41 # to make the result reproducible
                  )

'''
Right, you have created the iterators for augmenting the images. 
But how do you feed it to the neural network so that it can 
augment on the fly?

For that, all you need to do is feed the iterator as an input 
to the Keras fit_generator() method applied on the neural network 
model along with epochs, batch_size, and other important arguments
'''
# 3. Feed it to the NN with fit_generator() on your 
# .flow(), flow_from_directory() or flow_from_dataframe() 
# variable from above
'''
EPOCH = 1
model.fit_generator(train_generator, 
                    epochs=EPOCH,  # one forward/backward pass of training data
                    steps_per_epoch=X_train.shape[0]//batch_size,  # number of images comprising of one epoch
                    validation_data=(X_test, y_test), # Or validation_data=valid_generator
                    validation_steps=X_test.shape[0]//batch_size)
'''
