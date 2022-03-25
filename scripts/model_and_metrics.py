from tensorflow.keras import backend as K
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Dropout, Reshape, Permute

from tensorflow.keras.models import Model

from tensorflow.keras.applications import vgg16, EfficientNetB4
    # EfficientNetB4: 
        # This model takes input images of shape (224, 224, 3), 
        # and the input data should range [0, 255]. 
        # Normalization is included as part of the model.


import tensorflow as tf
import tensorflow_addons as tfa

import pandas as pd
import numpy as np

IMAGE_RES = (256,512) # H x W

def dice_coeff(y_true, y_pred):
    # expects y_true and y_pred reduced to 2D (H x W , Classes)
    
    smooth = 1.
    
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)   

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f + y_pred_f)

    score = (2. * intersection + smooth) / (union + smooth)

    return score

def dice_loss(y_true, y_pred):
    'set to 1- so it tends to 0, and we can set minimize to compiler'
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def dice_test(y_true, y_pred):
    'instead of class DiceMetric'
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours 
    # trying to debug and almost give up.
    
    result = tf.py_function(dice_coeff, [y_true, y_pred], tf.float32)

    return result

class DiceMetric(tf.keras.metrics.Metric):
    '''
    does not work well, it accumulates dice_coef over batches
    and then sums (accumulated + a-1 ) / nb_batch
    '''
    def __init__(self, num_classes, name="F1DiceMetric", **kwargs):                     
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice_coef = self.add_weight(name='dice_coef', initializer='zeros', dtype='float32')
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1, self.num_classes))
        y_pred = tf.reshape(y_pred, (-1, self.num_classes))

        ''' 
        assign_add: auto increments values and then add them at the end of the batch
         so here i take off the previous so the calcul is correct
        '''
        dice = dice_coeff(y_true, y_pred)
        previous = self.dice_coef
        self.dice_coef.assign_add(dice-previous)
        # tf.print("\n inside update dice",dice)
        # tf.print("\n inside update self.dice",self.dice_coef)

    def result(self):
        # return tf.reduce_mean(self.dice_coef)
        return self.dice_coef

    def reset_state(self):
        self.dice_coef.assign(0.)
        # self.batch_counter.assign(0.)
        K.batch_set_value([(v, 0) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()        

class IoU(tfa.metrics.FBetaScore):
    """Score IoU"""
    def __init__(
        self,
        num_classes,
        average='micro',
        threshold=None,
        name="IoU",
        dtype=None,
    ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1, self.num_classes))
        y_pred = tf.reshape(y_pred, (-1, self.num_classes))
        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def result(self):
        iou = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives + self.false_negatives
        )
        
        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            iou = tf.reduce_sum(iou * weights)

        elif self.average is not None:  # [micro, macro]
            iou = tf.reduce_mean(iou)

        return iou

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config

class DiceLossCls(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        return dice

# def single_conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
#     x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
#     x = Activation("relu")(x)
#     return x

# def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
#     x = Conv2D(filters=nfilters, kernel_size=(size, size), 
#                 padding=padding, kernel_initializer=initializer
#                 )(tensor)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)

#     x = Conv2D(filters=nfilters, kernel_size=(size, size), 
#                 padding=padding, kernel_initializer=initializer
#                 )(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     return x

# def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
#     y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
#     y = concatenate([y, residual], axis=3)
#     y = conv_block(y, nfilters)
#     return y


def conv_block(tensor, nfilters, size=3, padding='same', 
               initializer="he_normal", nb_blocks=1):

    #add at least one, and if nb_blocks > 1 add others
    x = Conv2D(filters=nfilters, kernel_size=(size, size), 
                padding=padding, kernel_initializer=initializer
                )(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for i in range(nb_blocks-1):
        x = Conv2D(filters=nfilters, kernel_size=(size, size), 
                    padding=padding, kernel_initializer=initializer
                    )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def deconv_block(tensor, residual, nfilters, size=3, 
                 padding='same', strides=(2, 2), nb_conv_blocks=2):

    y = Conv2DTranspose(nfilters, kernel_size=(size, size), 
                        strides=strides, padding=padding
                        )(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters, nb_blocks=nb_conv_blocks)
    return y

def get_multiple_outputs(model, only_first_blocks = True):
    # get all layers until the last one in the list (included)
    # define outputs for skip connections
    encoder_output_ids=[0,                           # 256 x 512 x 3
                        "block2a_expand_activation", # 128 x 256 x 144
                        "block3a_expand_activation", # 64 x 128 x 192 
                        "block4a_expand_activation", # 32 x 64 x 336
                        "block6a_expand_activation", # 16 x 32 x 960
                        ]

    if only_first_blocks == True:
      outputs = []
      # add first layer and then add layers in the list, names
      outputs.append(model.layers[0].output)

      for id in encoder_output_ids[1:]:
        outputs.append(model.get_layer(id).output)
  
      return Model(model.input, outputs)
    else:
      # get the whole model
      return Model(model.input, model.output)

def build_encoder():
  # use EfficientNet as encoder
    input_shape = (*IMAGE_RES, 3)
    inputs = Input(shape=input_shape)

    base_model = EfficientNetB4(include_top=False, 
                              weights='imagenet', 
                              # input_shape=input_shape
                              input_tensor=inputs
                              )

    only_first_blocks = True #if false, get entire model
    if only_first_blocks == True:
      encoder = get_multiple_outputs(base_model, only_first_blocks)
    else:
      encoder = get_multiple_outputs(base_model, only_first_blocks)

    return encoder


def my_Unet(img_height, img_width, nclasses=8, filters=64):
    # down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    scaled = layers.experimental.preprocessing.Rescaling(1./255., 0.0, 'rescaling')(input_layer)
    conv1 = conv_block(scaled, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)

    # up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    
    # output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    # output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('softmax')(output_layer)
    

    model = Model(inputs=input_layer, outputs=output_layer, name='my_Unet')
    return model    

def my_miniUnet(img_height, img_width, nclasses=8, filters=64):
    # down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    scaled = layers.experimental.preprocessing.Rescaling(1./255., 0.0, 'rescaling')(input_layer)
    conv1 = conv_block(scaled, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*4)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = conv_block(conv2_out, nfilters=filters*4)
    # conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = conv_block(conv3_out, nfilters=filters*8)
    # conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    # conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv2_out, nfilters=filters*12)
    conv5 = Dropout(0.5)(conv5)

    # up
    deconv6 = deconv_block(conv5, residual=conv2, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    # deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    # deconv7 = Dropout(0.5)(deconv7) 
    # deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv6, residual=conv1, nfilters=filters*2)

    # output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    # output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('softmax')(output_layer)
    

    model = Model(inputs=input_layer, outputs=output_layer, name='my_miniUnet')
    return model

def my_testUnet(img_height, img_width, num_classes):
    inputs = Input(shape=(img_height, img_width, 3))
    scaled = layers.experimental.preprocessing.Rescaling(1./255., 0.0, 'rescaling')(inputs)
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(scaled)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    # for filters in [336, 192, 144, 64]:
    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs, name='my_testUnet')
    return model

def from_book_model(num_classes=8):

    '''
    In the classification, we use MaxPooling2D layers to downsample feature maps. 
    
    Here, we downsample BY ADDING STRIDES to every other convolution layer.
    
    We do this because, in the case of image segmentation,we care a lot 
    about the spatial location of information in the image
    we need to produce per-pixel target masks as output of the model. 
    
    When you do 2x2 max pooling, you are completely 
    destroying location information within each pooling window 
    (you return one scalar value per window, with zero knowledge
    of which of the four locations in the windows the value came from). 
    
    So while max pooling layers perform well for classification tasks, 
    they would hurt us quite a bit for a segmentation task. 
    
    Meanwhile, strided convolutions do a better job at downsampling 
    feature maps while retaining location information. 
    
    We tend to use strides instead of max pooling 
    IN ANY MODEL THAT CARES ABOUT FEATURE LOCATION.
    '''

    inputs = Input(shape=(256, 512, 3))
    scaled = layers.experimental.preprocessing.Rescaling(1./255., 0.0, 'rescaling')(inputs)

    # downsampling
    # Note how we use padding="same" everywhere to avoid 
    # the influence of border padding on feature map size.
    x = Conv2D(64, 3, strides=2, activation="relu", padding="same")(scaled)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, activation="relu", padding="same")(x)
    x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(256, 3, activation="relu", padding="same")(x)

    # upsampling
    x = Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(256, 3, activation="relu", padding="same", 
                        strides=2)(x)
    x = Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, activation="relu", padding="same", 
                        strides=2)(x)
    x = Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, activation="relu", padding="same", 
                        strides=2)(x)
    
    # outputs
    # We end the model with a per-pixel three-way to classify each 
    # output pixel into one of our three categories.
    outputs = Conv2D(num_classes, 3, activation="softmax",
        padding="same")(x)

    model = Model(inputs, outputs)
    return model

def my_EfficientNet(num_classes=8, fine_tuning=False):
    # down
    encoder = build_encoder()
    CREATE_BOTTLENECK = True
    NB_UNFREEZE = len(encoder.layers)//10 #unfreeze 10% of layers

    # freeze all the layers of the base model
    for layer in encoder.layers:
        layer.trainable = False

    if fine_tuning == True: # unfreeze some to adapt some generic features
        for layer in encoder.layers[-NB_UNFREEZE:]:
            # do not unfreeze Normalization layers, 
            # see here https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
            if not isinstance(layer, BatchNormalization):
                    layer.trainable =  True

    conv1_residual = encoder.outputs[0] # 256 512   3
    conv2_residual = encoder.outputs[1] # 128 256 144
    conv3_residual = encoder.outputs[2] #  64 128 192
    conv4_residual = encoder.outputs[3] #  32  64 336
    # last layer
    conv5_residual = encoder.outputs[4] #  16  32 960
    last_layer_shape = conv5_residual.shape[-3:]
    last_layer_nb_filters = conv5_residual.shape[-1]

    if CREATE_BOTTLENECK == True:
        bottleneck = conv_block(conv5_residual, last_layer_nb_filters)
    else:
        bottleneck = conv5_residual

    # up
    deconv6 = deconv_block(bottleneck, residual=conv4_residual, nfilters=960)
    # deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3_residual, nfilters=336)
    # deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2_residual, nfilters=192)
    deconv9 = deconv_block(deconv8, residual=conv1_residual, nfilters=144)
    
    # output
    output_layer = Conv2D(filters=num_classes, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    # output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('softmax')(output_layer)
    

    model = Model(inputs=encoder.input, outputs=output_layer, name='my_EfficientNet')
    # model = Model(inputs=encoder.input, outputs=x, name='my_EfficientNet')
    return model  