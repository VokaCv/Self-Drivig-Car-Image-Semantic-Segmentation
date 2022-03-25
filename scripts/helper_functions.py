import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A


# helper function for data visualization
def get_classes(labels):
    '''we create two items: 
        1. df with all subcategories, so we can labelize all pixels
        2. main_classes, only main category, with one color for coloring the output 
    '''
    df = pd.DataFrame(labels)
    df = df[["name", "categoryId", "category", "color"]]
    main_classes = df.drop_duplicates(subset='category').reset_index().drop(columns=['index', 'name'])
    return [df, main_classes]

def form_2D_label(mask, classes, ins=False):
    '''
    takes mask in 3D array (H W RGB)
    takes labels as list (class_map here)
    takes colors as tuple(r,g,b) class_color here

    creates output with labels instead of color tuple with dimension (HxW)
    '''
    class_color = np.array(list(classes['color'].values))
    class_map = np.array(list(classes['categoryId'].values))

    label_2d = np.zeros(mask.shape[:2], dtype= np.uint8)

    # to test this    
    # label_to_color = {
    #                     0: (128,  64, 128),
    #                     1: (244,  35, 232),
    #                     2: (70,  70,  70),
    #                     3: (102, 102, 156),
    #                     4: (190, 153, 153),
    #                     5: (153, 153, 153),
    #                     6: (102, 102, 156)
    #                     }

    # for label_id, rgb in label_to_color.items():
    #     label_2d[(mask == rgb).all(axis=2)] = label_id

    for i, rgb in enumerate(class_color):
        label_2d[(mask == rgb).all(axis=2)] = class_map[i]

    return label_2d

def label_to_mask(mask, classes=""):
    # iterate here with classes
    label_to_color = {
                        0: [0  ,   0,   0],
                        1: [150,   0, 150],
                        2: [120, 120, 120],
                        3: [255, 255,   0],
                        4: [107, 142,  35],
                        5: [144, 184, 228],
                        6: [220,  20,  60],
                        7: [0  ,   0, 142],
                    }

    mask_max = np.reshape(np.argmax(mask, axis=-1),mask.shape[0]*mask.shape[1])
    output = np.zeros((mask.shape[0]*mask.shape[1],3), dtype= np.uint8)

    for label_id, rgb in label_to_color.items():
        output[(mask_max == int(label_id)), :] = rgb

    output = np.reshape(output,(mask.shape[0],mask.shape[1],3))
    return output

def visualize(image, mask, original_image=None, original_mask=None):
    mask = label_to_mask(mask)
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(16, 10))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 10))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image')
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask')
      
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image') 
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask')

def get_augmented_images(image_array, mask_array):
    '''
    image, mask as numpy array
    '''
    # créer quelques transformation pour le test
    transform_pipe = A.Compose([
                    A.HorizontalFlip(p=0.75),
                    A.RandomBrightnessContrast(p=0.8),
                    A.Blur(p=0.8),
                    ])

    transformed = transform_pipe(image=image_array, mask=mask_array)
    # renvoie le dictionnaire
    return transformed
