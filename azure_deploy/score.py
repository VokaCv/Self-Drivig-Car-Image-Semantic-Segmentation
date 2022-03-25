import os, json, logging, joblib, requests, shutil
from json import JSONEncoder
import pathlib

import numpy as np

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Model, Workspace, Dataset


from model_and_metrics import DiceLossCls, DiceMetric

from pathlib import Path

# global hierarchy
# global files

# hierarchy = {}
# def list_files(dir, loop_subfolders=True):
#     with os.scandir(dir) as it:
#         files = []
#         for iter in it:
#             if iter.is_dir():
#                 if loop_subfolders == True:
#                     list_files(dir=iter, loop_subfolders=True)
#                 else:
#                     pass

#             elif iter.is_file():
#                 files.append(iter.name)
            
#         print(files)
#         hierarchy[dir.name] = files
#     return hierarchy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


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

def delete_dir_content(dir_path):
    for f in os.listdir(dir_path):
        try:
            shutil.rmtree(dir_path + f)
        except:
            os.remove(dir_path + f)

def init():
    # """
    # This function is called when the container is initialized/started, typically after create/update of the deployment.
    # You can write the logic here to perform init operations like caching the model in memory
    # """

    global model, images_dataset, ws
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # '''
    # There are two ways to locate models in your entry script:
    #     AZUREML_MODEL_DIR: An environment variable containing the path to the model location.
    #     Model.get_model_path: An API that returns the path to model file using the registered model name.
    # '''

    # """ 
    # For Azure
    # """
    # model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 
    #                         "outputs", "best_model")
    MODEL_PATH_FOLDER = os.getenv("AZUREML_MODEL_DIR")

    WS_NAME = 'p8_ws'
    SUBSCRIPTION_ID_FREE_STUDENTS = 'd4258567-1f40-439d-93e3-23364c743517'
    RESSOURCE_GROUP = 'OpenClassroms'

    TENANT_ID = '7b9caa82-28df-475e-a0a7-01fd6a94aeb8'
    APP_ID ='2bd162b6-d683-4430-b122-e3ad47e0b076'
    APP_PASSWORD = 'FnU7Q~wHO9ONt.YOgy0VEy61a5F1A8YAoMgaF'

    if (MODEL_PATH_FOLDER==None):
        MODEL_PATH_FOLDER = '.'
    
    # !!! AZURE STORE RANDOMLY MODEL INSIDE outputs/best_model or only best model
    # CHECK WITH def list_files in run below for the folder hierarchy  
    # OR DOWNLOAD MODEL LOCALY TO SEE THE HIERARCHY  
    model_path = os.path.join(MODEL_PATH_FOLDER,"best_model")
    model = load_model(model_path, compile=False)
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, 
                loss=DiceLossCls(), 
                metrics=[DiceMetric(num_classes=8)])

    # #IN LOCAL
    # model_path = '../model_azure/efficient_finetune/best_model/'
    # model = load_model(model_path, compile=False)
    # optimizer = tf.keras.optimizers.RMSprop()
    # model.compile(optimizer=optimizer, 
    #             loss=DiceLossCls(), 
    #             metrics=[DiceMetric(num_classes=8)])

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=TENANT_ID,
        service_principal_id=APP_ID,
        service_principal_password=APP_PASSWORD)

    ws = Workspace(
        subscription_id=SUBSCRIPTION_ID_FREE_STUDENTS,
        resource_group=RESSOURCE_GROUP,
        workspace_name=WS_NAME,
        auth=svc_pr
        )

    images_dataset = ws.datasets['cityscapes']

def run(raw_data):
    # """
    # This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    # In the example we extract the data from the json input and call the scikit-learn model's predict()
    # method and return the result back
    # """

    # """
    # Raw data should be in this format :
    # {
    #     "image_name": 'image_name only root of the name, common for image and mask',
    # }
    # """
    if raw_data == "":
        return {}
    else:
        data = json.loads(raw_data)

    PATH_FOR_DWNLD_IMG = 'data/images/'
    pathlib.Path(PATH_FOR_DWNLD_IMG).mkdir(parents=True, exist_ok=True)

    datastore = ws.get_default_datastore() 
    base_name = data['image_name'] #"frankfurt_000000_000294"

    delete_dir_content(PATH_FOR_DWNLD_IMG)

    full_img_name = base_name + '_leftImg8bit.png'
    full_mask_name = base_name + '_gtFine_color.png'
    img_path_datastore = 'cityscapes/images/val/'+ full_img_name
    mask_path_datastore = 'cityscapes/ground_truth/val/'+ full_mask_name

    # download images from dataset and store it locally
    img = Dataset.File.from_files((datastore, img_path_datastore))
    mask = Dataset.File.from_files((datastore, mask_path_datastore))
    img.download(target_path=PATH_FOR_DWNLD_IMG)
    mask.download(target_path=PATH_FOR_DWNLD_IMG)
    FINAL_IMG_NAME = 'img.png'
    FINAL_MASK_NAME = 'mask.png'
    os.rename(PATH_FOR_DWNLD_IMG + full_img_name, PATH_FOR_DWNLD_IMG + FINAL_IMG_NAME)
    os.rename(PATH_FOR_DWNLD_IMG + full_mask_name, PATH_FOR_DWNLD_IMG + FINAL_MASK_NAME)

    img_path_local = PATH_FOR_DWNLD_IMG +  FINAL_IMG_NAME
    mask_path_local = PATH_FOR_DWNLD_IMG + FINAL_MASK_NAME

    # read the image
    x_img = Image.open(img_path_local).convert('RGB')
    x_image = x_img.resize((512,256))
    x_array = np.array(x_image)

    # read the mask
    y_img = Image.open(mask_path_local).convert('RGB')
    y_image = y_img.resize((512,256))
    y_array = np.array(y_image)

    # Get a prediction from the model
    pred = model.predict(np.expand_dims(x_array, 0))[0]
    mask_3D = label_to_mask(pred)
    

    # Serialization image, mask, prediction
    numpyData = {"image": x_array}
    encoded_img = json.dumps(numpyData, cls=NumpyArrayEncoder)  

    y_numpyData = {"mask": y_array}
    encoded_mask = json.dumps(y_numpyData, cls=NumpyArrayEncoder)  

    y_pred_numpyData = {"pred": mask_3D}
    encoded_pred = json.dumps(y_pred_numpyData, cls=NumpyArrayEncoder)  

    # TEST HERE THE FOLDER HIERARCHY IN AZURE
    # path = Path(model_path_folder)
    # return_dic = list_files(path)

    return_dic = {}
    return_dic['image'] = encoded_img
    return_dic['mask'] = encoded_mask
    return_dic['pred'] = encoded_pred

    # return_dic = {}
    # return_dic['image'] = 1
    # return_dic['mask'] = 2
    # return_dic['pred'] = 3
    return return_dic

if __name__ == '__main__ ':
    init()