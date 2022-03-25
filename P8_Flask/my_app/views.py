import os, shutil, random, requests, json

from flask import Flask, render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_pyfile('config.py')
# To get one variable, tape app.config['SECRET_KEY']

# TOUTE FONCTION DÉCORÉE PAR @APP.ROUTE EST UNE VUE.

# Vous pouvez lier associer une vue à plusieurs URL ! 
# Il vous suffit de les indiquer les unes à la suite des autres :
# @app.route('/')
# @app.route('/index/')

PATH_FOR_DWNLD_IMG = './my_app/static/images/'
PATH_FOR_DF = './my_app/static/'
# URL = 'http://127.0.0.1:5001/score' # in local with Azureml http server
URL = 'http://83884bbd-148f-43b2-a990-54f432df4c15.northeurope.azurecontainer.io/score'
def delete_dir_content(dir_path):
    for f in os.listdir(dir_path):
        try:
            shutil.rmtree(dir_path + f)
        except:
            os.remove(dir_path + f)

def get_df():
    df = pd.read_csv(PATH_FOR_DF + 'df_val.csv')
    liste = df['xPathLabel'].values.tolist()
    random.seed(42)
    random.shuffle(liste)
    return liste

def save_pics(response):
    delete_dir_content(PATH_FOR_DWNLD_IMG)
    
    # print("Decode JSON serialized NumPy array")
    decodedArrays = json.loads(response.text)

    # return from http has double dict {image: {'image': array}}
    # so we decode to json one second time (after decoding response.text)
    final_image_to_dict = json.loads(decodedArrays["image"])
    final_mask_to_dict = json.loads(decodedArrays["mask"])
    final_pred_to_dict = json.loads(decodedArrays["pred"])
    # and then we use the values from each key
    final_image = np.asarray(final_image_to_dict["image"])
    final_mask = np.asarray(final_mask_to_dict["mask"])
    final_pred = np.asarray(final_pred_to_dict["pred"])

    # exists always
    # pathlib.Path(PATH_FOR_DWNLD_IMG).mkdir(parents=True, exist_ok=True)

    # save pictures (need to convert them to unit8 or float)
    plt.imsave(PATH_FOR_DWNLD_IMG + 'img.png', np.uint8(final_image))
    plt.imsave(PATH_FOR_DWNLD_IMG + 'mask.png', np.uint8(final_mask))
    plt.imsave(PATH_FOR_DWNLD_IMG + 'pred.png', np.uint8(final_pred))

@app.route('/')
def index_base():
    dataset = get_df()
    return render_template('index.html',
                           dataset_name=dataset)

@app.route('/<img_id>/')
def index(img_id):
    img_name = img_id
    dataset = get_df()

    headers = {"Content-Type": "application/json", 'Accept':'application/json'}
    data_to_send = {"image_name": img_name}
    response = requests.post(URL, json=data_to_send, headers=headers)

    if not response.ok:
        pass
        # print(f"Erreur de type {response.status_code}")
    else:
        save_pics(response)

    return render_template('prediction.html',
                           dataset_name=dataset)


if __name__ == "__main__":
    app.run()