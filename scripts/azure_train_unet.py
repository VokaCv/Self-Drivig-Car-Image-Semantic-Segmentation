import os, sys, argparse

import time, datetime

from azureml.core import Run, Dataset, Workspace

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from generator import DataGenerator
from helper_functions import get_classes
from labels import labels
from model_and_metrics import my_Unet, my_miniUnet, my_testUnet, from_book_model
from model_and_metrics import DiceLossCls, DiceMetric, IoU, dice_test



physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print(physical_devices)


begin = time.time()

run = Run.get_context()
ws = run.experiment.workspace
# ws = Workspace.from_config()

print("datasets are loading...")

df_train = Dataset.get_by_name(ws, name='df_train').to_pandas_dataframe()
df_test = Dataset.get_by_name(ws, name='df_test').to_pandas_dataframe()
df_valid = Dataset.get_by_name(ws, name='df_valid').to_pandas_dataframe()

# # #limit the number of images cause it's too long
# df_train = df_train.sample(1000, ignore_index=True)
# df_test = df_test.sample(350, ignore_index=True)

print("reading args...")
#in ScriptRunConfig send dataset.as_mount() in arguments'
parser = argparse.ArgumentParser(description='Process dataset')
parser.add_argument('--image_dataset_mounted_dest', type=str, help='img dataset as_mount()')
parser.add_argument('--run_custom_name', type=str, help='a name to give to the run')

args = parser.parse_args()

mount_point = args.image_dataset_mounted_dest
# mount_point = sys.argv[1]
mounted_drive = str(mount_point) + '/'

# path = mounted_drive + str(df_train['xPathLabel'][0])
# run.log_image("test_img", path)

# Parameters
batch_size_min = min(4, len(df_train))
params = {'targetSize': (512,256),
          'batchSize': batch_size_min,
          'augment': 1,
          'nbChannels': 3,
          'shuffle': True}

class_labels = get_classes(labels)
print("dataGenrators are loading...")
trainGen = DataGenerator(data_dir=mounted_drive,
                         data=df_train,
                         xPathLabel='xPathLabel',
                         yPathLabel='yPathLabel',
                         classes=class_labels,
                         **params)

testGen = DataGenerator(data_dir=mounted_drive,
                        data=df_test,
                        xPathLabel='xPathLabel',
                        yPathLabel='yPathLabel',
                        classes=class_labels,
                        **params)

# # not needed here, use it in inference
# validGen = DataGenerator(data_dir=mounted_drive,
#                         data=df_valid,
#                         xPathLabel='xPathLabel',
#                         yPathLabel='yPathLabel',
#                         classes=class_labels,
#                         **params)

print("model is loading...")
# model = my_Unet(params['targetSize'][1],
#                 params['targetSize'][0], 
#                 nclasses=8, filters=64,
#                 )

# model = my_miniUnet(params['targetSize'][1],
#                 params['targetSize'][0], 
#                 nclasses=8, filters=64,
#                 )

# model = my_testUnet(params['targetSize'][1],
#                 params['targetSize'][0], 
#                 num_classes=8,
#                 )


model = from_book_model(num_classes=8)


print("model is compiling...")

loss_func = DiceLossCls()
# loss_func = 'categorical_crossentropy'

metrics_to_use = [DiceMetric(num_classes=8)]
# metrics_to_use = [IoU(num_classes=8)]
# metrics_to_use = [tf.keras.metrics.MeanIoU(num_classes=8)]
# metrics_to_use = [dice_test]

monitor_val = 'val_F1DiceMetric' #name given in the class
# monitor_val = 'val_IoU'
# monitor_val = 'val_mean_io_u'
# monitor_val = 'val_dice_test' 

# in callbacks decision to save the model is based on
# this value, in regard to metrics
mode_save_metric = 'max'
# mode_save_metric = 'auto'

print("loading learning rate optimizer...")
#use learning rate finder
INIT_LR = 1e-4
MAX_LR = 1e-2
steps_per_epoch = len(trainGen.dataset) // batch_size_min
c_lr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate = INIT_LR,
        maximal_learning_rate = MAX_LR,
        step_size = 2*steps_per_epoch,
        scale_fn = lambda x: 1/(2.**(x-1)),
        # scale_mode  = 'cycle',
        # name = 'CyclicalLearningRate'
        )

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.RMSprop()
# optimizer = tf.keras.optimizers.SGD()


model.compile(optimizer=optimizer, 
            loss=loss_func, 
            metrics=metrics_to_use,
            )

callbacks = keras.callbacks.ModelCheckpoint(filepath="outputs/checkpoint", 
                              monitor=monitor_val,
                              save_best_only=True,
                              mode=mode_save_metric,
                              save_weights_only=True)

print("model is training...")
NB_EPOCH = 5
history = model.fit(trainGen.dataset,
                    validation_data=testGen.dataset,
                    # use_multiprocessing=True,
                    # workers=6,
                    callbacks=[callbacks],
                    epochs=NB_EPOCH,
                    )

print("saving the best model...")
# On restaure les meilleurs paramètres
model.load_weights('outputs/checkpoint')

# On crée le dossier d'enregistrement du modèle s'il n'existe pas
MODEL_PATH = 'outputs/best_model/'
os.makedirs(MODEL_PATH, exist_ok=True)
# On enregistre le modèle
model.save('outputs/best_model/')

run.log_list("perf metric:", history.history[monitor_val])

end = time.time()
duration = datetime.timedelta(seconds=(round(end - begin)))

temp = end - begin
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes

final_time =  str(int(hours)).zfill(2) + ":" \
        + str(int(minutes)).zfill(2) + ":" \
        + str(int(seconds)).zfill(2)


run.log("Execution time:", final_time)

run.display_name = args.run_custom_name

print("TRAINING DONE!")
run.complete()