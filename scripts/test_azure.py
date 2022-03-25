import os, sys
import argparse

import tensorflow as tf

from azureml.core import Run, Dataset
from azureml.core import Run, Experiment

from model_and_metrics import my_EfficientNet

run = Run.get_context()
ws = run.experiment.workspace


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print(physical_devices)



df_train = Dataset.get_by_name(ws, name='df_train').to_pandas_dataframe()
df_test = Dataset.get_by_name(ws, name='df_test').to_pandas_dataframe()
df_valid = Dataset.get_by_name(ws, name='df_valid').to_pandas_dataframe()

'get arguments from ScriptRunConfig'
parser = argparse.ArgumentParser(description='Process dataset')

'''
parser.add_argument('first_var', type=str) 
creates a positional argument 
    !!!order matters in passing it in command line

Optional arguments are created just like positional 
arguments except that they have a '--' double dash 
at the start of their name (or a'-' single dash and 
one additional character for the short version). 

For example, you can create an optional argument 
with parser.add_argument('-m', '--my_optional').
'''

parser.add_argument('--image_dataset_mounted_dest', type=str, help='img dataset as_mount()')
parser.add_argument('--run_custom_name', type=str, help='a name to give to the run')

# from ScriptRunConfig
# The training run must use the same hyperparameter configuration and 
# mounted the outputs folders. The training script must accept the 
# 'resume-from' argument, which contains the checkpoint or model 
# files from which to resume the training run
parser.add_argument('--resume-from', type=str, default=None)

args = parser.parse_args()

mount_point = args.image_dataset_mounted_dest
# mount_point = sys.argv[1]

df_train = df_train.sample(2, ignore_index=True)
df_test = df_test.sample(2, ignore_index=True)


experiment_name = 'p8_EfficientNetEncoder'
# experiment_name = 'p8_new'

# give run a custom name 
# run_custom_name = "learning_rate_0.4"


run.log('resume from script', args.resume_from)

model = my_EfficientNet(8,True)
path_to_checkpoint = args.resume_from  + "/checkpoint"
model.load_weights(path_to_checkpoint)

i = 0
for l in model.layers:
    if l.trainable == True:
        i += 1

run.log('nb_layers', len(model.layers))
run.log('nb_trainable', i)
run.log("Path", mount_point)
run.log("len1", len(df_train))
run.log("len2", len(df_test))
run.log("len3", len(df_valid))
run.log('test', df_train['xPathLabel'][0])

run.display_name = 'custom_name'

path = str(mount_point) + '/' + str(df_train['xPathLabel'][0])

run.log_image("ROC", path)


run.complete()