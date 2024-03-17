# # **Imports**

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from glob import glob

import project_functions as pf

# # **Configurações Globais**

# Define a GPU visível (0 -> 3080; 1 -> 3090)
pf.set_gpu(0)

test_file = "git/speech_enhancement/config_files/Test3.yml"
my_dir = "/home/pertum/git/speech_enhancement/"

# load config
with open(test_file, 'r') as file:
    test_config = yaml.safe_load(file)

# Variáveis globais
fs = test_config['model']['fs']
nperseg = test_config['model']['nperseg']
nfft = test_config['model']['nfft']
noverlap = test_config['model']['noverlap']
time_frames = test_config['model']['time_frames']
phase_aware = test_config['model']['phase_aware']
noise_list = test_config['dataset']['noise_list']
SNR_list = [-5, 0, 5, 10, 15, 20]
testID = test_config['testID']

# load dataset of testing
test_dataset = pd.read_pickle(f"{my_dir}{testID}_test_dataset.pkl")
print(test_dataset.head(3))

# load models
models = {}
params = {}
model_name = "DenoisingCNN"
model_file = glob(my_dir + f"checkpoints/*{model_name}*{testID}.h5")[0]
models[model_name] = model_file

norm_params_file = glob(my_dir + f"data/*norm_params_*{testID}")[0]
if os.path.exists(norm_params_file):
    print('\nCarregando Parâmetros de Normalização...')
    with open(norm_params_file, 'rb') as f:
        norm_params = pickle.load(f)
else:
    raise Exception('Não há arquivos com parâmetros de normalização')

model_name = "DenoisingCRNN"
model_file = glob(my_dir + f"checkpoints/*{model_name}*{testID}.tf")[0]
models[model_name] = model_file

# test

K.clear_session()

model_CNN = pf.CR_CED_model((nfft//2 + 1, time_frames, 1), norm_params)

model_CNN.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0015, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1.0e-8),
              metrics=[pf.SDR])

model_CNN.load_weights(models["DenoisingCNN"])
