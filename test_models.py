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
from tqdm import tqdm
from scipy import signal

import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

import project_functions as pf

def pesq(y_true, y_predict, fs=16000):
    if fs == 16000:
        pesq = PerceptualEvaluationSpeechQuality(fs, "wb")
    elif fs == 8000:
        pesq = PerceptualEvaluationSpeechQuality(fs, "nb")
    pesq_value = pesq(y_predict, y_true)
    return pesq_value

# get the STFT of the windows
def get_stft_of_windows(audio_data,
                        nperseg = 512,
                        nfft = None,
                        time_frames = 16, 
                        noverlap = None,
                        sample_rate = 16000):
    if nfft == None:
        nfft = nperseg
    if noverlap == None:
        noverlap = nperseg - nperseg//4 # == ceilling(3*nperseg/4)
        
    # Determina a STFT do sinal de áudio completo
    f, t, audio_Zxx = signal.stft(audio_data,
                                  sample_rate,
                                  nperseg = nperseg,
                                  nfft = nfft,
                                  noverlap = noverlap,
                                  window = 'hann')
                
    # Número de espectrogramas a serem extraídos do sinal original
    T = len(t)//time_frames
    # Inicializa arrays auxiliares para armazenar as divisões do espectrograma
    audio_STFT = []
    audio_angle = []

    for i in range(0,len(t)-time_frames):
        Sn = audio_Zxx[:, i : i + time_frames]
        audio_STFT.append(np.abs(Sn))
        audio_angle.append(np.angle(Sn))

    return audio_STFT, audio_angle

# frame a audio data and get the stft
def get_batch_of_STFT_windows(audio_file,
                              sample_rate = 16000,
                              nperseg = 512, nfft = None,
                              time_frames = 16,
                              noverlap = None,
                              train_mode = True):
    if nfft == None:
        nfft = nperseg
    if noverlap == None:
        noverlap = nperseg - nperseg//4 # == ceilling(3*nperseg/4)
    
    # load the audio sample
    audio_data = pf.load_audio_file(audio_file, sample_rate=sample_rate)

    # get STFT
    audio_STFT, audio_angle = get_stft_of_windows(
                                    audio_data,
                                    nperseg = nperseg,
                                    nfft = nfft,
                                    time_frames = time_frames,
                                    noverlap = noverlap,
                                    sample_rate = sample_rate)

    # Adiciona ao batch os espectrogramas adquiridos - com shape (batch_size,nfft//2 + 1,time_frames,1)
    audio_STFT_batch = np.reshape(audio_STFT, np.shape(audio_STFT) + (1,))
    audio_angle_batch = np.reshape(audio_angle, np.shape(audio_STFT) + (1,))
    
    return (np.array(audio_STFT_batch), np.array(audio_angle_batch))

def inference(model, batch, noverlap = None):

    if noverlap == None:
        nperseg = 2*(np.shape(batch[0])[0] - 1)
        noverlap = nperseg - nperseg//4
    
    # batch manipulation
    audio_STFT_batch = batch[0]
    audio_angle_batch = batch[1][:,:,-1,0]
    audio_angle_batch = np.reshape(audio_angle_batch,np.shape(batch[1])[0:2]).T
    
    # predict
    pred_STFT_abs = model.predict(audio_STFT_batch,verbose = 0)
    
    # build signal
    pred_STFT_abs = np.reshape(pred_STFT_abs,np.shape(batch[1])[0:2]).T
    pred_STFT = pred_STFT_abs*np.exp(1j*audio_angle_batch)
    _, audio_signal = signal.istft(pred_STFT, noverlap = noverlap, window = 'hann')
    
    return audio_signal


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


for row in test_dataset.iterrows():
    reference_file = row["reference"]
    noisy_file = row["noisy"]

    batch = get_batch_of_STFT_windows(noisy_file)
    output_signal = inference(model_CNN,
                              batch=batch,
                              noverlap=noverlap,
                              )
    


