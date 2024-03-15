# %% [markdown]
# # Projeto Final da Disciplina de Aprendizado de Máquina
# ### **Tema**: Comparação de desempenho entre modelos de aprimoramento de fala utilizando redes neurais
# 
# ---
# 
# **Estudantes**: Augusto Cesar Becker &emsp; &emsp; &emsp; &emsp; **Data**: 06/07/2023  
# &emsp; &emsp; &emsp; &emsp; &nbsp; Gabriel Saatkamp Lazaretti
# 
# 
# **Descrição**: Comparação de desempenho entre modelos de aprimoramento de fala aplicados ao dataset NTCD-TIMIT. Os modelos testados são um rede neural convolucional e uma rede recorrente baseada em camadas LSTM.
# 
# **Dataset**: [NTCD-TIMIT](https://zenodo.org/record/1172064)

# %% [markdown]
# # **Imports**

# %%
import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

from tensorflow.data import Dataset as tf_ds
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import project_functions as pf

# %% [markdown]
# # **Configurações Globais**

# %%
# Define a GPU visível (0 -> 3080; 1 -> 3090)
pf.set_gpu(1)

# Carrega o arquivo de configurações do teste atual
with open('./config_files/CurrentTest.yml', 'r') as file:
#with open('./config_files/Test0.yml', 'r') as file:
    test_config = yaml.safe_load(file)

# Variáveis globais
fs = test_config['model']['fs']
nperseg = test_config['model']['nperseg']
nfft = test_config['model']['nfft']
noverlap = test_config['model']['noverlap']
window = test_config['model']['window']
time_frames = test_config['model']['time_frames']
phase_aware = test_config['model']['phase_aware']
batch_size = test_config['model']['batch_size']
epochs = test_config['model']['epochs']
random = test_config['model']['random']
buff_mult = test_config['dataset']['buff_mult']

noise_list = test_config['dataset']['noise_list']
SNR_list = test_config['dataset']['SNR_list']

testID = test_config['testID']

# %% [markdown]
# # 1. **Análise exploratória dos dados**

# %% [markdown]
# > ## 1.1. Informações do dataset

# %%
Clean_file_list = []
for dirpath,dirnames,filenames in os.walk('/datasets/ntcd_timit/Clean'):
    if dirpath.__contains__('volunteers'): # Linha para ignorar os lipspeakers
        for file in filenames:
            if file.endswith('.wav'):
                Clean_file_list.append(dirpath + '/' + file) 

# %%
n_SNRs = 6
n_Noises = 6
print('Number of clean files (Targets): ', len(Clean_file_list))
print('Total number of noisy files (Inputs): ', len(Clean_file_list)*n_SNRs*n_Noises)

# %% [markdown]
# # 2. **Separação do dataset**

# %% [markdown]
# > ## 2.0. Observações

# %% [markdown]
# * Nessa seção, o dataset será separado da seguinte forma:
# > * 70% dos falantes serão separados para treinamento, 20% para validação e 10% para testes
# > * Serão utilizados na avaliação do desempenho de testes e validação somente as frases não utilizadas em nenhum outro conjunto
# > * Realizar testes/validação em outro momento também com frases que ocorrem no treinamento (não inéditas)

# %% [markdown]
# > ## 2.1. Listas de voluntários e frases

# %%
# Define a pasta do subconjunto de dados com os áudios limpos
clean_folder_path = '/datasets/ntcd_timit/Clean'

# Lista as pastas correspondentes a cada voluntário
os.chdir(clean_folder_path + '/volunteers')
spk_list = os.listdir()

# Dataframe com todas as frases distintas, número de ocorrências e voluntários que as gravaram 
wav_df = pd.DataFrame(columns = ['file', 'occurrences'] + spk_list)
wav_df = wav_df.set_index('file')

# Dicionário contendo as frases ditas por cada voluntário
spk_wav_dict = {}

# Obtém as frases ditas, bem como seus dados no dataset
for speaker in spk_list:
    # Varre as pastas de cada voluntário em busca das frases
    os.chdir(clean_folder_path + '/volunteers/' + speaker + '/straightcam')
    spk_wav_list = os.listdir()
    
    # Adiciona as frases do diretório ao correspondente voluntário no dicionário
    spk_wav_dict[speaker] = spk_wav_list
    
    # Conta e registra as ocorrências de cada frase
    for wav in spk_wav_list:
        mask = list(map(lambda cmp: int(cmp == speaker), spk_list))
        if not wav_df.index.__contains__(wav):
            wav_df.loc[wav] = [1] + mask
        else:
            wav_df.loc[wav] += np.asarray([1] + mask)

# %% [markdown]
# > ## 2.2. Separação do dataset

# %%
# Separa a lista de voluntários em treino (70%), validação (20%) e teste (10%)
# Separar mais arquivos para teste e validação 
spk_train_val_list, spk_test_list = train_test_split(spk_list, test_size = 0.1, random_state = 0)
spk_train_list, spk_val_list = train_test_split(spk_train_val_list, test_size = 0.2, random_state = 0)

# %%
# Define a pasta do subconjunto de dados com o dataset copleto
full_ds_folder = '/datasets/ntcd_timit/'

# Listas de SNRs e ruídos de acordo com as pastas do dataset
SNR_classes_list = ['-5', '0', '5', '10', '15', '20']
Noise_classes_list = ['Babble', 'Cafe', 'Car', 'LR', 'Street', 'White']

# Constrói a lista com todos os arquivos a serem utilizados no treinamento          
train_files, train_wav_set = pf.ds_files_list(spk_train_list, spk_wav_dict, 
                                           Noise_classes_list, SNR_classes_list, full_ds_folder)

# Constrói a lista com todos os arquivos a serem utilizados na validação        
val_files, val_wav_set = pf.ds_files_list(spk_val_list, spk_wav_dict, Noise_classes_list, 
                                       SNR_classes_list, full_ds_folder, exclude_wav = train_wav_set)

# Constrói a lista com todos os arquivos a serem utilizados nos teste   
test_files, test_wav_set = pf.ds_files_list(spk_test_list, spk_wav_dict, Noise_classes_list, 
                                         SNR_classes_list, full_ds_folder, exclude_wav = train_wav_set | val_wav_set)

# Mostra a distribuição final dos arquivos nos datasets
print('Total de arquvivos para treino: ', len(train_files))
print('Sentenças únicas para treino: ', len(train_wav_set))
print('-'*50)
print('Total de arquvivos para validação: ', len(val_files))
print('Sentenças únicas para validação: ', len(val_wav_set))
print('-'*50)
print('Total de arquvivos para teste: ', len(test_files))
print('Sentenças únicas para teste: ', len(test_wav_set))

# %%
total_unique_sentences = len(train_wav_set) + len(val_wav_set) + len(test_wav_set)

print(f'Tamanho relativo do dataset de treinamento: {(len(train_wav_set)/total_unique_sentences):.2%}')
print(f'Tamanho relativo do dataset de validação: {(len(val_wav_set)/total_unique_sentences):.2%}')
print(f'Tamanho relativo do dataset de treinamento: {(len(test_wav_set)/total_unique_sentences):.2%}')

# %%
print("Exemplo de formato de par de arquivos a ser entregue aos batch generators:")
print(train_files[0])

# %% [markdown]
# # 3. **Treino da Rede Neural Convolucional**

# %% [markdown]
# > ## 3.1. Rede Neural Convolucional Recorrente (CRNN)

# %%
model_name = 'DenoisingCRNN'

# Reduz o dataset (samples_per_SNR = -1 -> dataset completo)
SNR_list = ['/'+ str(snr) + '/' for snr in SNR_list]
noise_list = ['/' + noise + '/' for noise in noise_list]

train_samples_per_SNR = -1
val_samples_per_SNR = -1
test_samples_per_SNR = -1

train_list = pf.reduce_dataset(train_files, noise_list, SNR_list, train_samples_per_SNR,
                            draw_function = pf.draw_files)
val_list = pf.reduce_dataset(val_files, noise_list, SNR_list, val_samples_per_SNR,
                          draw_function = pf.draw_files)
test_list = pf.reduce_dataset(test_files, noise_list, SNR_list, test_samples_per_SNR,
                          draw_function = pf.draw_files)

print('Calculando o total de batches de treinamento...')
batches_per_epoch = pf.compute_dataset_total_batches(train_list, batch_size, spectrogram_length = time_frames, sample_rate = fs,
                                                  noverlap = noverlap, nperseg = nperseg, window = window)
print('Pronto!')
print('Calculando o total de batches de validação...')
validation_steps = pf.compute_dataset_total_batches(val_list, batch_size, spectrogram_length = time_frames, sample_rate = fs,
                                                 noverlap = noverlap, nperseg = nperseg, window = window)
print('Pronto')

def train_gen():
    ref_gen = pf.batch_generator(train_list, batch_size, total_batches = batches_per_epoch - 1, time_frames = time_frames, 
                              phase_aware_target = phase_aware, random_batches = random, sample_rate = fs, noverlap = noverlap,
                              nperseg = nperseg, buffer_mult = buff_mult, window = window)
    while True:
        yield next(ref_gen)

def val_gen():
    ref_gen = pf.batch_generator(val_list, batch_size, total_batches = validation_steps - 1, time_frames = time_frames,
                              phase_aware_target = phase_aware, random_batches = False, sample_rate = fs, noverlap = noverlap,
                              nperseg = nperseg, buffer_mult = buff_mult, window = window)
    while True:
        yield next(ref_gen) 

train_ds = tf_ds.from_generator(train_gen, output_signature = 
                                (tf.TensorSpec(shape = (batch_size, nfft//2 + 1, time_frames, 1), dtype = tf.float32),
                                 tf.TensorSpec(shape = (batch_size, nfft//2 + 1,           1, 1), dtype = tf.float32)))
val_ds = tf_ds.from_generator(val_gen, output_signature = 
                              (tf.TensorSpec(shape = (batch_size, nfft//2 + 1, time_frames, 1), dtype = tf.float32),
                               tf.TensorSpec(shape = (batch_size, nfft//2 + 1,           1, 1), dtype = tf.float32)))

buff = 20
train_ds.prefetch(buffer_size = buff)
val_ds.prefetch(buffer_size = buff)

print('Total de batches de treinamento: ', batches_per_epoch)
print('Total de batches de validação:   ', validation_steps )
print('Total de arquivos de áudio de treinamento:', len(train_list))
print('Total de arquivos de áudio de validação:  ', len(val_list)  )

# %%
K.clear_session()

model_CRNN = pf.CRNN_model((nfft//2 + 1, time_frames, 1))

model_CRNN.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1.0e-8),
              metrics=[pf.SDR])

# Print the model summary anda save a image file with the model diagram
model_CRNN.summary()

dot_img_file = '/home/augustobecker/projects/speech_enhancement/figures/' + model_name + '_diagram.png'
tf.keras.utils.plot_model(model_CRNN, to_file=dot_img_file, show_shapes=True)

# %%
checkpoint_folder = '/home/augustobecker/projects/speech_enhancement/checkpoints/'
log_folder = '/home/augustobecker/projects/speech_enhancement/logs/'
CRNN_checkpoint_path = checkpoint_folder + model_name + '_model_best_'+ testID +'.tf'
CRNN_log_path = log_folder + model_name + '_model_best_'+ testID +'.csv'

if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

# %%
callbacks = [#EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10),
             ReduceLROnPlateau(monitor='val_SDR', factor=0.5, min_delta = 0.2, patience=4, mode='max', min_lr=1e-6,),
             ModelCheckpoint(filepath=CRNN_checkpoint_path, save_best_only = True, save_format='tf', monitor='val_SDR', mode='max'),
             #ModelCheckpoint(filepath=CNN_checkpoint_path, save_best_only = True, save_weights_only = True),
             CSVLogger(filename=CRNN_log_path, separator=',', append=False)
            ]

history_CRNN = model_CRNN.fit(train_ds,
                    batch_size = batch_size,
                    epochs = epochs,
                    steps_per_epoch = batches_per_epoch,
                    callbacks = callbacks,
                    validation_data = val_ds,
                    validation_steps = validation_steps,
                    max_queue_size = 10,
                    workers = 1,
                    use_multiprocessing = False,
                    verbose = 1)


plt.subplots(1,2,figsize = (10,5));

plt.subplot(1,2,1);
plt.plot(10*np.log10(history_CRNN.history['loss']),'--k',label = 'MSE de Treino');
plt.plot(10*np.log10(history_CRNN.history['val_loss']),'-k',label = 'MSE de Validação');

plt.xlabel('Épocas');
plt.ylabel('MSE (dB)');
plt.title('Curvas de MSE');
plt.legend();

plt.subplot(1,2,2)
plt.plot(history_CRNN.history['SDR'],'--k',label = 'SDR de Treino');
plt.plot(history_CRNN.history['val_SDR'],'-k',label = 'SDR de Validação');

plt.xlabel('Épocas');
plt.ylabel('SDR (dB)');
plt.title('Curvas de SDR');
plt.legend();

plt.suptitle('Curvas de aprendizagem (CNN)')
plt.tight_layout()
plt.savefig('/home/augustobecker/projects/speech_enhancement/figures/CRNN_History_' + testID + '.pdf')