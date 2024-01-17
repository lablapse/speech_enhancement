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
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

import project_functions as pf

# %% [markdown]
# # **Configurações Globais**

# %%
# Define a GPU visível (0 -> 3080; 1 -> 3090)
pf.set_gpu(0)

# Carrega o arquivo de configurações do teste atual
with open('./config_files/CurrentTest.yml', 'r') as file:
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
# # 4. **Visualização de resultados**

# %% [markdown]
# > ## 4.1. Avaliação da CNN

# %%
model_name = 'DenoisingCNN'
checkpoint_folder = '/home/augustobecker/projects/speech_enhancement/checkpoints/'
CNN_checkpoint_path = checkpoint_folder + model_name + '_model_best_'+ testID +'.h5'

model_name = 'DenoisingCRNN'
checkpoint_folder = '/home/augustobecker/projects/speech_enhancement/checkpoints/'
CRNN_checkpoint_path = checkpoint_folder + model_name + '_model_best_'+ testID +'.tf'

# %% [markdown]
# > ## 4.2. Avaliação da CRNN

# %%
# Carrega o modelo para os testes
K.clear_session()

model_CRNN = pf.CRNN_model((nfft//2 + 1, time_frames, 1))

model_CRNN.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0015, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1.0e-8),
              metrics=[pf.SDR])

model_CRNN.load_weights(CRNN_checkpoint_path)

# %%
# Reduz o dataset (samples_per_SNR = -1 -> dataset completo)

train_samples_per_SNR = -1
val_samples_per_SNR = -1
test_samples_per_SNR = -1


metrics = {'MSE': pf.MSE_metric,
           'SDR': pf.SDR_metric}
CRNN_curves = {'Train': np.zeros((len(SNR_list),)),
              'Val': np.zeros((len(SNR_list),)),
              'Test': np.zeros((len(SNR_list),)),}

for k in range(len(SNR_list)):
    SNR_str = '/' + str(SNR_list[k]) + '/'
    train_list = pf.reduce_dataset(train_files, noise_list, [SNR_str], train_samples_per_SNR,
                                draw_function = pf.draw_files)
    val_list = pf.reduce_dataset(val_files, noise_list, [SNR_str], val_samples_per_SNR,
                            draw_function = pf.draw_files)
    test_list = pf.reduce_dataset(test_files, noise_list, [SNR_str], test_samples_per_SNR,
                            draw_function = pf.draw_files)
    
    train_gen = pf.full_audio_batch_generator(train_list, nperseg = nperseg, time_frames = time_frames, noverlap=noverlap,
                                           sample_rate = fs, phase_aware_target = phase_aware, train_mode = False)
    val_gen = pf.full_audio_batch_generator(val_list, nperseg = nperseg, time_frames = time_frames, noverlap=noverlap,
                                         sample_rate = fs, phase_aware_target = phase_aware, train_mode = False)
    test_gen = pf.full_audio_batch_generator(test_list, nperseg = nperseg, time_frames = time_frames, noverlap=noverlap,
                                          sample_rate = fs, phase_aware_target = phase_aware, train_mode = False)

    print('Calculando métricas sobre o conjunto de treino medidas no domínio do tempo (em dB)...')
    CRNN_curves['Train'][k] = pf.time_tests(model_CRNN, train_gen, len(train_list), metrics_dict = metrics)['SDR']
    print('Calculando métricas sobre o conjunto de validação medidas no domínio do tempo (em dB)...')
    CRNN_curves['Val'][k] = pf.time_tests(model_CRNN, val_gen, len(val_list), metrics_dict = metrics)['SDR']
    print('Calculando métricas sobre o conjunto de teste medidas no domínio do tempo (em dB)...')
    CRNN_curves['Test'][k] = pf.time_tests(model_CRNN, test_gen, len(test_list), metrics_dict = metrics)['SDR']

print('Pronto!')

curves_file = 'CRNN_Curves' + testID
curves_file = '/home/augustobecker/projects/speech_enhancement/data/' + curves_file

print('\nSalvando os dados das curvas...')
with open(curves_file, 'wb') as f:
    pickle.dump(CRNN_curves, f)
print('Pronto!')

plt.plot(SNR_list, CRNN_curves['Train'], 'k.-', label = 'SDR de Treino')
plt.plot(SNR_list, CRNN_curves['Val']  , 'k--', label = 'SDR de Validação')
plt.plot(SNR_list, CRNN_curves['Test'] , 'k-' , label = 'SDR de Teste')

plt.xlabel('SNR de entrada (dB)')
plt.ylabel('SDR de saída (dB)')
plt.legend()
plt.savefig('/home/augustobecker/projects/speech_enhancement/figures/CRNN_SDR_Curves_' + testID + '.pdf')
