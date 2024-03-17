# # **Imports**

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras import backend as K

import project_functions as pf

# # **Configurações Globais**

# Define a GPU visível (0 -> 3080; 1 -> 3090)
pf.set_gpu(0)

test_file = "git/speech_enhancement/config_files/Test3.yml"

my_dir = "/home/pertum/git/speech_enhancement/"

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

# # 1. **Análise exploratória dos dados**

# > ## 1.1. Informações do dataset

Clean_file_list = []
for dirpath,dirnames,filenames in os.walk('/datasets/ntcd_timit/Clean'):
    if dirpath.__contains__('volunteers'): # Linha para ignorar os lipspeakers
        for file in filenames:
            if file.endswith('.wav'):
                Clean_file_list.append(dirpath + '/' + file) 

n_SNRs = 6
n_Noises = 6
print('Number of clean files (Targets): ', len(Clean_file_list))
print('Total number of noisy files (Inputs): ', len(Clean_file_list)*n_SNRs*n_Noises)

# # 2. **Separação do dataset**

# > ## 2.0. Observações

# * Nessa seção, o dataset será separado da seguinte forma:
# > * 70% dos falantes serão separados para treinamento, 20% para validação e 10% para testes
# > * Serão utilizados na avaliação do desempenho de testes e validação somente as frases não utilizadas em nenhum outro conjunto
# > * Realizar testes/validação em outro momento também com frases que ocorrem no treinamento (não inéditas)

# > ## 2.1. Listas de voluntários e frases

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

# > ## 2.2. Separação do dataset

# Separa a lista de voluntários em treino (70%), validação (20%) e teste (10%)
# Separar mais arquivos para teste e validação 
spk_train_val_list, spk_test_list = train_test_split(spk_list, test_size = 0.1, random_state = 0)
spk_train_list, spk_val_list = train_test_split(spk_train_val_list, test_size = 0.2, random_state = 0)

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

total_unique_sentences = len(train_wav_set) + len(val_wav_set) + len(test_wav_set)

print(f'Tamanho relativo do dataset de treinamento: {(len(train_wav_set)/total_unique_sentences):.2%}')
print(f'Tamanho relativo do dataset de validação: {(len(val_wav_set)/total_unique_sentences):.2%}')
print(f'Tamanho relativo do dataset de teste: {(len(test_wav_set)/total_unique_sentences):.2%}')

# print("Exemplo de formato de par de arquivos a ser entregue aos batch generators:")
# print(train_files[0])

# function to split audio and reference
def split_audio_list(audio_list):
    data = {
        'pair': [],
        'reference': [],
        'noisy': [],
        'type': [],
        'SNR': [],
        'gender': [],
        'user_id': [],
        'audio_tag': [],
    }
    for pair in audio_list:
        noisy_filename = pair[0]
        splited_noisy_filename = noisy_filename.split(os.sep)
        data["pair"].append(pair)
        data["reference"].append(pair[1])
        data["noisy"].append(noisy_filename)
        data["type"].append(splited_noisy_filename[-6])
        data["SNR"].append(splited_noisy_filename[-5])
        data["gender"].append(splited_noisy_filename[-3][-1])
        data["user_id"].append(splited_noisy_filename[-3][:-1])
        data["audio_tag"].append(splited_noisy_filename[-1][:-4])

    return data

# get the test set name
test_set = test_file.split(os.sep)[-1][:-4]

# split and save the test dataset
data = split_audio_list(test_files)
test_dataset = pd.DataFrame(data)
test_dataset_filename = my_dir + f"{test_set}_test_dataset.pkl"
test_dataset.to_pickle(test_dataset_filename)

# split and save the validation dataset
data = split_audio_list(val_files)
val_dataset = pd.DataFrame(data)
val_dataset_filename = my_dir + f"{test_set}_val_dataset.pkl"
test_dataset.to_pickle(val_dataset_filename)

# split and save the dataset of training
data = split_audio_list(train_files)
train_dataset = pd.DataFrame(data)
train_dataset_filename = my_dir + f"{test_set}_train_dataset.pkl"
train_dataset.to_pickle(train_dataset_filename)

print("\n\n Done! \n")