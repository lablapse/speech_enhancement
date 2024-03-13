#################################################################################################################
# Imports                                                                                                       #
#################################################################################################################
import os
import pickle
import librosa
import numpy as np
import pandas as pd
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io.wavfile import write
from fractions import Fraction
from scipy import signal
from tqdm import tqdm

import tensorflow as tf
from multiprocessing import Lock

from tensorflow.keras import Model
from tensorflow.keras.utils import register_keras_serializable, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Input, Add
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, UnitNormalization
from tensorflow.keras.layers import GRU
from tensorflow.keras import backend as K

#################################################################################################################
# GPU                                                                                                           #
#################################################################################################################
# Função para definir a GPU visível para alocação pelo Keras
def set_gpu(gpu_index):
    '''
    Set the GPU to be used by TensorFlow.
    
    :param gpu_index: Index of the GPU to be used (0-based index)
    '''
    # Ensure that the GPU order follows PCI_BUS_ID order
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' 
    
    # List available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    
    # Ensure that the GPU index is within the valid range
    if gpu_index < 0 or gpu_index >= len(physical_devices):
        print('Invalid GPU index.')
        return False
    
    try:
        # Set the visible GPU devices
        tf.config.set_visible_devices(physical_devices[gpu_index:gpu_index + 1], 'GPU')

        # Validate that only the selected GPU is set as a logical device
        assert len(tf.config.list_logical_devices('GPU')) == 1
        
        print(f'GPU {gpu_index} has been set as the visible device.')
        return True
    except Exception as e:
        print(f'An error occurred while setting the GPU: {e}')
        return False

#################################################################################################################
# Dataset                                                                                                       #
#################################################################################################################

tensor = lambda x: tf.convert_to_tensor(x, dtype = tf.float32)

def draw_files(file_list, n_draws):
    # Indices da lista de arquivo
    indexes = range(len(file_list))
    # Função para transformar array de comprimento 2 em tupla
    as_tuple_pair = lambda len2_list: (len2_list[0],len2_list[1])
    
    # Verifica se há arquivos suficientes na lista (ou se o número de arquivos sorteados é válido)
    if n_draws <= len(file_list) and n_draws > 0:
        # Sorteia os índices dos arquivos
        drawn_indexes = np.random.choice(indexes, size=n_draws, replace = False)
        # Determina a lista de arquivos sorteados (as tuplas da lista são perdidas/transformadas em array)
        drawn_list = np.asarray(list(file_list))[drawn_indexes]
        # Restaura as tuplas com os pares de arquivos da lista
        tuple_list = list(map(as_tuple_pair,drawn_list))
        return tuple_list
    else:
        # Retorna a própria lista de arquivo se o número de arquivos sorteados não for válido
        return list(file_list)

def load_audio_file(file_path, sample_rate = None, debug = False):
    audio, fs = librosa.load(file_path, sr=sample_rate, mono=True)
    if debug:
        print(f'Frequência do áudio: {fs} Hz')
    return audio

def compute_dataset_total_batches(file_list, batch_size, nperseg = 512, nfft = None, spectrogram_length = 16,
                                  sample_rate = 16000, noverlap = None):
    if nfft     == None:
        nfft     = nperseg
    if noverlap == None:
        noverlap = nperseg - nperseg//4 # == ceilling(3*nperseg/4)
        
    Total_TFs = 0
    for file_pair in tqdm(file_list):
        audio = load_audio_file(file_pair[0], sample_rate = sample_rate)
        _, t, _ = signal.stft(audio, sample_rate, nperseg = nperseg, 
                              nfft = nfft, noverlap = noverlap, window = 'hann')

        Total_TFs += len(t) - spectrogram_length
        Total_Batches = int(np.ceil(Total_TFs/batch_size))
    return Total_Batches

def reduce_dataset(file_list, noise_list, SNR_list, samples_per_SNR, 
                   draw_function = lambda files, N: files[0:N]):
    # noise_list e SNR_list precisam ser listas com strings contendo o nome das pastas 
    # com cada tipo de ruído e cada SNR a serem separados 
    # (para as SNRs usar o formato '_XX' no Kaggle e '/XX/' no servidor da UTFPR)
    
    # Instacia uma lista para conter o itens selecionados do dataset
    reduced_ds = []
    for noise in noise_list:
        # Encontra os arquivos com cada tipo de ruído
        has_noise = lambda file_pair: file_pair[0].__contains__(noise)
        noise_files = list(filter(has_noise,file_list))
        for SNR in SNR_list:
            # Encontra os arquivos com cada SNR
            has_SNR = lambda file_pair: file_pair[0].__contains__(SNR)
            SNR_files = list(filter(has_SNR,noise_files))
            
            # Seleciona e armazena os itens para cada SNR
            reduced_ds.extend(draw_function(SNR_files, samples_per_SNR))
    
    return reduced_ds

# Pré-processa o áudio, calculando a STFT e determinando os espectrogramas
# Algumas janelas e overlaps mínimos para cada uma:
# https://dsp.stackexchange.com/questions/13436/choosing-the-right-overlap-for-a-window-function
def audio_pre_processing(clean_audio, noisy_audio, nperseg = 512, nfft = None, time_frames = 16, 
                         noverlap = None, sample_rate = 16000, part_signal = False,
                         phase_aware_target = False):
    if nfft == None:
        nfft = nperseg
    if noverlap == None:
        noverlap = nperseg - nperseg//4 # == ceilling(3*nperseg/4)
        
    # Tenta corrigir a amplitude do áudio limpo para coincidir com a amplitude desse sinal presente no áudio corrompido
    #   > Ao estudar o dataset não foi possível concluir se houve ou não um cuidado para manter as amplitudes dos sinais 
    #     limpos coincidindo com a amplitude desses sinais nos arquivos corrompidos. 
    #   > Caso as amplitudes não sejam correspondentes as redes são obrigadas a corrigí-las, adicionando complexidade ao
    #     problema. Outro ponto, como o processamento é feito em janelas, diferenças de amplitude previstas entre uma 
    #     janela e outra introduzem erros no sinal temporal.
    #   > A correção é feita na amplitude do sinal limpo, uma vez que este só é utilizado no treinamento e em situações
    #     práticas seria impossível efetuar a correção da amplitude do sinal corrompido.
    c = np.dot(clean_audio,noisy_audio)/np.linalg.norm(clean_audio)
    clean_audio = c*clean_audio
    
    # Determina a STFT do sinal de áudio completo
    if part_signal:
        f, t, clean_Zxx = signal.stft(clean_audio, sample_rate, nperseg = nperseg,
                                nfft = nfft, noverlap = noverlap, boundary = None, padded= False,
                                window = 'hann')
        f, t, noisy_Zxx = signal.stft(noisy_audio, sample_rate, nperseg = nperseg,
                                nfft = nfft, noverlap = noverlap, boundary = None, padded= False,
                                window = 'hann')
    else:
        f, t, clean_Zxx = signal.stft(clean_audio, sample_rate, nperseg = nperseg,
                                nfft = nfft, noverlap = noverlap, window = 'hann')
        f, t, noisy_Zxx = signal.stft(noisy_audio, sample_rate, nperseg = nperseg,
                                nfft = nfft, noverlap = noverlap, window = 'hann')
                
    # Número de espectrogramas a serem extraídos do sinal original
    T = len(t)//time_frames
    # Inicializa arrays auxiliares para armazenar as divisões do espectrograma
    clean_STFT = []
    clean_angle = []
    noisy_STFT = []
    noisy_angle = []
    # Determina o valor absoluto do espectrograma em dB e separa em (T) ou (len(t)-time_frames) partes de tamanho time_frames
    if phase_aware_target:
        # Determina o espectrograma limpo (target) corrigindo a amplitude de acordo com a regra "phase aware"
        #   Tal regra faz o modelo aprender a ignorar (reduzir a amplitude) de bins de frequência cuja 
        #   diferença de ângulo (predita pelo modelo) entre o sinal limpo e ruidoso seja muito alta, 
        #   melhorando a qualidade do sinal. A regra é dada por |Sp| = max(|Sc|*cos(Theta_c - Theta_n), 0)
        for i in range(0,len(t)-time_frames):
            Sc = clean_Zxx[:, [i + time_frames - 1]] # Espectro limpo
            Sn_aux = noisy_Zxx[:, [i + time_frames - 1]] 
            Sn = noisy_Zxx[:, i : i + time_frames]   # Espectro ruidos
            Sp = np.real(Sc*np.conjugate(Sn_aux))/np.abs(Sn_aux) # == |Sc|*cos(Theta_c - Theta_n)
            # Zera completamente a amplitude de qualquer bin com diferença de fase > 90°
            Sp = Sp*(Sp > 0)
            
            clean_STFT.append(Sp)
            clean_angle.append(np.angle(Sc))
            noisy_STFT.append(np.abs(Sn))
            noisy_angle.append(np.angle(Sn))
    else:
        for i in range(0,len(t)-time_frames):
            Sc = clean_Zxx[:, [i + time_frames - 1]]
            Sn = noisy_Zxx[:, i : i + time_frames]
            
            clean_STFT.append(np.abs(Sc))
            clean_angle.append(np.angle(Sc))
            noisy_STFT.append(np.abs(Sn))
            noisy_angle.append(np.angle(Sn))
                        
    return noisy_STFT, clean_STFT, noisy_angle, clean_angle
            
    
def batch_generator(file_list, batch_size, total_batches, sample_rate = 16000, nperseg = 512, nfft = None,
                    time_frames = 16, noverlap = None, debug = False, random_batches = True, buffer_mult = 20,
                    phase_aware_target = False):
    if nfft == None:
        nfft = nperseg
    if noverlap == None:
        noverlap = nperseg - nperseg//4 # == ceilling(3*nperseg/4)
    
    k = 0;      # Debug
    while True:
        if debug and k > 0: 
            print('Total batches: ', k); k = 0;       # Debug
        
        # Inicializa as listas contendo as amostras excedentes da última chamada ao generator
        # (Armazenadas quando o números de espectrogramas carregados excede o batch_size)
        last_run_clean_STFT = []
        last_run_noisy_STFT = []
        
        # (Re)inicializa a lista de arquivos restantes a serem carregados
        remaining_files = set(file_list)
        
        # Loop para carregar os arquivos da lista, enquanto houverem novos arquivos para tal
        loaded_batches = 0
        while len(remaining_files) > 0 or np.shape(last_run_clean_STFT)[0] > 0:
            # Impede que o generator retorne o último batch, em geral com menos amostras do que batch_size
            # (Para compatibilidade com tf.data.Dataset.from_generator())
            if (not debug) and (loaded_batches == total_batches):
                break
            # Inicializa o batch com as amostras excedentes da última chamada ao gerador
            clean_STFT_batch = last_run_clean_STFT
            noisy_STFT_batch = last_run_noisy_STFT
            
            # Loop para carregar os espectrogramas (amostras) até preencher todo o batch (com batch_size amostras)
            while np.shape(clean_STFT_batch)[0] < buffer_mult*batch_size and len(remaining_files) > 0:
                # Seleciona aleatoriamente um par de arquivos da lista (com um áudio corrompido e seu respectivo áudio limpo)
                file_pair = draw_files(remaining_files, 1)
                # Remove o arquivo da lista (utilizando operações de conjunto)
                remaining_files = remaining_files - set(file_pair)
                # Carrega os áudios do arquivo selecionado
                clean_audio = load_audio_file(file_pair[0][1], sample_rate = sample_rate)
                noisy_audio = load_audio_file(file_pair[0][0], sample_rate = sample_rate)
                
                noisy_STFT, clean_STFT, _, _ = audio_pre_processing(
                    clean_audio, noisy_audio, nperseg = nperseg, nfft = nfft, time_frames = time_frames,  
                    noverlap = noverlap, sample_rate = sample_rate, phase_aware_target = phase_aware_target)
                
                # Adiciona ao batch os espectrogramas adquiridos - com shape (batch_size,nfft//2 + 1,time_frames,1)
                audio_BS = np.shape(clean_STFT)[0]
                clean_STFT_batch.extend(np.reshape(clean_STFT, (audio_BS, nfft//2 + 1, 1, 1)))
                noisy_STFT_batch.extend(np.reshape(noisy_STFT, (audio_BS, nfft//2 + 1, time_frames, 1)))
            
            # Código para embaralhar o batch -> Necessário para entregar diversidade ao batch durante o treinamento
            if random_batches:
                shuffle_mask = np.arange(0,np.shape(clean_STFT_batch)[0],1,dtype = int)
                np.random.shuffle(shuffle_mask)
                clean_STFT_batch = np.array(clean_STFT_batch)[shuffle_mask]
                noisy_STFT_batch = np.array(noisy_STFT_batch)[shuffle_mask]
                
            # Verifica se o batch possui excedente de amostras
            if len(clean_STFT_batch) > batch_size:
                # Atualiza o array com o excedente de amostras
                last_run_clean_STFT = list(clean_STFT_batch[batch_size:])
                last_run_noisy_STFT = list(noisy_STFT_batch[batch_size:])
            else: # len(clean_STFT_batch) <= batch_size
                # Limpa o array com o excedente de amostras
                last_run_clean_STFT = []
                last_run_noisy_STFT = []
            
            # Retorna a tupla com o batch de amostras corrompidas e limpas
            k += 1    # Debug
            loaded_batches += 1

            batch = (tensor(noisy_STFT_batch[0:batch_size]),tensor(clean_STFT_batch[0:batch_size]))
            
            yield batch

def ds_files_list(spk_list, spk_wav_dict, Noise_list, SNR_list, full_ds_folder, exclude_wav = []):
    # Exclude_wav representa uma lista de frases a serem excluídas do conjunto de dados
    
    # Garante que o diretório termine com a barra
    if not full_ds_folder.endswith('/'):
        full_ds_folder = full_ds_folder + '/'
    
    # Diretório referente aos áudios limpos
    clean_path = full_ds_folder + 'Clean/volunteers/'
    
    # Inicializa a lista de arquivos
    file_list = []
    wav_set = set([])
    
    # Loops para a criação da lista de arquivos do conjunto
    for noise in Noise_list:
        for SNR in SNR_list:
            # Define o diretório do subconjunto de dados
            subset_path = full_ds_folder + noise + '/' + SNR + '/volunteers/'
            for speaker in spk_list:
                # Carrega a lista de frases ditas pelo voluntário
                wav_list = spk_wav_dict[speaker]
                # Seleciona as frases válidas para este conjunto de dados
                valid_wavs = set(wav_list) - set(exclude_wav)
                # Atualiza o conjunto de áudios únicos presente no conjunto de dados
                wav_set = wav_set | set(valid_wavs)
                for wav in valid_wavs:
                    # Define o diretório de cada arquivo e adiciona à lista
                    noisy_file = subset_path + speaker + '/straightcam/' + wav
                    clean_file = clean_path + speaker + '/straightcam/' + wav
                    file_list.append((noisy_file, clean_file))
    return file_list, wav_set

#################################################################################################################
# Modelo                                                                                                        #
#################################################################################################################

def get_normalization_parameters(train_generator, n_batches):
    # Inicializa as variáveis necessárias
    total_samples = 0
    norm_params = {
        'mean': 0,
        'var': 0,
                  }
    # Percorre os batches do conjunto de treinamento
    for n in tqdm(range(n_batches)):
        batch = next(train_generator)
        
        # Percorre as amostras do batch
        for sample in batch[0]:
            # Atualiza a contagem de amostras
            total_samples += 1
            
            # Atualiza a estimativa da média
            norm_params['mean'] += sample
            
            # Atualiza a estimativa da auto-correlação
            norm_params['var'] += sample**2
            
    # Determina a média
    norm_params['mean'] = norm_params['mean']/total_samples
    
    # Determina a variância
    norm_params['var'] = norm_params['var']/total_samples - norm_params['mean']**2
    
    return norm_params

@tf.keras.utils.register_keras_serializable()     # Necessário para salvar os pesos definidos na layer
class Normalize_input(tf.keras.layers.Layer):
    def __init__(self, norm_params, **kwargs):
        self.norm_params = norm_params
        super(Normalize_input, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.mean = self.add_weight(name = 'mean',shape = input_shape[1:], initializer = "zeros", trainable = False)
        self.std = self.add_weight(name = 'std_deviation',shape = input_shape[1:], initializer = "zeros", trainable = False)
        self.mean.assign(self.norm_params['mean'])
        self.std.assign(self.norm_params['var']**.5)
        
    def call(self, inputs):
        return (inputs - self.mean)/self.std
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self.mean,
            "std": self.std,
        })
        return config

def CR_CED_model(input_shape, norm_params = None, n_reps = 5, skip = True):
    length = input_shape[1]
    
    i = Input(input_shape)
    if norm_params != None:
        norm_i = Normalize_input(norm_params)(i)
        x = norm_i
    else:
        x = i
    
    kwargs = {'kernel_initializer': 'glorot_uniform',
              'bias_initializer': 'zeros'}
    
    skip_vertices = [x] 
    
    for k in range(n_reps):
        if skip and k > 0:
            x = Add()([skip_vertices[k-1], skip_vertices[k]])
        else:
            x = skip_vertices[k]
        x = Conv2D(18, (9, length),padding='valid', use_bias = False, **kwargs)(x)
        x = BatchNormalization()(x)
        x = ReLU(negative_slope=0.01)(x)
        x = Conv2D(30, (5, 1),padding='same', use_bias = False,**kwargs)(x)
        x = BatchNormalization()(x)
        x = ReLU(negative_slope=0.01)(x)
        x = Conv2DTranspose(length, (9, 1),padding='valid', use_bias = False, **kwargs)(x)
        x = BatchNormalization()(x)
        x = ReLU(negative_slope=0.01)(x)
        #x = Dropout(0.1)(x)
        if k < n_reps - 1:
            # Faz o reshape de (129,1,8) para (129,8,1), mantendo a estrutura da próxima rede R-CED
            x = Reshape(input_shape)(x)
            # Salva o próximo ponto de origem dos dados da conexão skip
            skip_vertices.append(x) 
    
    x = Conv2D(1, (input_shape[0], 1), padding='same',**kwargs)(x)
    
    model = Model(i, x, name = 'CR_CED_Model')
    
    return model

def CRNN_model(input_shape):
    time_frames = input_shape[1]

    model_CRNN = Sequential(
        [
            Input(shape = input_shape),
            Conv2D(90, kernel_size=(9, 1), strides=(3, 1), padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'),
            BatchNormalization(),  
            Activation('relu'),
            Dropout(0.3),
            Conv2D(90, kernel_size=(3, 2), strides=(2, 1), padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            Reshape((time_frames-1, -1)),
            GRU(256, return_sequences=True),
            GRU(256, return_sequences=True),
            Reshape((256,time_frames-1,1)),
            Conv2DTranspose(8, (5, 1), strides=(2, 1), padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            Conv2DTranspose(1, (3, 1), strides=(1, 1),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            Reshape((1, -1)),
            Dense(input_shape[0], activation='sigmoid'),
            Reshape((-1, 1, 1))
        ]
    )

    #modelCRN.summary()
    
    return model_CRNN

# Signal to distortion ratio metric (in dB)
def SDR(y_true, y_pred):
    P_distortion = K.mean(K.square(y_true - y_pred))
    P_clean_signal = K.mean(K.square(y_true))
    return 10*K.log(P_clean_signal/P_distortion)/K.log(10.0)

#################################################################################################################
# Testes                                                                                                        #
#################################################################################################################

# Gerador de batches para carregamento de um áudio completo em único batch, para testes somente
def full_audio_batch_generator(file_list, sample_rate = 16000, nperseg = 512, nfft = None,
                               time_frames = 16, noverlap = None, phase_aware_target = False):
    if nfft == None:
        nfft = nperseg
    if noverlap == None:
        noverlap = nperseg - nperseg//4 # == ceilling(3*nperseg/4)
    
    while True:
        # (Re)inicializa a lista de arquivos restantes a serem carregados
        remaining_files = set(file_list)
        
        while len(remaining_files) > 0:
            # Seleciona aleatoriamente um par de arquivos da lista (com um áudio corrompido e seu respectivo áudio limpo)
            file_pair = draw_files(remaining_files, 1)
            # Remove o arquivo da lista (utilizando operações de conjunto)
            remaining_files = remaining_files - set(file_pair)
            # Carrega os áudios do arquivo selecionado
            clean_audio = load_audio_file(file_pair[0][1], sample_rate = sample_rate)
            noisy_audio = load_audio_file(file_pair[0][0], sample_rate = sample_rate)
                
            noisy_STFT, clean_STFT, noisy_angle, clean_angle = audio_pre_processing(
                    clean_audio, noisy_audio, nperseg = nperseg, nfft = nfft, time_frames = time_frames,  
                    noverlap = noverlap, sample_rate = sample_rate, phase_aware_target = phase_aware_target
                    )
            
            # Adiciona ao batch os espectrogramas adquiridos - com shape (batch_size,nfft//2 + 1,time_frames,1)
            clean_STFT_batch = np.reshape(clean_STFT, np.shape(clean_STFT) + (1,))
            noisy_STFT_batch = np.reshape(noisy_STFT, np.shape(noisy_STFT) + (1,))
            clean_angle_batch = np.reshape(clean_angle, np.shape(clean_STFT) + (1,))
            noisy_angle_batch = np.reshape(noisy_angle, np.shape(noisy_STFT) + (1,))
            
            yield (tensor(noisy_STFT_batch),tensor(clean_STFT_batch),
                   tensor(noisy_angle_batch),tensor(clean_angle_batch))
                
#            batch = (tensor(noisy_STFT_batch[0:batch_size]),tensor(clean_STFT_batch[0:batch_size]))
#            
#            yield batch

# Função para reconstruir os sinais a partir dos batches com espectrogramas do áudio ruidoso
# e espectros do áudio limpo
def reconstruct_signal(model, batch, noverlap = None):    
    if noverlap == None:
        nperseg = 2*(np.shape(batch[0])[0] - 1)
        noverlap = nperseg - nperseg//4
    
    # --------- Noisy and Predicted Signals --------
    noisy_STFT_batch = batch[0]
    noisy_angle_batch = batch[2][:,:,-1,0]
    noisy_angle_batch = np.reshape(noisy_angle_batch,np.shape(batch[1])[0:2]).T
        
    pred_STFT_abs = model(noisy_STFT_batch)  # Chama o modelo diretamente, teoricamente evita os memory leaks de model.predict()
    pred_STFT_abs = np.reshape(pred_STFT_abs,np.shape(batch[1])[0:2]).T
    
    noisy_STFT_abs = batch[0][:,:,-1,0]
    noisy_STFT_abs = np.reshape(noisy_STFT_abs,np.shape(batch[1])[0:2]).T
    
    pred_STFT = pred_STFT_abs*np.exp(1j*noisy_angle_batch)
    noisy_STFT = noisy_STFT_abs*np.exp(1j*noisy_angle_batch)
    
    _,pred_signal = signal.istft(pred_STFT, noverlap = noverlap, window = 'hann')
    _,noisy_signal = signal.istft(noisy_STFT, noverlap = noverlap, window = 'hann')
        
    # --------- Clean Signal --------    
    clean_STFT_abs = batch[1]
    clean_angle = batch[3]
    clean_STFT_abs = np.reshape(clean_STFT_abs,np.shape(batch[1])[0:2]).T
    clean_angle = np.reshape(clean_angle,np.shape(batch[1])[0:2]).T
        
    clean_STFT = clean_STFT_abs*np.exp(1j*clean_angle)
        
    _, clean_signal = signal.istft(clean_STFT, noverlap = noverlap, window = 'hann')
    
    return noisy_signal, pred_signal, clean_signal

def SDR_metric(y_true, y_pred, fs = 16000):
    return 10*np.log10(np.mean(y_true**2)/np.mean((y_true - y_pred)**2))

def MSE_metric(y_true, y_pred, fs = 16000):
    return 10*np.log10(np.mean((y_true - y_pred)**2))
    
def time_tests(model, data, n_batches, metrics_dict = {'MSE': MSE_metric}, fs = 16000, noverlap = 64):
    # Monta um dicionário com as métricas ({"nome": list(Métricas para cada batch)})
    scores = {metric: [] for metric in metrics_dict}
    
    for k in tqdm(range(n_batches)):
        batch = next(data)
        
        # Reconstroi realiza o denoising e reconstroi os sinais do batch carregado
        _, pred_signal, clean_signal = reconstruct_signal(model, batch, noverlap = noverlap)
        
        # Roda as métricas selecionadas
        for metric in metrics_dict:
            scores[metric].append(metrics_dict[metric](clean_signal, pred_signal))
        
    # Calcula a média de todos os batches para cada métrica
    for metric in scores:
        scores[metric] = np.mean(scores[metric])
    
    return scores