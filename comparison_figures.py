import os
import yaml
import pickle
import matplotlib.pyplot as plt

# Carrega o arquivo de configurações do teste atual
# Se um arquivo de teste atual não existir pede ao usuário para informar o arquivo de configuração
config_file = './config_files/CurrentTest.yml'
while not os.path.exists(config_file):
    config_file = input('\nInforme um arquivo de configuração válido (ex: Test0.yml): ')
    config_file = './config_files/' + config_file

with open(config_file, 'r') as file:
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

# Nomes dos modelos
CNN_name = 'DenoisingCNN'
CNN_checkpoint_folder = '/home/augustobecker/projects/speech_enhancement/checkpoints/'
CNN_checkpoint_path = CNN_checkpoint_folder + CNN_name + '_model_best_'+ testID +'.h5'

CRNN_name = 'DenoisingCRNN'
CRNN_checkpoint_folder = '/home/augustobecker/projects/speech_enhancement/checkpoints/'
CRNN_checkpoint_path = CRNN_checkpoint_folder + CRNN_name + '_model_best_'+ testID +'.tf'

# Nomes dos arquivos com os resultados das métricas
CNN_curves_file = 'CNN_Curves' + testID
CNN_curves_file = '/home/augustobecker/projects/speech_enhancement/data/' + CNN_curves_file

CRNN_curves_file = 'CRNN_Curves' + testID
CRNN_curves_file = '/home/augustobecker/projects/speech_enhancement/data/' + CRNN_curves_file

print('\nCarregando os dados das curvas...')
with open(CNN_curves_file, 'rb') as f:
    CNN_curves = pickle.load(f)
with open(CRNN_curves_file, 'rb') as f:
    CRNN_curves = pickle.load(f)
print("Pronto!\n")

plt.plot(SNR_list, CNN_curves['Test'] , 'k-' , label = 'CNN')
plt.plot(SNR_list, CRNN_curves['Test'] , 'k--' , label = 'CRNN')

plt.xlabel('SNR de entrada (dB)')
plt.ylabel('SDR de saída (dB)')
plt.legend()
plt.savefig('/home/augustobecker/projects/speech_enhancement/figures/Comp_SDR_Curves_' + testID + '.pdf')