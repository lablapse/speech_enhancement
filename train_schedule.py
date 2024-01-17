from shutil import copy
import subprocess
import os

config_folder = '/home/augustobecker/projects/speech_enhancement/config_files/'
current_config = '/home/augustobecker/projects/speech_enhancement/config_files/CurrentTest.yml'
config_files = os.listdir(config_folder)
print(config_files)

input("Pressione \"enter\" para continuar...")

for cfg in config_files:
    copy(config_folder + cfg, current_config)
    subprocess.run(["python3", "cnn_pipeline.py"], check = True)
    subprocess.run(["python3", "crnn_pipeline.py"], check = True)
    subprocess.run(["python3", "cnn_tests.py"], check = True)
    subprocess.run(["python3", "crnn_tests.py"], check = True)
    
os.remove(current_config)