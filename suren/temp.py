import configparser
import os

def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        print(section)
        for key in config[section]:
            print(section, (key, config[section][key]))

file = "../data/temp_files/oxford.ini"
read_ini(file)

print(os.path.exists(file))