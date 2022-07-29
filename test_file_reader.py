import argparse, configparser
from svom.params_reader import ParamsReader



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", type=str, help='Config file')

file_arg = parser.parse_args()
reader = ParamsReader(file_arg.config_file, parser)
args = reader.load_parameters_file()
print(args.width*2)

