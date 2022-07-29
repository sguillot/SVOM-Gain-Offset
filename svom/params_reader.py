import argparse
import configparser


class ParamsReader:
    def __init__(self, params_filepath, parser):
        self.parser = parser
        self.parameters_file = params_filepath
        self.parameters = None
        self.read_parameters()

    def read_parameters(self):
        """
        From the config file, get all available section and associated parameters.
        Transform the config file and configparser object into a dictionary.
        """
        config = configparser.ConfigParser()
        config.read(self.parameters_file)
        available_sections = dict(config.items())
        for section in available_sections.keys():
            parameters_dictionary = dict(config.items(section))
            available_sections[section] = parameters_dictionary
        self.parameters = available_sections
        self.parameters.pop('DEFAULT')

    def parameters_to_arguments_parser(self):
        """
        With each values of the parameter dictionary, format and add argument to the parser.
        """
        for section in self.parameters.values():
            self.parser.add_argument(
                f"--{section['name']}",
                help=section['description'],
                type=self.get_type(section['type']),
                default=section['value']
            )

    def get_type(self, parameter_type):
        """
        With the information od the type in string, return the class of
        the corresponding python type
        """
        type_python = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool
        }.get(parameter_type)
        return type_python

    def load_parameters_file(self):
        """
        This function traduce the config file into arguments parser.
        """
        self.parameters_to_arguments_parser()
        return self.parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help='Config file')
    file_arg = parser.parse_args()
    reader = ParamsReader(file_arg.config_file, parser)
    args = reader.load_parameters_file()
    print(args)
