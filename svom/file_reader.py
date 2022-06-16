"""
Class to decode parameters files.
"""
import json


class FileReader:
    """
    FileReader class
    """
    def __init__(self, path):
        """
        Constructor for FileReader
        :param path: the path to the parameters file
        """
        self.path = path
        self.params_list = []
        self.content_to_list()

    def content_to_list(self):
        """
        Open the parameters file and format each line to create a two dimensional list of parameters
        :return: None
        """
        try:
            with open(self.path, 'r') as file:
                for line in file:
                    if len(line) > 1 and not line.startswith('#'):
                        self.line_formatting(line)
        except FileNotFoundError:
            pass

    @staticmethod
    def cleaning_string(character_string: str):
        """
        With a word or a character string, remove all redundant quotations, the punctuation for
        line breaks, and the unnecessary blanks.
        :param character_string: a character string
        :return: a string
        """
        new_character_string = character_string.replace('"', '')    # Prevents redundant quotations
        if new_character_string[-2:] == '/n':
            new_character_string = new_character_string[:-2]
        no_blank_string = " ".join(new_character_string.split())
        return no_blank_string

    def line_formatting(self, line: str):
        """
        With a comma as separator, create a list with the character string.
        If the comma is between double quotes, it's not use as separator.
        :param line: a character string
        :return: None
        """
        character_list = []
        words_list = []
        character_string = False
        for character in line:
            # marks the beginning or the ending of a character string with possible commas in it
            if character == '"':
                character_string = False if character_string else True

            # if we're not in a character string, the comma separate the parameters in the line
            if character == "," and character_string is False:
                new_string = ''.join(character_list)
                words_list.append(self.cleaning_string(new_string))
                character_list.clear()
            else:
                character_list.append(character)
        # add the last character string to the list
        new_string = ''.join(character_list)
        words_list.append(self.cleaning_string(new_string))
        self.params_list.append(words_list)

    def params_serializer(self):
        """
        Serialize the parameters list into a string.
        :return: a string
        """
        return json.dumps(self.params_list)
