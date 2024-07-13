import numpy as np


class aa_letter_converter(): # bi-map between 3-letter and 1-letter aa name
    def __init__(self):
        self.aa_three_letter_list = [
            'ALA',
            'ARG',
            'ASN',
            'ASP',
            'CYS',
            'GLN',
            'GLU',
            'GLY',
            'HIS',
            'ILE',
            'LEU',
            'LYS',
            'MET',
            'PHE',
            'PRO',
            'SER',
            'THR',
            'TRP',
            'TYR',
            'VAL',
        ]
        self.aa_one_letter_list = [
            'A',
            'R',
            'N',
            'D',
            'C',
            'Q',
            'E',
            'G',
            'H',
            'I',
            'L',
            'K',
            'M',
            'F',
            'P',
            'S',
            'T',
            'W',
            'Y',
            'V',
        ]
    def convert(self, input: str):
        if type(input) == np.ndarray:
            input = input[0].astype(str)
        if len(input) == 1:
            index = self.aa_one_letter_list.index(input)
            return self.aa_three_letter_list[index]
        elif len(input) == 3:
            index = self.aa_three_letter_list.index(input)
            return self.aa_one_letter_list[index]
        else:
            raise ValueError


class aa_bit_converter(): # bi-map between aa identity and aa bit encoding
    def __init__(self):
        self.aa_list = [
            'A',
            'R',
            'N',
            'D',
            'C',
            'Q',
            'E',
            'G',
            'H',
            'I',
            'L',
            'K',
            'M',
            'F',
            'P',
            'S',
            'T',
            'W',
            'Y',
            'V',
        ]
        self.bit_array = np.array([
            [ 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1,-1],
            [ 1, 1, 1,-1, 1],
            [ 1, 1, 1,-1,-1],
            [ 1, 1,-1, 1, 1],
            [ 1, 1,-1, 1,-1],
            [ 1, 1,-1,-1, 1],
            [ 1, 1,-1,-1,-1],
            [ 1,-1, 1, 1, 1],
            [ 1,-1, 1, 1,-1],
            [ 1,-1, 1,-1, 1],
            [ 1,-1, 1,-1,-1],
            [ 1,-1,-1, 1, 1],
            [ 1,-1,-1, 1,-1],
            [ 1,-1,-1,-1, 1],
            [ 1,-1,-1,-1,-1],
            [-1, 1, 1, 1, 1],
            [-1, 1, 1, 1,-1],
            [-1, 1, 1,-1, 1],
            [-1, 1, 1,-1,-1],
        ])

    def convert_aa_to_bit(self, input, reverse = False):
        if reverse == False: # aa to bit
            index = self.aa_list.index(input.astype(str))
            return self.bit_array[index, :]
        elif reverse == True: # bit to aa
            try:
                index = np.where(np.all(self.bit_array == input, axis=1))[0][0]
            except:
                index = 5 # if no match, return Glutamine
            return self.aa_list[index]





