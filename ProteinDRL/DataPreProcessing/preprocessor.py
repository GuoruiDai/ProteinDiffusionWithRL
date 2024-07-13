import sys
sys.path.append('./ProteinDRL/')
import Bio.PDB as bp
import os
import numpy as np
import local_dir

#? Parse mmCIF files then convert to 2D numpy array with the following format:
#? Format of each row: [x, y, z, atom_label, aa_type, aa_index]
# only ATOM atoms will be kept (i.e. no ligand or nucleic acid)
# keep only first conformation (_atom_site.label_alt_id that is not '.' or 'A' are removed)


class Preprocessor():
    def __init__(self):
        self.parser = bp.MMCIF2Dict.MMCIF2Dict
        self.raw_mmcif_folder_path = local_dir.raw_mmcif_folder_path
        self.preprocessed_folder_path = local_dir.preprocessed_folder_path
        self.file_processed = 0

    def process_file(self):
        mmCIF_file_list = tuple(os.walk(self.raw_mmcif_folder_path))[0][2]
        for file in mmCIF_file_list:
            file_nameOnly, _ = file.split('.')
            filepath = self.raw_mmcif_folder_path+'/'+file
            data_dict = dict(self.parser(filepath)) # parse mmCIF into python dict

            try:
                structure_matrix = self.convertDataToStructureMatrix(data_dict)
                output = self.processStructureMatrix(structure_matrix)
                np.savez_compressed(self.preprocessed_folder_path+'/'+file_nameOnly, data = output)
            except:
                pass

            self.file_processed += 1
            print('Processed:', self.file_processed, end='\r')


    # takes in parsed dict, convert data into numpy array
    def convertDataToStructureMatrix(self, _dict: dict):
        column_x = _dict['_atom_site.Cartn_x']
        column_y = _dict['_atom_site.Cartn_y']
        column_z = _dict['_atom_site.Cartn_z']
        column_label = _dict['_atom_site.label_atom_id']
        column_aa = _dict['_atom_site.label_comp_id']
        column_atom_group = _dict['_atom_site.group_PDB'] # ATOM is for protein atom, other group would be heterogeneous atoms
        column_alt_conformation = _dict['_atom_site.label_alt_id'] # checks for alternate conformation
        column_chain_id = _dict['_atom_site.label_asym_id'] # asym_id is for unique chain id
        matrix = np.stack((column_x, column_y, column_z, column_label, column_aa, column_atom_group, column_alt_conformation, column_chain_id), axis=-1)
        return matrix


    # input type is dictionary and numpy array
    def processStructureMatrix(self, matrix):
        atom_group_col = matrix[:,5]
        ATOM_row_index = np.where(atom_group_col == 'ATOM')[0]
        matrix = matrix[ATOM_row_index, :] # only keep ATOM rows

        alt_conformation_col = matrix[:,6]
        first_conformation_row_index = np.where((alt_conformation_col == '.') | (alt_conformation_col == 'A'))[0]
        matrix = matrix[first_conformation_row_index, :] # remove rows that are from alternative conformations

        chain_id_col = matrix[:,7]
        first_unique_chain_row_index = np.where(chain_id_col == 'A')[0]
        matrix = matrix[first_unique_chain_row_index, :] # only keeps the first unique chain

        trimmed_matrix = matrix[:, 0:5] # only keep coordinates, atom label and amino acid type information

        #? create amino acid index (1-indexed)
        nitrogen_counter = 0
        aa_index_column = []
        for row in trimmed_matrix[:, 3].tolist():
            if row == 'N': # N represents the first atom/nitrogen of a new amino acid in the file, since it is represented by only letter N
                nitrogen_counter += 1
            aa_index_column.append(nitrogen_counter)
        aa_index_column = np.array([aa_index_column]).reshape(-1,1)
        trimmed_matrix = np.append(trimmed_matrix, aa_index_column, axis=-1)
        return trimmed_matrix


Preprocessor().process_file() # start processing
