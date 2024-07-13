import sys
sys.path.append('./ProteinDRL/')
import os
import local_dir
import numpy as np
import aa_encoding


class StructureEncoder():
	def __init__(self):
		self.preprocessed_folder_path = local_dir.preprocessed_folder_path
		self.encoded_structure_folder_path = local_dir.training_data_folder
		self.aa_letter_converter = aa_encoding.aa_letter_converter()

		self.min_num_residue = 20
		self.max_num_residue = 80
		self._file_processed = 0


	def encoding_algorithm(self, all_residue_array: np.ndarray, num_residue: int):
		#? Input format of each row: [x, y, z, atom_label, aa_type, aa_index]
		alpha_carbon_rows = all_residue_array[all_residue_array[:, 3] == 'CA']
		alpha_carbon_position_array = alpha_carbon_rows[:, 0:3].astype(float)
		amino_acid_identity_array = np.apply_along_axis(self.aa_letter_converter.convert, 1, alpha_carbon_rows[:, 4].reshape(-1, 1)).reshape(-1, 1)

		if alpha_carbon_rows.shape[0] != num_residue: #? Few samples have incomplete last residue, exclude them
			raise ValueError

		complete_encoding_array = np.concatenate([alpha_carbon_position_array, amino_acid_identity_array], axis=-1)
		return complete_encoding_array


	def encode(self): #? Call this function directly to start encoding
		for file_name_with_extension in tuple(os.walk(self.preprocessed_folder_path))[0][2]:
			file_name, _ = file_name_with_extension.split('.')
			input_file_path = self.preprocessed_folder_path + '/' + file_name_with_extension
			output_file_path = self.encoded_structure_folder_path + '/' + file_name
			protein_data_array = np.load(input_file_path)['data']

			try:
				num_residue = np.max(protein_data_array[:, -1].astype(int))
				if num_residue >= self.min_num_residue and num_residue <= self.max_num_residue: # exclude proteins that are too small or too big
					encoded_data = self.encoding_algorithm(protein_data_array, num_residue)
					np.savez_compressed(output_file_path, data = encoded_data) # save as .npz
			except:
				pass
			self._file_processed += 1
			print('Processed:', self._file_processed, end='\r')


StructureEncoder().encode()