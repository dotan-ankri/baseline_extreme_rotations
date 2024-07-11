import numpy as np
import pickle

pairs_file = 'metadata/cambridge_final_colab.npy'
# pairs_file = 'metadata/cambridge_final_full_image_size_after_rotating_fixed_pair_type_to_storage.npy'
pairs = np.load(pairs_file, allow_pickle=True).item()
print(pairs.keys())


