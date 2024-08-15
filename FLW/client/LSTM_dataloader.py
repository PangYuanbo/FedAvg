import  torch
import matplotlib.pyplot as plt
import numpy as np
import platform
import pathlib
import os
cache_dir = './tmp'
dataset_file_name = 'shakespeare.txt'
dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
# Create a cache directory.
cache_dir_path = pathlib.Path(cache_dir).absolute()
cache_dir_path.mkdir(parents=True,exist_ok=True)

# Download the dataset.
dataset_file_path = cache_dir_path/dataset_file_name
torch.hub.download_url_to_file(dataset_file_origin,dataset_file_path)

print(f"Dataset file downloaded to {dataset_file_path}")

text = open(dataset_file_path, mode='r').read()
vocab = sorted(set(text))

# Map characters to their indices in vocabulary.
char2index = {char: index for index, char in enumerate(vocab)}
print('  ...\n}')
# Map character indices to characters from vacabulary.
index2char = np.array(vocab)