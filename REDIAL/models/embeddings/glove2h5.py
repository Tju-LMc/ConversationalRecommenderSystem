"""Convert raw GloVe word vector text file to h5."""
import h5py
import numpy as np
from pathlib import PurePath

nowpath = PurePath(__file__).parent
print(nowpath / 'glove.840B.300d.txt')

glove_vectors = [
    line.strip().split()
    for line in open(nowpath / 'glove.840B.300d_500m.txt', 'r')
]

vocab = [line[0] for line in glove_vectors]
vectors = np.array(
    [[float(val) for val in line[1:] if type(val) is float] for line in glove_vectors]
).astype(np.float32)
vocab = '\n'.join(vocab)

f = h5py.File(nowpath / 'glove.840B.300d.h5', 'w')
f.create_dataset(data=vectors, name='embedding')
f.create_dataset(data=vocab, name='words_flatten')
f.close()
