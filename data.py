import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


RAW_DATA = pd.read_json("data/train.json")


def get_audio():
    xtrain = list(RAW_DATA['audio_embedding'])
    # Pad the audio features so that all are "10 seconds" long
    x_train = pad_sequences(xtrain, maxlen=10)
    return x_train


def get_labels(extra_dim=False):
    ytrain = RAW_DATA['is_turkey'].values
    y_train = np.asarray(ytrain)
    if extra_dim:
        return y_train[:, None]
    return y_train
