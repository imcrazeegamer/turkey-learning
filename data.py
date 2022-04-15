import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold, cross_val_score, train_test_split

RAW_TRAIN_DATA = pd.read_json("data/train.json")
RAW_TEST_DATA = pd.read_json("data/test.json")


def get_audio(data_frame):
    xtrain = list(data_frame['audio_embedding'])
    # Pad the audio features so that all are "10 seconds" long
    x_train = pad_sequences(xtrain, maxlen=10)
    return x_train


def get_labels(extra_dim=False):
    ytrain = RAW_TRAIN_DATA['is_turkey'].values
    y_train = np.asarray(ytrain)
    if extra_dim:
        return y_train[:, None]
    return y_train


def split_validation_train(training_data, training_label):
    return train_test_split(training_data, training_label, train_size=0.8, random_state=1)