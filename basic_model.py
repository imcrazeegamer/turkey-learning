import keras.metrics
import numpy as np
import tensorflow as tf
import data as dt


class TurkeyLearning:
    def __init__(self, model_input_shape=None, hidden_layer_size=128, load_model_name=None):
        self._init_models(model_input_shape, hidden_layer_size)
        if load_model_name is not None:
            self.load(load_model_name)

    def train(self, training_data, training_labels, epochs, batch_size=None):
        x_train, x_val, y_train, y_val = dt.split_validation_train(training_data, training_labels)
        # print(f"x_train {len(x_train)}, y_train {len(y_train)}, x_val {len(x_val)}, y_val {len(y_val)}")
        if batch_size is None:
            self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
        else:
            self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size)

    def test(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def save(self, name):
        self.model.save_weights(f"save/{name}")
        print(f"model {name} saved")

    def load(self, name):
        self.model.load_weights(f"save/{name}")
        print(f"model {name} loaded")

    def _init_models(self, input_shape, hidden_size):

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='softmax'),
        ])

        # self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC(from_logits=True, multi_label=False)])
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        # self.model.summary()
