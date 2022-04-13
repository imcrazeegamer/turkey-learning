import numpy as np
import tensorflow as tf


class TurkeyLearning:
    def __init__(self, model_input_shape=None, hidden_layer_size=128, load_model_name=None):
        self._init_models(model_input_shape, hidden_layer_size)
        if load_model_name is not None:
            self.load(load_model_name)

    def train(self, training_data, training_labels, epochs, batch_size=None):
        if batch_size is None:
            self.model.fit(training_data, training_labels, epochs=epochs)
        else:
            self.model.fit(training_data, training_labels, epochs=epochs, batch_size=batch_size)

    def test(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

    def predict(self, inputs):
        return np.argmax(self.probability_model.predict(inputs), axis=-1)

    def save(self, name):
        self.model.save_weights(f"save/{name}")
        print(f"model {name} saved")

    def load(self, name):
        self.model.load_weights(f"save/{name}")
        print(f"model {name} loaded")

    def _init_models(self, input_shape, hidden_size, dynamic_learning_rate=False):
        STEPS_PER_EPOCH = 100

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2)
        ])

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH * 1000,
            decay_rate=1,
            staircase=False)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule) if dynamic_learning_rate else 'adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[tf.metrics.SparseCategoricalAccuracy()])

        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
