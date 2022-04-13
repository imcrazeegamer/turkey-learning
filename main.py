from data import get_audio, get_labels
from basic_model import TurkeyLearning

if __name__ == '__main__':
    training_data = get_audio()
    training_labels = get_labels(extra_dim=True)
    print('=== Data ===')
    print(f'training_data: {training_data.shape}')
    print(f'training_labels: {training_labels.shape}')

    tl = TurkeyLearning(model_input_shape=training_data[0].shape)
    tl.train(training_data, training_labels, epochs=10)
    tl.save("test")
    tl.load("test")
    print(f"Predictions {tl.predict(training_data[:10])}")