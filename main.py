import data as dt
from basic_model import TurkeyLearning

if __name__ == '__main__':
    training_data = dt.get_audio(dt.RAW_TRAIN_DATA)
    training_labels = dt.get_labels(extra_dim=True)
    testing_data = dt.get_audio(dt.RAW_TEST_DATA)

    print('=== Data ===')
    print(f'training_data: {training_data.shape}')
    print(f'training_labels: {training_labels.shape}')
    print(f'testing_data: {testing_data.shape}')

    tl = TurkeyLearning(model_input_shape=training_data[0].shape)
    tl.train(training_data, training_labels, epochs=20, batch_size=100)
    # tl.save("test")
    # tl.load("test")

    print(f"Predictions {tl.predict(testing_data)}")