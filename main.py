from data import get_audio, get_labels


def main():
    training_data = get_audio()
    training_labels = get_labels(extra_dim=True)
    print('=== Data ===')
    print(f'training_data: {training_data.shape}')
    print(f'training_labels: {training_labels.shape}')


if __name__ == '__main__':
    main()
