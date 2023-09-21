import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
from tensorflow.lite.python.util import convert_bytes_to_c_source

SAMPLES = 1500


def print_ver():
    print(tf.__version__)


def create_values():
    input_values = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES)
    plt.plot(input_values)
    plt.show()

    # We randomize x more and add output_values which is cosine value of x
    np.random.shuffle(input_values)
    output_values = np.cos(input_values)
    plt.plot(input_values, output_values, 'b.')
    plt.show()

    # We will add more noise
    output_values += 0.1 * np.random.randn(*output_values.shape)

    plt.plot(input_values, output_values, 'b.')
    plt.show()
    return input_values, output_values


def split_datas(input_val, output_val):
    train_split = int(0.6 * SAMPLES)
    test_split = int(0.2 * SAMPLES + train_split)
    x_train, x_test, x_validate = np.split(input_val, [train_split, test_split])
    y_train, y_test, y_validate = np.split(output_val, [train_split, test_split])

    assert SAMPLES == (x_train.size + x_validate.size + x_test.size)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(x_train, y_train, 'b.', label="Train")
    # plt.plot(x_train, y_train, 'b.', label="Train")
    axis[0, 0].set_title("Train Data")
    axis[0, 0].legend()

    axis[0, 1].plot(x_validate, y_validate, 'y.', label="Validate")
    # plt.plot(x_validate, y_validate, 'y.', label="Validate")
    axis[0, 1].set_title("Validation Data")
    axis[0, 1].legend()

    axis[1, 0].plot(x_test, y_test, 'r.', label="Test")
    # plt.plot(x_test, y_test, 'r.', label="Test")
    axis[1, 0].set_title("Test Data")
    axis[1, 0].legend()
    plt.show()
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def plot_loss(info):
    loss = info.history['loss']
    validation_loss = info.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'g-', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print_ver()
    input_data, output_data = create_values()
    input_train, input_validate, input_test, output_train, output_validate, output_test = split_datas(input_data,
                                                                                                      output_data)
    # Designing the model
    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1))  # output layer
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    training_info = model.fit(input_train, output_train, epochs=350, batch_size=64,
                              validation_data=(input_validate, output_validate))
    plot_loss(training_info)

    # Prediction plot
    plot_loss = model.evaluate(input_test, output_test)
    predictions = model.predict(input_test)

    plt.clf()
    plt.title('Comparison of predictions and actual values')
    plt.plot(input_test, output_test, 'b.', label='Actual')
    plt.plot(input_test, predictions, 'r.', label='Predicted')
    plt.legend()
    plt.show()

    # We will convert our model to tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    open("cosinewave_model.tflite", "wb").write(tflite_model)

    source_text, header_text = convert_bytes_to_c_source(tflite_model, "cosine_model")

    with open('cosine_model.h', 'w') as file:
        file.write(header_text)

    with open('cosine_model.cc', 'w') as file:
        file.write(source_text)
