import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import re
import pandas as pd

# Определение класса CTCLayer
class CTCLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

    def get_config(self):
        config = super(CTCLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Словарь символов
char_list = [' ', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
             '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'Y', 'o', 'x', 'А', 'Б',
             'В', 'Г', 'Е', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т',
             'У', 'Ц', 'Ч', 'Э', 'г', 'е', 'и', 'й', 'к', 'п', 'р', 'у', '№']
index_to_char = {i: char for i, char in enumerate(char_list)}
blank_index = len(char_list)

# Декодер CTC
def ctc_decoder(preds, blank_index):
    # Получаем индексы с максимальной вероятностью для каждого временного шага
    pred_indices = np.argmax(preds, axis=-1)

    # Убираем повторяющиеся символы и символы blank
    decoded_texts = []
    for pred in pred_indices:
        decoded_text = []
        previous_char = None
        for char_index in pred:
            if char_index != blank_index and char_index != previous_char:
                decoded_text.append(char_list[char_index])
            previous_char = char_index
        decoded_texts.append(''.join(decoded_text))

    return decoded_texts

# Предобработка изображения
def preprocess_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)

    return sharpened

# Создание модели для инференса
def create_inference_model(model):
    image_input = model.input[0]
    dense_output = model.get_layer(name="dense2").output
    inference_model = tf.keras.models.Model(inputs=image_input, outputs=dense_output)
    return inference_model

# Функция для распознавания текста с использованием CRNN
def recognize_text_with_crnn(image, model):
    image = preprocess_image(image)
    image = cv2.resize(image, (128, 64))
    image = image.astype('float32') / 255.0

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    inference_model = create_inference_model(model)
    preds = inference_model.predict(image)

    decoded_text = ctc_decoder(preds, blank_index)
    return decoded_text[0]

# Функция для загрузки модели CRNN
def load_crnn_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})