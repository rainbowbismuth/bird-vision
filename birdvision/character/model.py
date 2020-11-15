"""
This modules contains a class, `CharacterModel`, to read the two main fonts used in FFT, as well as functions
to train the models in the first place.
"""

import os

import numpy as np

SMALL_DIGIT_CHARSET = "0123456789"
ALPHA_NUM_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+"


def _read_model(model, charset, characters):
    if characters is None:
        return []
    y_pred = model(np.array([char / 255.0 for char in characters]))
    chars = [charset[i] for i in np.argmax(y_pred, axis=1)]
    certainty = np.max(y_pred, axis=1)
    return chars, certainty


class CharacterModel:
    """
    CharacterModel encapsulates two tensorflow models, one for FFT's small digit font (for HP/MP/CT etc) and it's
    general purpose font used for all other text.

    The character arrays mentioned are supposed to be 32x32 uint8 arrays that have already been preprocessed.
    """

    def __init__(self):
        import tensorflow as tf
        self.small_digit_model = tf.keras.models.load_model(os.environ['SMALL_DIGIT_MODEL'])
        self.alphanum_model = tf.keras.models.load_model(os.environ['ALPHA_NUM_MODEL'])

    def read_small_digits(self, characters):
        return _read_model(self.small_digit_model, SMALL_DIGIT_CHARSET, characters)

    def read_alpha_num(self, characters):
        return _read_model(self.alphanum_model, ALPHA_NUM_CHARSET, characters)


def _load_labelled_characters(src, charset):
    from pathlib import Path
    import cv2
    xs = []
    ys = []

    for path in Path(src).iterdir():
        if path.name[0] == '.':
            continue

        # We look at just the last character. The reason for this is macOS is case-insensitive, so we need to be able
        # to have both an 'a' and a 'cA' directory, for lower-case and capital A respectively.
        char = path.name[-1]
        index = charset.index(char)
        for image_path in path.glob('*.png'):
            image = cv2.imread(image_path.as_posix())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            xs.append(gray)
            ys.append(index)

    return np.array(xs), np.array(ys)


def _train(src, charset, dst):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    # LeNet-5
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((32, 32, 1)),
        tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(charset),
                              activation='softmax',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    xs, ys = _load_labelled_characters(src, charset)
    xs = xs / 255.0
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, monitor='val_loss', restore_best_weights=True)

    print(X_train.shape)
    model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping_cb])
    model.evaluate(X_test, y_test, verbose=2)
    model.save(dst)


def train_alpha_num():
    _train(os.environ['ALPHA_NUM_SRC'], ALPHA_NUM_CHARSET, os.environ['ALPHA_NUM_MODEL'])


def train_small_digit():
    _train(os.environ['SMALL_DIGIT_SRC'], SMALL_DIGIT_CHARSET, os.environ['SMALL_DIGIT_MODEL'])
