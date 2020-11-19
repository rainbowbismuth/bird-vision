"""
TODO: Module comment
"""

import os
from dataclasses import dataclass

import numpy as np

from birdvision.node import Node
from birdvision.rectangle import Rectangle

BLACK = 'Black'
COMMERCIAL = 'Commercial'
STREAM = 'Stream'
STREAM_FIGHT = 'Stream_Fight'
STREAM_BETTING_OPEN = 'Stream_Betting_Open'
PREGAME = 'Pregame'
PREGAME_UNIT_CARD = 'Pregame_UnitCard'
GAME = 'Game'
GAME_LARGE_EFFECT = 'Game_LargeEffect'
GAME_SELECT_REACTION = 'Game_Select_Reaction'
GAME_SELECT_HALF_LEFT = 'Game_Select_Half_Left'
GAME_SELECT_HALF_RIGHT = 'Game_Select_Half_Right'
GAME_SELECT_FULL = 'Game_Select_Full'
GAME_ABILITY_TAG = 'Game_AbilityTag'
STREAM_WINNER = 'Stream_Winner'
STREAM_RESULT = 'Stream_Result'

STREAM_STATES = [
    BLACK,
    COMMERCIAL,
    STREAM,
    STREAM_FIGHT,
    STREAM_BETTING_OPEN,
    PREGAME,
    PREGAME_UNIT_CARD,
    GAME,
    GAME_LARGE_EFFECT,
    GAME_SELECT_REACTION,
    GAME_SELECT_HALF_LEFT,
    GAME_SELECT_HALF_RIGHT,
    GAME_SELECT_FULL,
    GAME_ABILITY_TAG,
    STREAM_WINNER,
    STREAM_RESULT,
]


def in_game(state: str) -> bool:
    return state.startswith('Game')


def unit_select(state: str) -> bool:
    return state.startswith('Game_Select')


@dataclass
class StreamState:
    name: str
    certainty: float
    node: Node


class StreamStateModel:
    """
    CharacterModel encapsulates two tensorflow models, one for FFT's small digit font (for HP/MP/CT etc) and it's
    general purpose font used for all other text.

    The character arrays mentioned are supposed to be 32x32 uint8 arrays that have already been preprocessed.
    """

    def __init__(self):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(os.environ['STREAM_STATE_MODEL'])

    def __call__(self, frame: Node) -> StreamState:
        prepared = prepare_frame(frame)
        y_pred = self.model(np.array([prepared / 255.0]))
        idx: int = np.argmax(y_pred, axis=1)[0]
        certainty: float = np.max(y_pred, axis=1)
        state = STREAM_STATES[idx]
        return StreamState(state, certainty, Node(prepared))


def prepare_frame(frame: Node) -> np.ndarray:
    gray = frame.gray

    everything = gray.thumbnail32.image
    bottom_left = gray.crop(Rectangle(40, 522, 470, 175)).thumbnail32.image
    bottom_right = gray.crop(Rectangle(520, 522, 470, 175)).thumbnail32.image
    effect_area = gray.crop(Rectangle(260, 94, 450, 95)).thumbnail32.image

    # TODO: Huh... I guess with my current architecture this isn't easy to represent.
    return np.block([[everything, effect_area], [bottom_left, bottom_right]])


def load_labelled_states():
    from pathlib import Path
    import cv2
    xs = []
    ys = []

    for path in Path(os.environ['STREAM_STATE_SRC']).iterdir():
        if path.name[0] == '.':
            continue

        state = path.name
        index = STREAM_STATES.index(state)
        for image_path in path.glob('*.jpg'):
            frame = Node(cv2.imread(image_path.as_posix()))
            img = prepare_frame(frame)
            xs.append(img)
            ys.append(index)

    return np.array(xs), np.array(ys)


def train_stream_state():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    dst = os.environ['STREAM_STATE_MODEL']

    # LeNet-5
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((64, 64, 1)),
        tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(STREAM_STATES),
                              activation='softmax',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    xs, ys = load_labelled_states()
    xs = xs / 255.0
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, monitor='val_loss', restore_best_weights=True)

    print(X_train.shape)
    model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping_cb])
    model.evaluate(X_test, y_test, verbose=2)
    model.save(dst)
