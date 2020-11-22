"""
A model for detecting objects in a frame, and it's associated support code.

The strategies I've tried so far aren't really working well at all. I want something that runs fine on just CPU, and
the architectures I've tried so far aren't there.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from birdvision.constants import STREAM_WIDTH, STREAM_HEIGHT
from birdvision.node import Node
from birdvision.rectangle import Rectangle

TILE_WIDTH = 45
TILE_HEIGHT = 37
SCALE = 2


def load_classes() -> List[str]:
    root = Path(os.environ['OBJECTS_SRC'])
    classes = ['None']
    for path in root.iterdir():
        if path.is_dir():
            classes.append(path.name)
    return sorted(classes)


def node_to_tiles(frame: Node) -> Iterable[Node]:
    load_classes()
    width, height = frame.width, frame.height
    tiles_wide = width // TILE_WIDTH
    tiles_high = height // TILE_HEIGHT
    for i in range(tiles_wide):
        for j in range(tiles_high):
            x_offset = i * TILE_WIDTH
            y_offset = j * TILE_HEIGHT
            rect = Rectangle(i * TILE_WIDTH, j * TILE_HEIGHT, TILE_WIDTH, TILE_HEIGHT)
            yield x_offset, y_offset, frame.crop(rect)


@dataclass(frozen=True)
class ObjectPrediction:
    kind: str
    confidence: float
    rect: Rectangle
    node: Node


class ObjectModel:
    """
    ObjectModel splits the screen up into a bunch of equally sized tiles and tries to classify what kind of object
    is in there.
    """

    def __init__(self):
        import tensorflow as tf
        self.classes = load_classes()
        self.model = tf.keras.models.load_model(os.environ['OBJECT_MODEL'])

    def __call__(self, frame: Node) -> List[ObjectPrediction]:
        import tensorflow as tf
        tiles = list(node_to_tiles(frame.resize(STREAM_WIDTH // SCALE, STREAM_HEIGHT // SCALE)))
        y_pred = self.model(tf.stack([process_image(tile.image) for (_, _, tile) in tiles]))

        pred_class = [self.classes[i] for i in np.argmax(y_pred, axis=1)]
        confidence = np.max(y_pred, axis=1)

        out = []
        for i in range(len(tiles)):
            x_offset, y_offset, tile = tiles[i]
            rect = Rectangle(x_offset * SCALE, y_offset * SCALE, TILE_WIDTH, TILE_HEIGHT)
            out.append(ObjectPrediction(pred_class[i], confidence[i], rect, tile))
        return out


def load_relevant_sprites() -> Iterable[Tuple[str, Node]]:
    root = Path(os.environ['OBJECTS_SRC'])
    for path in root.glob('**/*.png'):
        img = cv2.imread(path.as_posix())
        node = Node(img)
        if node.width < 24 or node.height < 24:
            continue
        # Filter out portraits (TODO: I should delete them.)
        if node.width == 48 and node.height == 32:
            continue
        yield path.parent.parent.name, node


def load_relevant_backgrounds() -> Iterable[Node]:
    root = Path(os.environ['GENERATIVE_BGS_SRC'])
    for path in root.glob('*.png'):
        img = cv2.imread(path.as_posix())
        node = Node(img)
        yield node.resize(node.width // 2, node.height // 2)


def blit(dest: np.ndarray, src: np.ndarray, loc: Tuple[int, int]):
    pos = [i if i >= 0 else None for i in loc]
    neg = [-i if i < 0 else None for i in loc]
    target = dest[tuple((slice(i, None) for i in pos))]
    src = src[tuple((slice(i, j) for i, j in zip(neg, target.shape)))]
    target[tuple((slice(None, i) for i in src.shape))] = src


def alpha_blit(dest: np.ndarray, src: np.ndarray, loc: Tuple[int, int]):
    dest_like = np.zeros(dest.shape, dtype=np.uint8)
    blit(dest_like, src, loc)
    not_obj = dest_like == 0

    np.multiply(dest, not_obj, out=dest)
    np.add(dest, dest_like, out=dest)


def select_random_bg(backgrounds):
    bg = random.choice(backgrounds)

    if random.random() < 0.5:
        bg = bg.flip_horizontally

    x = random.randint(0, bg.width - TILE_WIDTH)
    y = random.randint(0, bg.height - TILE_HEIGHT)
    return Rectangle(x, y, TILE_WIDTH, TILE_HEIGHT).crop(bg.image)


def add_random_sprite(dest, sprites):
    (kind, sprite) = random.choice(sprites)

    if random.random() < 0.5:
        sprite = sprite.flip_horizontally

    width, height = sprite.width, sprite.height
    width_div2 = width // 2
    height_div2 = height // 2
    offset_x = random.randint(-width_div2, width_div2)
    offset_y = random.randint(-height_div2, height_div2)
    alpha_blit(dest, sprite.image, (offset_y, offset_x))
    return kind


def process_image(image):
    import tensorflow as tf
    import tensorflow.keras.applications.mobilenet as mn
    image = tf.image.resize(image, [128, 128])
    image = mn.preprocess_input(image, data_format='channels_last')
    return image


def generate_batches(batch_size=64, max_batches=20_000_000):
    import tensorflow as tf
    classes = load_classes()
    none_idx = classes.index('None')
    sprites = list(load_relevant_sprites())
    backgrounds = list(load_relevant_backgrounds())

    for i in range(max_batches):
        just_bg = select_random_bg(backgrounds)

        xs = [process_image(just_bg), process_image(np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8))]
        ys = [none_idx, none_idx]
        for j in range(batch_size-2):
            generated_tile = select_random_bg(backgrounds)
            kind = add_random_sprite(generated_tile, sprites)
            # cv2.imwrite(f'/Volumes/RAM_Disk/batch/{i}_{j}.png', generated_tile)

            xs.append(process_image(generated_tile))
            ys.append(classes.index(kind))

        yield tf.stack(xs), np.array(ys)


def train_object_model():
    import tensorflow as tf
    import tensorflow.keras.applications.mobilenet as mn

    base_model = mn.MobileNet(include_top=False, input_shape=(128, 128, 3))
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(len(load_classes()), activation='softmax')(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False

    dst = os.environ['OBJECT_MODEL']

    optimizer = tf.keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, monitor='loss', restore_best_weights=True)

    model.fit(generate_batches(), epochs=10, steps_per_epoch=200, callbacks=[early_stopping_cb])

    for layer in base_model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(generate_batches(), epochs=20, steps_per_epoch=200, callbacks=[early_stopping_cb])

    model.save(dst)
