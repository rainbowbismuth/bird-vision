from typing import Callable, Iterable, List
from typing import Optional
from uuid import UUID

import cv2
import numpy as np

from birdvision.rectangle import Rectangle


class Node:
    """
    A lazily computed image node. Each accessor generates another node that is memorized, allowing us to review
    each step in our computation.
    """
    test_uuid: Optional[UUID] = None
    test_result: Optional[object] = None

    def __init__(self, image: np.ndarray, parents: Optional[List['Node']] = None, key=None):
        self.image = image
        self.parents = parents
        self.key = key
        self.children = {}

    def ancestors(self) -> Iterable['Node']:
        if not self.parents:
            return
        for parent in self.parents:
            yield from parent.ancestors_and_me()

    def ancestors_and_me(self) -> Iterable['Node']:
        yield self
        yield from self.ancestors()

    def descendents(self) -> Iterable['Node']:
        for child in self.children.values():
            yield from child.descendents_and_me()

    def descendents_and_me(self) -> Iterable['Node']:
        yield self
        yield from self.descendents()

    @property
    def gray(self) -> 'Node':
        return gray(self)

    @property
    def gray_min(self) -> 'Node':
        return gray_min(self)

    @property
    def gray_max(self) -> 'Node':
        return gray_max(self)

    @property
    def thumbnail32(self) -> 'Node':
        return resize(self, 32, 32)

    @property
    def thumbnail64(self) -> 'Node':
        return resize(self, 64, 64)

    def resize(self, width: int, height: int) -> 'Node':
        return resize(self, width, height)

    def crop(self, rect: Rectangle) -> 'Node':
        return crop(self, rect)

    def threshold_binary(self, threshold: int, max_val: int) -> 'Node':
        return threshold_binary(self, threshold, max_val)

    def threshold_binary_inv(self, threshold: int, max_val: int) -> 'Node':
        return threshold_binary_inv(self, threshold, max_val)


NODE_NAMES = {}


def memoized_node(func: Callable[..., np.ndarray]):
    """
    A decorator to create a function that is memoized in a node graph. The function should take a node as the
    first argument, and return an image ndarray. The decorated function will actually return the new node, however.

    Each memoized node function should have a unique name, and if a convenience method is added to `Node` it should
    have the same name as well.

    All arguments to the function should support being a dictionary key, and kwargs are not supported.
    """
    assert func.__name__ not in NODE_NAMES
    NODE_NAMES[func.__name__] = func

    def wrapper(*args):
        node = args[0]
        key = (func.__name__, *args[1:])
        val = node.children.get(key)
        if val is not None:
            return val
        image = func(*args)
        assert image is not None
        child = Node(image, parents=[node], key=key)
        node.children[key] = child
        return child

    return wrapper


@memoized_node
def gray(node: Node):
    return cv2.cvtColor(node.image, cv2.COLOR_BGR2GRAY)


@memoized_node
def gray_min(node: Node):
    return np.min(node.image, axis=2)


@memoized_node
def gray_max(node: Node):
    return np.max(node.image, axis=2)


@memoized_node
def resize(node: Node, width: int, height: int):
    return cv2.resize(node.image, (width, height))


@memoized_node
def crop(node: Node, rect: Rectangle):
    return rect.crop(node.image)


@memoized_node
def threshold_binary(node: Node, threshold: int, max_val: int):
    return cv2.threshold(node.image, threshold, max_val, cv2.THRESH_BINARY)[1]


@memoized_node
def threshold_binary_inv(node: Node, threshold: int, max_val: int):
    return cv2.threshold(node.image, threshold, max_val, cv2.THRESH_BINARY_INV)[1]
