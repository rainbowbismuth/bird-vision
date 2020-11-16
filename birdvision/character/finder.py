from typing import List, Iterable

import cv2
import numpy as np

from birdvision.character.model import CharacterModel
from birdvision.finder import Finder, Found
from birdvision.node import Node
from birdvision.rectangle import Rectangle

PREPARED_CHAR_DIMENSIONS = (32, 32)


def _find_character_rects(img):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]

    # Filter out rects that are too small or large to be letters
    filtered_rects = []
    for (x, y, w, h) in rects:
        if w < 5 or h < 9:
            continue
        if w > 40 or h > 40:
            continue
        filtered_rects.append((x, y, w, h))

    # Stop early if no letters/numbers
    if not filtered_rects:
        return []

    median_x = np.median([rect[0] for rect in filtered_rects])
    rects_with_distance = [(abs(median_x - x), x, y, w, h) for (x, y, w, h) in filtered_rects]

    # Sort rects by distance from median, so closest is first
    rects_with_distance.sort(key=lambda rect: rect[0])

    filtered_rects = []
    prev_dist = rects_with_distance[0][0]
    for (dist, x, y, w, h) in rects_with_distance:
        if dist - prev_dist > 25:
            break
        prev_dist = dist
        filtered_rects.append((x, y, w, h))

    # Sort from left-right
    filtered_rects.sort(key=lambda rect: rect[0])

    out = []

    # Find the minimum y so all letters can start there
    min_y = min([rect[1] for rect in filtered_rects])

    for (x, y, w, h) in filtered_rects:
        # Adjust our bounds so that all rects start at min_y
        diff_y = min_y - y
        y = int(min_y)
        h -= int(diff_y)

        # Reject rects too tall after the extension
        if h > 40:
            continue
        out.append(Rectangle(max(x - 1, 0), max(y - 1, 0), w + 2, h + 2))

    return out


def _calculate_spaces(rects: List[Rectangle]):
    """Returns a list of positions in which there is a space after."""
    out = []
    if not rects:
        return out
    last_end = rects[0].right_x
    for i, rect in enumerate(rects):
        if rect.x - last_end > 7:
            out.append(i - 1)
        last_end = rect.right_x
    return out


class CharacterFinder(Finder):
    def __init__(self, name: str, rect: Rectangle, prepare_fn, reader_fn):
        self.name = name
        self.rect = rect
        self.prepare_fn = prepare_fn
        self.reader_fn = reader_fn

    def find(self, frame: Node) -> Iterable[Found]:
        prepared_node = self.prepare_fn(frame, self.rect)
        rects = _find_character_rects(prepared_node.image)
        crops = [prepared_node.crop(rect).thumbnail32 for rect in rects]

        chars, certainty = self.reader_fn([crop.image for crop in crops])
        spaces = _calculate_spaces(rects)

        for i, char in enumerate(chars):
            yield Found(self, char, certainty[i], crops[i])
            if i in spaces:
                yield Found(self, ' ', 1.0, None)


def _light_text(frame: Node, rect: Rectangle):
    return frame.gray_min.crop(rect).threshold_binary(125, 255)


def _dark_text(frame: Node, rect: Rectangle):
    return frame.gray_max.crop(rect).threshold_binary_inv(110, 255)


def finders_from_model(model: CharacterModel):
    small_digit = model.read_small_digits
    alpha_num = model.read_alpha_num

    return [
        CharacterFinder('minHP', Rectangle(350, 588, 60, 27), prepare_fn=_light_text, reader_fn=small_digit),
        CharacterFinder('maxHP', Rectangle(423, 601, 60, 27), prepare_fn=_light_text, reader_fn=small_digit),
        CharacterFinder('minMP', Rectangle(350, 623, 60, 27), prepare_fn=_light_text, reader_fn=small_digit),
        CharacterFinder('maxMP', Rectangle(423, 636, 60, 27), prepare_fn=_light_text, reader_fn=small_digit),
        CharacterFinder('minCT', Rectangle(350, 658, 60, 27), prepare_fn=_light_text, reader_fn=small_digit),
        CharacterFinder('maxCT', Rectangle(423, 671, 60, 27), prepare_fn=_light_text, reader_fn=small_digit),
        CharacterFinder('brave', Rectangle(725, 653, 42, 30), prepare_fn=_dark_text, reader_fn=small_digit),
        CharacterFinder('faith', Rectangle(877, 653, 42, 30), prepare_fn=_dark_text, reader_fn=small_digit),
        CharacterFinder('name', Rectangle(610, 545, 320, 40), prepare_fn=_dark_text, reader_fn=alpha_num),
        CharacterFinder('job', Rectangle(610, 595, 320, 40), prepare_fn=_dark_text, reader_fn=alpha_num),
        CharacterFinder('ability', Rectangle(270, 122, 425, 58), prepare_fn=_dark_text, reader_fn=alpha_num),
    ]


def found_to_string(found_chars):
    """Join a list of found characters together into a string"""
    return ''.join([found.value for found in found_chars])
