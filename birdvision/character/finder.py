from dataclasses import dataclass
from typing import List, Iterable
from typing import Optional

import cv2
import numpy as np

from birdvision.character.model import CharacterModel
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


def _split_large_chars(chars: List[Node], rects: List[Rectangle]) -> Iterable[Node]:
    extra = 0
    for i, char in enumerate(chars):
        height, width = char.image.shape
        if width < 32:
            yield char
            continue

        min_x = int(width / 3)
        max_x = int(width / 1.5)
        center = char.image[:int(height*0.60), min_x:max_x]
        energy = center.sum(axis=0)
        min_seam = np.argmin(energy)

        split_point = min_seam + min_x

        if energy[min_seam] < 128:
            yield Node(char.image[:, :split_point], parents=[char], key=['split_left'])
            yield Node(char.image[:, split_point:], parents=[char], key=['split_right'])
            rects.insert(i+extra, rects[i+extra])
            extra += 1
        else:
            yield char


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


@dataclass
class String:
    chars: [str]
    certainty: [float]
    nodes: [Optional[Node]]

    def to_str(self) -> str:
        return ''.join(self.chars)

    def to_int(self) -> Optional[int]:
        s = self.to_str()
        if len(s) == 0:
            return None
        return int(s)


class StringFinder:
    def __init__(self, name: str, rect: Rectangle, prepare_fn, reader_fn, find_spaces: bool = False):
        self.name = name
        self.rect = rect
        self.prepare_fn = prepare_fn
        self.reader_fn = reader_fn
        self.find_spaces = find_spaces

    def __call__(self, frame: Node) -> String:
        prepared_node = self.prepare_fn(frame, self.rect)
        rects = _find_character_rects(prepared_node.image)
        rect_crops = [prepared_node.crop(rect) for rect in rects]
        split_chars = _split_large_chars(rect_crops, rects)
        final_crops = [char.thumbnail32 for char in split_chars]
        chars, certainty = self.reader_fn([crop.image for crop in final_crops])

        if self.find_spaces:
            spaces = _calculate_spaces(rects)
        else:
            spaces = []

        res = String([], [], [])
        for i, char in enumerate(chars):
            res.chars.append(char)
            res.certainty.append(certainty[i])
            res.nodes.append(final_crops[i])
            if i in spaces:
                res.chars.append(' ')
                res.certainty.append(1.0)
                res.nodes.append(None)

        return res


def light_text(frame: Node, rect: Rectangle):
    return frame.gray_min.crop(rect).threshold_binary(125, 255)


def dark_text(frame: Node, rect: Rectangle):
    return frame.gray_max.crop(rect).threshold_binary_inv(110, 255)


def finders_from_model(model: CharacterModel):
    small_digit = model.read_small_digits
    alpha_num = model.read_alpha_num

    return [
        StringFinder('curHP', Rectangle(350, 588, 60, 27), prepare_fn=light_text, reader_fn=small_digit),
        StringFinder('maxHP', Rectangle(423, 601, 60, 27), prepare_fn=light_text, reader_fn=small_digit),
        StringFinder('curMP', Rectangle(350, 623, 60, 27), prepare_fn=light_text, reader_fn=small_digit),
        StringFinder('maxMP', Rectangle(423, 636, 60, 27), prepare_fn=light_text, reader_fn=small_digit),
        StringFinder('curCT', Rectangle(350, 658, 60, 27), prepare_fn=light_text, reader_fn=small_digit),
        StringFinder('maxCT', Rectangle(423, 671, 60, 27), prepare_fn=light_text, reader_fn=small_digit),
        StringFinder('brave', Rectangle(725, 653, 42, 30), prepare_fn=dark_text, reader_fn=small_digit),
        StringFinder('faith', Rectangle(877, 653, 42, 30), prepare_fn=dark_text, reader_fn=small_digit),
        StringFinder('name', Rectangle(610, 545, 320, 40), prepare_fn=dark_text, reader_fn=alpha_num, find_spaces=True),
        StringFinder('job', Rectangle(610, 595, 320, 40), prepare_fn=dark_text, reader_fn=alpha_num, find_spaces=True),
        StringFinder('ability', Rectangle(270, 122, 425, 58), prepare_fn=dark_text, reader_fn=alpha_num,
                     find_spaces=True),
    ]
