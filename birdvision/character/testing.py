import json
from pathlib import Path

import cv2

import birdvision.character as character
from birdvision.node import Node
from birdvision.testing import TestResult


def run():
    test_cases = json.loads(Path('data/tests/character.json').read_text())
    char_model = character.CharacterModel()
    char_finders = character.finders_from_model(char_model)
    by_name = {finder.name: finder for finder in char_finders}

    for fp, case in test_cases.items():
        img = cv2.imread('data/tests/character/' + fp)

        for key, expected in case.items():
            frame = Node(img)
            finder = by_name[key]
            string = finder(frame)
            actual = string.to_str()
            yield TestResult(fp, name=finder.name, frame=frame, data=string, ok=actual == expected, actual=actual,
                             expected=expected, relevant_nodes=string.nodes)
