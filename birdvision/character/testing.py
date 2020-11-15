import cv2
from pathlib import Path
import json
from birdvision.frame import Frame
import birdvision.character as character
from birdvision.testing import TestResult


def run():
    test_cases = json.loads(Path('data/tests/character.json').read_text())
    char_model = character.CharacterModel()
    char_finders = character.finders_from_model(char_model)
    by_name = {finder.name : finder for finder in char_finders}

    for fp, case in test_cases.items():
        img = cv2.imread('data/tests/character/' + fp)
        frame = Frame(img)

        for key, expected in case.items():
            finder = by_name[key]
            notes = {}
            readings = list(finder.find(frame, notes))
            actual = character.found_to_string(readings)
            yield TestResult(fp, actual == expected, actual, expected, readings, notes)
