import os
from pathlib import Path

import cv2

import birdvision.stream_state as stream_state
from birdvision.node import Node
from birdvision.testing import TestResult


def run():
    # TODO: Replace this with hand picked test cases instead of including everything
    stream_state_model = stream_state.StreamStateModel()

    for path in Path(os.environ['STREAM_STATE_SRC']).iterdir():
        if path.name[0] == '.':
            continue

        expected = path.name
        for image_path in path.glob('*.jpg'):
            frame = Node(cv2.imread(image_path.as_posix()))
            state = stream_state_model(frame)
            actual = state.name
            yield TestResult(image_path.as_posix(), name='stream_state', frame=frame, data=state,
                             ok=actual == expected, actual=actual, expected=expected, relevant_nodes=[state.node])
