"""
This module contains a simple testing framework, so that we can test finders and view their results.

It's designed so that all test results are stored in a global variable, and you will need to re-run the program
to rerun the tests.
"""
from termcolor import colored
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List
import sys
import json

from birdvision.finder import Found
import itertools

RESULTS = []

OK_DOT = colored('.', 'green')
FAIL_DOT = colored('.', 'red')
WRAP_AT = 100


@dataclass
class TestResult:
    file: str
    ok: bool
    actual: object
    expected: object
    readings: List[Found]
    notes: dict
    idx: int = 0


def add_test_result(result: TestResult):
    result.idx = len(RESULTS)
    RESULTS.append(result)
    sys.stdout.write(OK_DOT if result.ok else FAIL_DOT)
    if len(RESULTS) % WRAP_AT == (WRAP_AT - 1):
        sys.stdout.write('\n')


def run_all_tests():
    import birdvision.character.testing

    test_sets = [
        birdvision.character.testing.run()
    ]

    for result in itertools.chain(*test_sets):
        add_test_result(result)
