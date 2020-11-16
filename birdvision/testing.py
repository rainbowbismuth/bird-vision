"""
This module contains a simple testing framework, so that we can test finders and view their results.
"""
import itertools
import sys
from dataclasses import dataclass
from importlib.abc import Finder
from typing import List, Iterable
from uuid import UUID, uuid4

from termcolor import colored

from birdvision.finder import Found
from birdvision.node import Node

OK_DOT = colored('.', 'green')
FAIL_DOT = colored('.', 'red')
WRAP_AT = 100


@dataclass
class TestResult:
    file: str
    finder: Finder
    frame: Node
    ok: bool
    actual: object
    expected: object
    readings: List[Found]
    idx: int = 0


class TestFramework:
    def __init__(self):
        self.results = []
        self.id_to_node = {}
        self.node_to_id = {}
        self.node_to_result = {}

    def record(self, result: TestResult):
        result.idx = len(self.results)
        self.results.append(result)

        for node in result.frame.descendents_and_me():
            new_id = uuid4()
            node.test_uuid = new_id
            node.test_result = result
            self.id_to_node[new_id] = node

        sys.stdout.write(OK_DOT if result.ok else FAIL_DOT)
        if len(self.results) % WRAP_AT == (WRAP_AT - 1):
            sys.stdout.write('\n')

    def done(self):
        if len(self.results) % WRAP_AT != 0:
            print()

    def failures(self) -> Iterable[TestResult]:
        for result in self.results:
            if not result.ok:
                yield result

    def summarize_to_stdout(self):
        failures = len(list(self.failures()))
        print(f'\n{failures} failures / {len(self.results)} total')

    def get_node(self, node_id: UUID) -> Node:
        return self.id_to_node[node_id]


def run_all_tests() -> TestFramework:
    import birdvision.character.testing

    framework = TestFramework()
    test_sets = [
        birdvision.character.testing.run()
    ]

    for result in itertools.chain(*test_sets):
        framework.record(result)

    framework.done()
    return framework
