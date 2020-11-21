"""
This module contains a simple testing framework, so that we can test finders and view their results.
"""
import itertools
import sys
from dataclasses import dataclass
from typing import List, Iterable
from uuid import UUID, uuid4

from termcolor import colored

from birdvision.node import Node

OK_DOT = colored('.', 'green')
FAIL_DOT = colored('.', 'red')
WRAP_AT = 100


@dataclass
class TestResult:
    file: str
    name: str
    frame: Node
    ok: bool
    data: object
    actual: object
    expected: object
    relevant_nodes: List[Node]
    idx: int = 0


class TestFramework:
    def __init__(self):
        self.results = []
        self.id_to_node = {}
        self.node_to_id = {}
        self.node_to_result = {}

    def add_nodes(self, node: Node, result: TestResult):
        if node is None:
            return
        for descendent in node.descendents_and_me():
            if descendent.test_uuid is not None:
                continue
            new_id = uuid4()
            descendent.test_uuid = new_id
            descendent.test_result = result
            self.id_to_node[new_id] = descendent
        for ancestor in node.ancestors_and_me():
            if ancestor.test_uuid is not None:
                continue
            new_id = uuid4()
            ancestor.test_uuid = new_id
            ancestor.test_result = result
            self.id_to_node[new_id] = ancestor

    def record(self, result: TestResult):
        result.idx = len(self.results)
        self.results.append(result)

        self.add_nodes(result.frame, result)
        for node in result.relevant_nodes:
            self.add_nodes(node, result)

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
    import birdvision.stream_state.testing

    framework = TestFramework()
    test_sets = [
        birdvision.character.testing.run(),
        birdvision.stream_state.testing.run()
    ]

    for result in itertools.chain(*test_sets):
        framework.record(result)

    framework.done()
    return framework
