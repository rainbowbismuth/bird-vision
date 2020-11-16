"""
This module contains `Finder` and `Found`, which allow you to find specific things in a frame.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

from birdvision.node import Node


@dataclass(frozen=True)
class Found:
    """
    A found object in a frame.

    Attributes:
        finder: The finder this object was found with.
        value: The object itself, could be anything depending on the specific Finder.
        certainty: The certainty in which we found this value.
        most_relevant_node: The most relevant node to this found object, if any.
    """
    finder: 'Finder'
    value: object
    certainty: float
    most_relevant_node: Optional[Node]


class Finder(ABC):
    name: str

    @property
    def full_name(self):
        return f'{self.__class__.__name__}::{self.name}'

    @abstractmethod
    def find(self, frame: Node) -> Iterable[Found]:
        """Start finding things in the `frame`."""
        pass
