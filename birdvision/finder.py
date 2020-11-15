"""
This module contains `Finder` and `Found`, which allow you to find specific things in a frame.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterable

from birdvision.frame import Frame


@dataclass(frozen=True)
class Found:
    """
    A found object in a frame.

    Attributes:
        finder: The finder this object was found with.
        value: The object itself, could be anything depending on the specific Finder.
        certainty: The certainty in which we found this value.
        notes: An optional set of notes to go along with what we found.
    """
    finder: 'Finder'
    value: object
    certainty: float
    notes: Optional[dict]


class Finder(ABC):
    name: str

    @abstractmethod
    def find(self, frame: Frame, notes: dict = None) -> Iterable[Found]:
        """
        Start finding things in the `frame`.

        An optional notes dictionary can be given that the finder can leave notes in. The intent is that our testing
        framework should be able to 'peek inside' this functions processing, but we don't need to do it live.
        """
        pass
