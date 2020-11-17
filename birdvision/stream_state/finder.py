from typing import Iterable

from birdvision.finder import Finder, Found
from birdvision.node import Node
from birdvision.stream_state.model import StreamStateModel


class StreamStateFinder(Finder):
    def __init__(self, model: StreamStateModel):
        self.name = "state"
        self.model = model

    def find(self, frame: Node) -> Iterable[Found]:
        # TODO: Yeah see? The transformation inside the model class is hidden...
        state, certainty, node = self.model.stream_state(frame)
        yield Found(self, state, certainty, node)
