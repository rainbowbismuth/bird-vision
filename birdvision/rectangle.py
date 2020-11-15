from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=True)
class Rectangle:
    """Yet another Rectangle class."""

    x: int
    y: int
    width: int
    height: int

    def crop(self, arr: np.ndarray) -> np.ndarray:
        """Crop a multi-dimensional array, where y comes first."""
        return arr[self.y:self.y + self.height, self.x + self.x + self.width]

    def move(self, x_offset, y_offset) -> 'Rectangle':
        """Return a new Rectangle that has been moved by an offset."""
        return Rectangle(self.x + x_offset, self.y + y_offset, self.width, self.height)

    @property
    def right_x(self):
        """Return the x position of the two points on the right side of the rectangle."""
        return self.x + self.width

    @property
    def bottom_y(self):
        """Return the y position of the two points on the bottom side of the rectangle."""
        return self.y + self.height
