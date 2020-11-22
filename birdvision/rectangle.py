from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=True)
class Rectangle:
    """Yet another Rectangle class."""

    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def from_coords(x1, y1, x2, y2):
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        a_x = min(x1, x2)
        b_x = max(x1, x2)
        a_y = min(y1, y2)
        b_y = max(y1, y2)
        return Rectangle(a_x, a_y, b_x - a_x, b_y - a_y)

    def crop(self, arr: np.ndarray) -> np.ndarray:
        """Crop a multi-dimensional array, where y comes first."""
        return arr[self.y:self.bottom_y, self.x: self.right_x]

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

    @property
    def top_left(self):
        """Returns the top left point of the rectangle, as a tuple"""
        return self.x, self.y

    @property
    def bottom_right(self):
        """Returns the bottom right point of the rectangle, as a tuple"""
        return self.right_x, self.bottom_y

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersection(self, other: 'Rectangle') -> 'Rectangle':
        x_a = max(self.x, other.x)
        y_a = max(self.y, other.y)
        x_b = min(self.right_x, other.right_x)
        y_b = min(self.bottom_y, other.bottom_y)
        return Rectangle(x_a, y_a, max(0, x_b - x_a), max(0, y_b - y_a))

    def intersection_over_union(self, other: 'Rectangle') -> float:
        inter_area = self.intersection(other).area
        return inter_area / float(self.area + other.area - inter_area)

    def __repr__(self):
        return f'Rectangle({self.x}, {self.y}, {self.width}, {self.height})'
