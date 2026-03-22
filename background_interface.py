"""
Interface definition for parallax background rendering.

Defines the public API that all background implementations must follow.
"""

from abc import ABC, abstractmethod


class IBackgroundRenderer(ABC):
    """Interface for parallax background rendering - defines public API."""

    @abstractmethod
    def update(self, camera_x: float, dt: float) -> None:
        """Update background state based on camera position.

        Args:
            camera_x: Camera x position in world coordinates
            dt: Delta time in seconds
        """
        pass

    @abstractmethod
    def draw(self, surf) -> None:
        """Render the background layers to a surface.

        Args:
            surf: Pygame surface to draw on
        """
        pass

    @abstractmethod
    def set_camera_position(self, x: float, y: float = 0) -> None:
        """Set the camera position for parallax scrolling.

        Args:
            x: Camera x position in world coordinates
            y: Camera y position in world coordinates (optional)
        """
        pass
