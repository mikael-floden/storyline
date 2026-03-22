"""
Interface definition for actor rendering.

Defines the public API that all actor implementations must follow.
"""

from abc import ABC, abstractmethod


class IActorRenderer(ABC):
    """Interface for actor rendering - defines public API."""

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update animation state.

        Args:
            dt: Delta time in seconds
        """
        pass

    @abstractmethod
    def set_animation(self, anim_name: str) -> None:
        """Switch to a different animation with blending.

        Args:
            anim_name: Name of animation to switch to
        """
        pass

    @abstractmethod
    def draw(self, surf, sprite_set: dict) -> None:
        """Render the actor to a surface.

        Args:
            surf: Pygame surface to draw on
            sprite_set: Dictionary of part name -> sprite surface
        """
        pass

    @abstractmethod
    def apply_model(self, model: dict) -> None:
        """Apply a loaded model to this actor.

        Args:
            model: Model data dictionary containing pivots, metrics, animations, etc.
        """
        pass

    @abstractmethod
    def speak(self, text: str, **kwargs) -> bool:
        """Start speaking text and switch to the talk animation while active."""
        pass

    @abstractmethod
    def stop_speaking(self) -> None:
        """Stop active speech playback/synthesis."""
        pass
