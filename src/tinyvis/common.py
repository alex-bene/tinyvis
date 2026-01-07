"""Common utilities for TinyVis."""

from typing import TYPE_CHECKING, Union

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


ColorInput = Union[np.ndarray, "torch.Tensor", str, None]


def to_numpy_colors(colors: ColorInput) -> np.ndarray | str | None:
    """Convert various color representations to numpy array or string.

    Args:
        colors (torch.Tensor | np.ndarray | PIL.Image | str | None): Input colors

    Returns:
        np.ndarray | str | None: Numpy array of colors or string

    """
    if colors is None or isinstance(colors, str):
        return colors
    if isinstance(colors, Image.Image):
        return np.array(colors)
    if hasattr(colors, "detach"):
        colors = colors.detach().cpu().numpy()
    return colors
