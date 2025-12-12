"""Module to translate pointcloud representations to plotly Scatter3d."""

from typing import TYPE_CHECKING, Union

import numpy as np
from PIL import Image
from plotly import graph_objects as go

try:
    import torch  # pyright: ignore[reportMissingImports]
except ImportError:
    torch = None

if TYPE_CHECKING:
    from pytorch3d.structures import Pointclouds  # pyright: ignore[reportMissingImports]
    from trimesh.points import PointCloud  # pyright: ignore[reportMissingImports]

PointcloudInput = Union["torch.Tensor", np.ndarray, "Pointclouds", "PointCloud"]
PointcloudColorInput = Union[Image.Image, np.ndarray, "torch.Tensor", str, None]

__all__ = ["pointcloud_to_plotly_scatter"]


def to_numpy_pointcloud(points: PointcloudInput) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert various pointcloud representations to numpy array.

    Args:
        points (torch.Tensor | np.ndarray | pytorch3d.structures.Pointclouds | trimesh.points.PointCloud): Pointcloud

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Numpy array of points and optional colors

    """
    colors = None
    if hasattr(points, "points_packed"):
        points = points.points_packed().cpu().numpy()
    elif hasattr(points, "vertices"):
        points = points.vertices
        colors = points.colors if hasattr(points, "colors") else None
    elif torch is not None and isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    return points, colors


def to_numpy_colors(colors: PointcloudColorInput) -> np.ndarray | str | None:
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
    if torch is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    return colors


def pointcloud_to_plotly_scatter(
    points: PointcloudInput,
    colors: PointcloudColorInput = None,
    marker_size: float = 1.0,
    name: str | None = None,
    showlegend: bool = True,
) -> go.Scatter3d:
    """Convert a 3D point cloud to a Plotly scatter plot."""
    points, colors_from_pointcloud = to_numpy_pointcloud(points)
    # Flatten the points array to get x, y, z coordinates
    flat_points = points.reshape(-1, 3)
    good_idxs = np.isfinite(flat_points)
    good_idxs = np.any(good_idxs, axis=1)
    x = flat_points[:, 0][good_idxs]
    y = flat_points[:, 1][good_idxs]
    z = flat_points[:, 2][good_idxs]

    colors = to_numpy_colors(colors) if colors is not None else colors_from_pointcloud
    if isinstance(colors, np.ndarray):
        # Flatten image for coloring
        colors = colors.reshape(-1, 3)[good_idxs]
        colors = [f"rgb({','.join([str(int(cc)) for cc in c.tolist()])})" for c in colors]

    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker={"size": marker_size, "color": colors}, name=name, showlegend=showlegend
    )
