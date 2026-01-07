"""Module to translate pointcloud representations to plotly Scatter3d."""

from typing import TYPE_CHECKING, Union

import numpy as np
from plotly import graph_objects as go

from .common import ColorInput, to_numpy_colors

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]
    from pytorch3d.structures import Pointclouds  # pyright: ignore[reportMissingImports]
    from trimesh.points import PointCloud  # pyright: ignore[reportMissingImports]

PointcloudInput = Union["torch.Tensor", np.ndarray, "Pointclouds", "PointCloud"]

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
    elif hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return points, colors


def pointcloud_to_plotly_scatter(
    points: PointcloudInput,
    colors: ColorInput = None,
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
