"""Module to translate pointcloud representations to plotly Scatter3d."""

from typing import TYPE_CHECKING, Union

import numpy as np
from plotly import graph_objects as go

from .common import ColorInput, to_numpy_colors

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]
    from pytorch3d.cameras import CamerasBase  # pyright: ignore[reportMissingImports]

CameraInput = Union["torch.Tensor", np.ndarray, "CamerasBase"]

__all__ = ["camera_to_plotly_scatter"]


def get_camera_wireframe(scale: float = 0.3) -> np.ndarray:
    """Create a wireframe of a 3D line-plot of a camera symbol.

    Source: https://github.com/facebookresearch/pytorch3d/blob/f5f6b78e70e0a1b70f3be9a09b5b001e9b3a7a03/pytorch3d/vis/plotly_vis.py#L68

    """
    a = 0.5 * np.array([-2, 1.5, 4])
    up1 = 0.5 * np.array([0, 1.5, 4])
    up2 = 0.5 * np.array([0, 2, 4])
    b = 0.5 * np.array([2, 1.5, 4])
    c = 0.5 * np.array([-2, -1.5, 4])
    d = 0.5 * np.array([2, -1.5, 4])
    C = np.zeros(3)
    F = np.array([0, 0, 3])
    return np.stack([a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]).astype(float) * scale


def camera_to_plotly_scatter(
    camera: CameraInput,
    color: ColorInput = None,
    name: str | None = None,
    showlegend: bool = True,
    camera_scale: float = 0.2,
    marker_size: float = 1.0,
) -> go.Scatter3d:
    """Convert a 3D point cloud to a Plotly scatter plot."""
    if hasattr(camera, "R") and hasattr(camera, "T"):
        R, T = camera.R, camera.T
        if hasattr(R, "detach"):
            R = R.detach().cpu().numpy().squeeze()
        if hasattr(T, "detach"):
            T = T.detach().cpu().numpy().squeeze()
    if hasattr(camera, "detach"):
        camera = camera.detach().cpu().numpy().squeeze()
    if isinstance(camera, np.ndarray):
        camera = camera.squeeze()
        if camera.shape in ((3, 4), (4, 4)):
            Rc2w = camera[:3, :3].T
            Tc2w = -Rc2w @ camera[:3, 3]
        else:
            msg = "Input camera extrinsics must be of shape (3,4) or (4,4)."
            raise ValueError(msg)
    else:
        msg = "Input camera must be a pytorch3d CamerasBase, torch.Tensor, or numpy.ndarray."
        raise TypeError(msg)

    cam_points = get_camera_wireframe(camera_scale)
    cam_points = np.einsum("ij,kj->ki", Rc2w, cam_points) + Tc2w[None, :]

    # Flatten the points array to get x, y, z coordinates
    flat_points = cam_points.reshape(-1, 3)
    x = flat_points[:, 0]
    y = flat_points[:, 1]
    z = flat_points[:, 2]

    return go.Scatter3d(
        x=x, y=y, z=z, marker={"size": marker_size, "color": to_numpy_colors(color)}, name=name, showlegend=showlegend
    )
