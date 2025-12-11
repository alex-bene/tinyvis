"""Visualization utilities for 3D meshes and point clouds using Plotly."""

import numpy as np
import trimesh
from PIL import Image
from plotly import graph_objects as go

from .meshes import trimesh_to_plotly_mesh
from .pointclouds import pointcloud_to_plotly_scatter


def visualize(
    meshes: list[trimesh.Trimesh] | None = None,
    meshes_colors: list[str | None] | None = None,
    meshes_opacities: list[float] | None = None,
    pointclouds: list[np.ndarray] | None = None,
    pointclouds_colors: list[Image.Image | np.ndarray | str | None] | None = None,
    points: np.ndarray | None = None,
    points_colors: list[str | None] | None = None,
    points_names: list[str] | None = None,
    points_sizes: list[float | None] | None = None,
    figure_height: int = 800,
) -> go.Figure:
    """Visualize a list of trimesh.Trimesh objects and an array of points in an interactive Plotly 3D viewer."""
    fig = go.Figure(
        layout=go.Layout(
            scene_camera={"up": {"x": 0, "y": 1, "z": 0}},
            margin={"l": 10, "r": 10, "b": 10, "t": 10},
            height=figure_height,
            scene={
                "xaxis_title": "X Axis",
                "yaxis_title": "Y Axis",
                "zaxis_title": "Z Axis",
                "aspectmode": "data",
                "aspectratio": {"x": 1, "y": 1, "z": 1},
            },
        )
    )

    if pointclouds is not None:
        pointclouds_colors = pointclouds_colors or [None] * len(pointclouds)
        for pointcloud, pointcloud_color in zip(pointclouds, pointclouds_colors, strict=True):
            fig.add_trace(pointcloud_to_plotly_scatter(pointcloud, pointcloud_color))

    if meshes is not None:
        meshes_colors = meshes_colors or [None] * len(meshes)
        meshes_opacities = meshes_opacities or [1.0] * len(meshes)
        for mesh, mesh_color, mesh_opacity in zip(meshes, meshes_colors, meshes_opacities, strict=True):
            human_mesh_trace = trimesh_to_plotly_mesh(mesh, mesh_color, mesh_opacity)
            fig.add_trace(human_mesh_trace)

    if points is not None:
        if points_names is None:
            points_names = [None] * len(points)
        if points_colors is None:
            points_colors = [None] * len(points)
        if points_sizes is None:
            points_sizes = [None] * len(points)
        for point, point_name, point_color, point_size in zip(
            points, points_names, points_colors, points_sizes, strict=True
        ):
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0]],
                    y=[point[1]],
                    z=[point[2]],
                    mode="markers",
                    marker={"size": point_size, "color": point_color},
                    showlegend=True,
                    name=point_name,
                )
            )

    return fig
