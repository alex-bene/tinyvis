"""Visualization utilities for 3D meshes and point clouds using Plotly."""

import numpy as np
import trimesh
from PIL import Image
from plotly import graph_objects as go


def trimesh_to_plotly_mesh(
    mesh: trimesh.Trimesh, color: str | None = None, opacity: float = 1.0, flatshading: bool = False
) -> go.Mesh3d:
    """Visualize a trimesh.Trimesh object in an interactive Plotly 3D viewer.

    This function now supports converting a UV-based texture to vertex-based
    colors to display the texture in the viewer, as Plotly's Mesh3d trace
    does not support texture mapping directly.

    Args:
        mesh (trimesh.Trimesh): The trimesh object to visualize.
        color (str | None): Optional color to apply to the mesh and override any texture.
        opacity (float): Opacity of the mesh.
        flatshading (bool): Whether to use flat shading for the mesh. Is overridden if opacity < 1. Defaults to False.

    Returns:
        go.Mesh3d: A Plotly Mesh3d trace representing the trimesh object.

    """
    # Check if the input is a trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        msg = "Input must be a trimesh.Trimesh object."
        raise TypeError(msg)

    # Trimesh -> Plotly 3D coordinates
    ## [x, y, z] -> [-x, z, y]
    plotly_mesh = mesh.copy()

    # Get vertices and faces from the trimesh object
    vertices = plotly_mesh.vertices
    faces = plotly_mesh.faces

    # Initialize trace parameters
    trace_kwargs = {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
        "flatshading": flatshading or opacity < 1.0,
        "lighting": {"ambient": 0.5, "diffuse": 0.5, "specular": 0.5},
        "opacity": opacity,
    }

    if color is not None:
        trace_kwargs["color"] = color
        return go.Mesh3d(**trace_kwargs)

    # UV Texture to Vertex Color Conversion
    # This bakes the texture onto the vertices to be visualized by Plotly.
    # We check if the mesh has both UV coordinates and a texture image.
    if (
        hasattr(plotly_mesh.visual, "uv")
        and plotly_mesh.visual.uv.size > 0
        and hasattr(plotly_mesh.visual, "material")
        and hasattr(plotly_mesh.visual.material, "baseColorTexture")
    ):
        # Correctly convert the UV texture to vertex colors.
        # We must pass the UV coordinates and the texture image to the function.
        plotly_mesh.visual.vertex_colors = trimesh.visual.color.uv_to_interpolated_color(
            uv=plotly_mesh.visual.uv, image=plotly_mesh.visual.material.baseColorTexture
        )

    # Check for visual attributes and apply them to the trace
    if hasattr(plotly_mesh.visual, "vertex_colors") and plotly_mesh.visual.vertex_colors.size > 0:
        trace_kwargs["vertexcolor"] = plotly_mesh.visual.vertex_colors
    elif hasattr(plotly_mesh.visual, "face_colors") and plotly_mesh.visual.face_colors.size > 0:
        colors = plotly_mesh.visual.face_colors
        if colors.ndim == 2 and colors.shape[1] >= 3:
            trace_kwargs["facecolor"] = [f"rgba({r},{g},{b},{a / 255.0})" for r, g, b, a in colors]

    # Create a plotly Mesh3d trace
    return go.Mesh3d(**trace_kwargs)


def pointcloud_to_plotly_scatter(
    points: np.ndarray, colors: Image.Image | np.ndarray | str | None = None
) -> go.Scatter3d:
    """Convert a 3D point cloud to a Plotly scatter plot."""
    # Flatten the points array to get x, y, z coordinates
    flat_points = points.reshape(-1, 3)
    good_idxs = np.isfinite(flat_points)
    good_idxs = np.any(good_idxs, axis=1)
    x = flat_points[:, 0][good_idxs]
    y = flat_points[:, 1][good_idxs]
    z = flat_points[:, 2][good_idxs]

    if isinstance(colors, Image.Image):
        colors = np.array(colors)

    if isinstance(colors, np.ndarray):
        # Flatten image for coloring
        colors = colors.reshape(-1, 3)[good_idxs]
        colors = [f"rgb({','.join([str(int(cc)) for cc in c.tolist()])})" for c in colors]

    return go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={"size": 1, "color": colors}, showlegend=False)


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
