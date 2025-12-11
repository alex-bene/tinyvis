"""Module to translate meshes representations to plotly Mesh3d."""

import trimesh
from plotly import graph_objects as go


def trimesh_to_plotly_mesh(
    mesh: trimesh.Trimesh,
    color: str | None = None,
    opacity: float = 1.0,
    flatshading: bool = False,
    name: str | None = None,
    showlegend: bool = True,
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
        name (str | None): Name of the mesh trace.
        showlegend (bool): Whether to show the mesh in the legend.

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
        "name": name,
        "showlegend": showlegend,
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
