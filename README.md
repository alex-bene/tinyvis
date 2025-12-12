# tinyvis

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/alex-bene/tinyvis/main.svg)](https://results.pre-commit.ci/latest/github/alex-bene/tinyvis/main)
[![Development Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/alex-bene/tinyvis)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small collection of visualization tools for 3D meshes and point clouds using Plotly.

## Features

- **3D Mesh Visualization**: Easily visualize `trimesh.Trimesh` objects in interactive Plotly 3D viewers.
- **Offscreen Rendering**: High-quality offscreen rendering using `pyrender` via the `PyRenderer` class.
- **Texture Support**: Automatically converts UV textures to vertex colors for display in Plotly.
- **Point Cloud Support**: Visualize multiple 3D point clouds (`numpy.ndarray`, `torch.Tensor`, or `pytorch3d.structures.Pointclouds`) as scatter plots.
- **Custom Points**: Add individual points with custom names, colors, and sizes to your scene.

## Installation

You can install `tinyvis` directly from GitHub:

```bash
pip install git+https://github.com/alex-bene/tinyvis.git
```

Or if you are using `uv`:

```bash
uv add git+https://github.com/alex-bene/tinyvis.git
```

## Usage

### Interactive Visualization

Here is a simple example of how to visualize a mesh and a point cloud:

```python
import trimesh
import numpy as np
from tinyvis import visualize

# Create a simple box mesh
mesh = trimesh.creation.box()

# Generate a random point cloud
pointcloud = np.random.rand(100, 3)

# Visualize them together
fig = visualize(
    meshes=[mesh],
    pointclouds=[pointcloud],
    figure_height=800
)

fig.show()
```

### Advanced Interactive Visualization

You can customize colors, opacities, and add specific labeled points:

```python
import trimesh
import numpy as np
from tinyvis import visualize

# Create meshes
mesh1 = trimesh.creation.box()
mesh1.apply_translation([-1, 0, 0])
mesh2 = trimesh.creation.sphere()
mesh2.apply_translation([1, 0, 0])

# Define specific points of interest
points = np.array([
    [0, 0, 0],
    [0, 1, 0]
])

fig = visualize(
    meshes=[mesh1, mesh2],
    meshes_colors=["red", "blue"],
    meshes_opacities=[0.5, 1.0],
    meshes_names=["Box", "Sphere"],
    points=points,
    points_names=["Origin", "Top"],
    points_colors=["green", "yellow"],
    points_sizes=[5, 10]
)

fig.show()
```

### Offscreen Rendering

You can use `PyRenderer` for high-quality offscreen rendering:

```python
import trimesh
from tinyvis import PyRenderer
from matplotlib import pyplot as plt

# Create a mesh
mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=1.0)

# Initialize renderer
renderer = PyRenderer(image_size=(512, 512))

# Render image
img, depth = renderer.render_image(
    meshes=[mesh],
    view="top",
    render_params={"shadows": True}
)

# Show result
plt.imshow(img)
plt.show()
```

## API Reference

### `visualize`

The main function `visualize` accepts the following arguments:

- `meshes` (list[trimesh.Trimesh] | None): List of meshes to visualize.
- `meshes_colors` (list[str | None] | None): List of colors for the meshes.
- `meshes_opacities` (list[float] | None): List of opacities for the meshes.
- `meshes_names` (list[str] | None): List of names for the meshes.
- `pointclouds` (list[np.ndarray] | None): List of (N, 3) arrays representing point clouds.
- `pointclouds_colors` (list[Image.Image | np.ndarray | str | None] | None): List of colors for the point clouds.
- `pointclouds_names` (list[str] | None): List of names for the point clouds.
- `pointclouds_sizes` (list[float | None] | None): List of marker sizes for the point clouds.
- `points` (np.ndarray | None): Specific points to highlight (N, 3).
- `points_colors` (list[str | None] | None): Colors for the specific points.
- `points_names` (list[str] | None): Names/Labels for the specific points.
- `points_sizes` (list[float | None] | None): Sizes for the specific points.
- `figure_height` (int): Height of the Plotly figure (default: 800).

Returns a `plotly.graph_objects.Figure`.

### `PyRenderer`

The `PyRenderer` class wraps `pyrender` for easy offscreen rendering of `trimesh` objects.

**Initialization**:
```python
renderer = PyRenderer(image_size=(512, 512), use_raymond_lighting=True)
```

**Main Methods**:

- **`render_image(...)`**: Renders a single scene.
  - Supports standard views (`"front"`, `"top"`, etc.) or custom camera poses.
  - Returns `(PIL.Image, np.ndarray)` tuple containing the RGB image and depth map.

- **`render_sequence(...)`**: Renders a list of frames.
  - Efficiently renders animations or video overlays.
  - Accepts `camera_poses` (world-to-camera transforms) for moving cameras.

**Key Features**:
- **Render Flags**: Pass `render_params` dict to enable effects:
  - `shadows`: Enable directional and spot shadows.
  - `all_wireframe`: Render wireframes.
  - `vertex_normals` / `face_normals`: Visualize normals.
  - `flat`: Flat shading.
  - `segmentation`: Render segmentation masks (requires `meshes_colors`).
- **Camera Control**: extensive properties to get/set camera intrinsics and extrinsics (`renderer.camera_pose`, `renderer.camera_focal_length`).

## Development

To contribute to this project, please ensure you have `uv` installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/alex-bene/tinyvis.git
   cd tinyvis
   ```

2. Install dependencies and pre-commit hooks:
   ```bash
   uv sync
   uv run pre-commit install
   ```

3. Run checks manually (optional):
   ```bash
   uv run ruff check
   uv run ruff format
   ```

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. We use [pre-commit](https://pre-commit.com/) hooks to ensure code quality.

- **Local**: Hooks run before every commit (requires `pre-commit install`).
- **GitHub Actions**: Runs on every push to **auto-fix** issues on all branches.
- **pre-commit.ci**: Runs on every push to **check** code quality (fixes are handled by the GitHub Action).

## TODO
- [ ] Support orientation vector visualization
- [ ] Support camera frustum visualization

## License

This project is licensed under the [MIT License](LICENSE).
