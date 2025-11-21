# tinyvis

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/alex-bene/tinyvis/main.svg)](https://results.pre-commit.ci/latest/github/alex-bene/tinyvis/main)
[![Development Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/alex-bene/tinyvis)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small collection of visualization tools for 3D meshes and point clouds using Plotly.

## Features

- **3D Mesh Visualization**: Easily visualize `trimesh.Trimesh` objects in interactive Plotly 3D viewers.
- **Texture Support**: Automatically converts UV textures to vertex colors for display in Plotly.
- **Point Cloud Support**: Visualize large 3D point clouds (`numpy.ndarray`) as scatter plots.
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

### Basic Visualization

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
    pointcloud=pointcloud,
    figure_height=800
)

fig.show()
```

### Advanced Usage

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
    points=points,
    points_names=["Origin", "Top"],
    points_colors=["green", "yellow"],
    points_sizes=[5, 10]
)

fig.show()
```

## API Reference

### `visualize`

The main function `visualize` accepts the following arguments:

- `meshes` (list[trimesh.Trimesh] | None): List of meshes to visualize.
- `meshes_colors` (list[str] | None): List of colors for the meshes.
- `meshes_opacities` (list[float] | None): List of opacities for the meshes.
- `pointcloud` (np.ndarray | None): A (N, 3) array representing a point cloud.
- `pointcloud_colors` (Image.Image | np.ndarray | str | None): Colors for the point cloud.
- `points` (np.ndarray | None): Specific points to highlight.
- `points_colors` (list[str] | None): Colors for the specific points.
- `points_names` (list[str] | None): Names/Labels for the specific points.
- `points_sizes` (list[float] | None): Sizes for the specific points.
- `figure_height` (int): Height of the Plotly figure (default: 800).

Returns a `plotly.graph_objects.Figure`.

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
- [ ] Support pytorch3d Meshes/Pointclouds & `trimesh.PointCloud` as input to `visualize`
- [ ] Support pytorch3d `CameraBase` as input to `visualize`

## License

This project is licensed under the [MIT License](LICENSE).
