import gmsh
import sys
import meshio
import numpy as np
def convert_vtk_to_xdmf(mesh_filename, dim=3):
    import meshio
    mesh = meshio.read(mesh_filename + ".vtk")
    
    if dim == 2:
        # For 2D, keep only the x and y coordinates
        mesh.points = mesh.points[:, :2]
    elif dim == 3:
        # For 3D, filter cells: keep only tetrahedral cells.
        cells_3d = []
        cell_data_3d = {}
        
        # mesh.cells is a list of CellBlock objects.
        # We'll keep only those with type "tetra" or "tetra10".
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type in ["tetra", "tetra10"]:
                cells_3d.append(cell_block)
                # If cell_data is available, collect the corresponding data.
                if mesh.cell_data:
                    for key, data_list in mesh.cell_data.items():
                        # Initialize the list for this key if not already done.
                        if key not in cell_data_3d:
                            cell_data_3d[key] = []
                        cell_data_3d[key].append(data_list[i])
        # Update the mesh with the filtered cells and cell data.
        mesh.cells = cells_3d
        if cell_data_3d:
            mesh.cell_data = cell_data_3d

    # Write out the filtered mesh to an XDMF file.
    meshio.write(mesh_filename + ".xdmf", mesh)


gmsh.initialize()
gmsh.model.add("bar_with_pillars")

# --- Bar dimensions ---
x_len, y_len, z_len = 0.5,0.5,1
mesh_size = 0.02  # Adjust for resolution
pillar_radius = 0.05
pillar_spacing = 0.15  # space between holes
pillar_rows_z = [0.25, 0.5, 0.75]  # Z-positions (normalized) of 3 pillar rows

# --- Create main bar ---
bar = gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)

# --- Create cylindrical holes (parallel to x-axis) ---
cylinders = []
for z_frac in pillar_rows_z:
    z = z_frac * z_len
    y_positions = np.arange(pillar_spacing / 2, y_len, pillar_spacing)
    for y in y_positions:
        cyl = gmsh.model.occ.addCylinder(
            0, y, z,             # base center
            x_len, 0, 0,         # cylinder direction (x-axis)
            pillar_radius        # radius
        )
        cylinders.append((3, cyl))

# --- Cut holes from the bar ---
gmsh.model.occ.synchronize()
cut_result = gmsh.model.occ.cut([(3, bar)], cylinders, removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()

# --- Physical groups for BCs ---
box = cut_result[0][0][1]

faces_top = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, z_len, x_len, y_len, z_len, 2)
faces_bot = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, 0, x_len, y_len, 0, 2)

if faces_top:
    gmsh.model.addPhysicalGroup(2, [f[1] for f in faces_top], tag=1)
if faces_bot:
    gmsh.model.addPhysicalGroup(2, [f[1] for f in faces_bot], tag=2)

gmsh.model.addPhysicalGroup(3, [box], tag=10)

# --- Mesh generation ---
gmsh.model.occ.synchronize()


gmsh.model.mesh.generate(3)

# --- Save output ---
gmsh.write("bar_with_pillars.vtk")
gmsh.write("bar_with_pillars.msh")


convert_vtk_to_xdmf("bar_with_pillars")

gmsh.finalize()