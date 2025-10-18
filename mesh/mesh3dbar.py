import gmsh
import sys

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

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("bar3D")

# Geometry dimensions
x_len, y_len, z_len = 0.5, 0.5, 1.0
mesh_size = 0.05  # Desired resolution

# Create corner points with mesh size
p1 = gmsh.model.occ.addPoint(0,     0,     0,     mesh_size)
p2 = gmsh.model.occ.addPoint(x_len, 0,     0,     mesh_size)
p3 = gmsh.model.occ.addPoint(x_len, y_len, 0,     mesh_size)
p4 = gmsh.model.occ.addPoint(0,     y_len, 0,     mesh_size)
p5 = gmsh.model.occ.addPoint(0,     0,     z_len, mesh_size)
p6 = gmsh.model.occ.addPoint(x_len, 0,     z_len, mesh_size)
p7 = gmsh.model.occ.addPoint(x_len, y_len, z_len, mesh_size)
p8 = gmsh.model.occ.addPoint(0,     y_len, z_len, mesh_size)

# Create box using one volume command
box = gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
gmsh.model.occ.synchronize()

# Get and tag boundary faces
left   = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, 0, 0, y_len, z_len, 2)
right  = gmsh.model.occ.getEntitiesInBoundingBox(x_len, 0, 0, x_len, y_len, z_len, 2)
front  = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, 0, x_len, 0, z_len, 2)
back   = gmsh.model.occ.getEntitiesInBoundingBox(0, y_len, 0, x_len, y_len, z_len, 2)
bottom = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, 0, x_len, y_len, 0, 2)
top    = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, z_len, x_len, y_len, z_len, 2)

# Add physical tags (useful for BCs in FEniCS or other solvers)
if top:    gmsh.model.addPhysicalGroup(2, [s[1] for s in top], tag=1)
if bottom: gmsh.model.addPhysicalGroup(2, [s[1] for s in bottom], tag=2)
gmsh.model.addPhysicalGroup(3, [box], tag=10)

# Generate mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

# Export mesh files
gmsh.write("bar3D.msh")
gmsh.write("bar3D.vtk")

convert_vtk_to_xdmf("bar3D", dim=3)
gmsh.finalize()