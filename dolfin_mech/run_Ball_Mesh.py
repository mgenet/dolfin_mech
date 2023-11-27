#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import gmsh
import meshio

################################################################################

def run_Ball_Mesh(
        params={}):

    X0 = params.get("X0", 0.5)
    Y0 = params.get("Y0", 0.5)
    Z0 = params.get("Z0", 0.5)
    R  = params.get("R" , 0.3)
    l  = params.get("l" , 0.01)

    mesh_filebasename = params.get("mesh_filebasename", "mesh")

    ################################################################### Mesh ###
    
    gmsh.initialize()
    gmsh.clear()
    factory = gmsh.model.occ

    sp = factory.addSphere(xc=X0, yc=Y0, zc=Z0, radius=R)

    factory.synchronize()

    ps = gmsh.model.addPhysicalGroup(dim=3, tags=[sp])

    mesh_gmsh = gmsh.model.mesh

    mesh_gmsh.setSize(gmsh.model.getEntities(0), l)
    mesh_gmsh.generate(dim=3)

    gmsh.write(mesh_filebasename+".vtk")

    gmsh.finalize()

    mesh_meshio = meshio.read(mesh_filebasename+".vtk")

    meshio.write(mesh_filebasename+".xdmf", mesh_meshio)

    mesh = dolfin.Mesh()
    dolfin.XDMFFile(mesh_filebasename+".xdmf").read(mesh)

    dolfin.File(mesh_filebasename+".xml") << mesh

    ############################################################# Boundaries ###

    boundaries_mf = dolfin.MeshFunction(
        value_type="size_t",
        mesh=mesh,
        dim=1)

    boundaries_mf.set_all(0)

    S_sd = dolfin.AutoSubDomain(
        lambda x, on_boundary:
            on_boundary and\
            dolfin.near(
                (x[0]-X0)**2 + (x[1]-Y0)**2 + (x[2]-Z0)**2,
                R**2,
                eps=1e-3))

    S_id = 1; S_sd.mark(boundaries_mf, S_id)

    # dolfin.XDMFFile(mesh_filebasename+"-boundaries.xdmf").write(boundaries_mf)

    ################################################################# Return ###

    return mesh, boundaries_mf, S_id
