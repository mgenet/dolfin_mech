#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import gmsh
import meshio

################################################################################

def run_Disc_Mesh(
        params : dict = {}):

    X0 = params.get("X0", 0.5)
    Y0 = params.get("Y0", 0.5)
    R  = params.get("R" , 0.3)
    l  = params.get("l" , 0.1)

    mesh_filebasename = params.get("mesh_filebasename", "mesh")

    ################################################################### Mesh ###
    
    gmsh.initialize()
    gmsh.clear()
    factory = gmsh.model.geo

    p0 = factory.addPoint(x=X0  , y=Y0  , z=0, meshSize=l)
    p1 = factory.addPoint(x=X0+R, y=Y0  , z=0, meshSize=l)
    p2 = factory.addPoint(x=X0  , y=Y0+R, z=0, meshSize=l)
    p3 = factory.addPoint(x=X0-R, y=Y0  , z=0, meshSize=l)
    p4 = factory.addPoint(x=X0  , y=Y0-R, z=0, meshSize=l)

    l1 = factory.addCircleArc(p1, p0, p2)
    l2 = factory.addCircleArc(p2, p0, p3)
    l3 = factory.addCircleArc(p3, p0, p4)
    l4 = factory.addCircleArc(p4, p0, p1)

    cl = factory.addCurveLoop([l1, l2, l3, l4])

    s = factory.addPlaneSurface([cl])

    factory.synchronize()

    ps = gmsh.model.addPhysicalGroup(dim=2, tags=[s])

    mesh_gmsh = gmsh.model.mesh

    mesh_gmsh.generate(dim=2)

    gmsh.write(mesh_filebasename+".vtk")

    gmsh.finalize()

    mesh_meshio = meshio.read(mesh_filebasename+".vtk")

    mesh_meshio.points = mesh_meshio.points[:, :2]

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
                (x[0]-X0)**2 + (x[1]-Y0)**2,
                R**2,
                eps=1e-3))

    S_id = 1; S_sd.mark(boundaries_mf, S_id)

    # dolfin.XDMFFile(mesh_filebasename+"-boundaries.xdmf").write(boundaries_mf)

    ################################################################# Points ###

    points_mf = dolfin.MeshFunction(
        value_type="size_t",
        mesh=mesh,
        dim=0)

    points_mf.set_all(0)

    x1 = [X0+R, Y0]
    x1_sd = dolfin.AutoSubDomain(
        lambda x, on_boundary:
            dolfin.near(x[0], x1[0], eps=1e-3)
        and dolfin.near(x[1], x1[1], eps=1e-3))
    x2 = [X0, Y0+R]
    x2_sd = dolfin.AutoSubDomain(
        lambda x, on_boundary:
            dolfin.near(x[0], x2[0], eps=1e-3)
        and dolfin.near(x[1], x2[1], eps=1e-3))
    x3 = [X0-R, Y0]
    x3_sd = dolfin.AutoSubDomain(
        lambda x, on_boundary:
            dolfin.near(x[0], x3[0], eps=1e-3)
        and dolfin.near(x[1], x3[1], eps=1e-3))
    x4 = [X0, Y0-R]
    x4_sd = dolfin.AutoSubDomain(
        lambda x, on_boundary:
            dolfin.near(x[0], x4[0], eps=1e-3)
        and dolfin.near(x[1], x4[1], eps=1e-3))

    x1_id = 1; x1_sd.mark(points_mf, x1_id)
    x2_id = 2; x2_sd.mark(points_mf, x2_id)
    x3_id = 3; x3_sd.mark(points_mf, x3_id)
    x4_id = 4; x4_sd.mark(points_mf, x4_id)

    # dolfin.XDMFFile(mesh_filebasename+"-points.xdmf").write(points_mf)

    return mesh, boundaries_mf, S_id, points_mf, x1_sd, x2_sd, x3_sd, x4_sd
