#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

################################################################################

def run_RivlinCube_Mesh(
        dim    : int  = 3 ,
        params : dict = {}):

    X0 = params.get("X0", 0.)
    X1 = params.get("X1", 1.)
    Y0 = params.get("Y0", 0.)
    Y1 = params.get("Y1", 1.)
    if (dim==3): Z0 = params.get("Z0", 0.)
    if (dim==3): Z1 = params.get("Z1", 1.)
    l = params.get("l", 1.)

    mesh_filebasename = params.get("mesh_filebasename", "mesh")

    LX = X1-X0
    LY = Y1-Y0
    if (dim==3): LZ = Z1-Z0

    NX = int(LX/l)
    NY = int(LY/l)
    if (dim==3): NZ = int(LZ/l)

    if (dim==2):
        mesh = dolfin.RectangleMesh(
            dolfin.Point(X0, Y0, 0.), dolfin.Point(Y1, Y1, 0.),
            NX, NY,
            "crossed")
    elif (dim==3):
        mesh = dolfin.BoxMesh(
            dolfin.Point(X0, Y0, Z0), dolfin.Point(X1, Y1, Z1),
            NX, NY, NZ)

    xdmf_file_mesh = dolfin.XDMFFile(mesh_filebasename+".xdmf")
    xdmf_file_mesh.write(mesh)
    xdmf_file_mesh.close()

    dolfin.File(mesh_filebasename+".xml") << mesh

    ################################################## Subdomains & Measures ###

    xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=X0)
    xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=X1)
    ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=Y0)
    ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=Y1)
    if (dim==3): zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=Z0)
    if (dim==3): zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=Z1)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    boundaries_mf.set_all(0)
    xmin_sd.mark(boundaries_mf, xmin_id)
    xmax_sd.mark(boundaries_mf, xmax_id)
    ymin_sd.mark(boundaries_mf, ymin_id)
    ymax_sd.mark(boundaries_mf, ymax_id)
    if (dim==3): zmin_sd.mark(boundaries_mf, zmin_id)
    if (dim==3): zmax_sd.mark(boundaries_mf, zmax_id)

    # xdmf_file_boundaries = dolfin.XDMFFile("boundaries.xdmf")
    # xdmf_file_boundaries.write(boundaries_mf)
    # xdmf_file_boundaries.close()

    if (dim==2):
        return mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id
    elif (dim==3):
        return mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id
