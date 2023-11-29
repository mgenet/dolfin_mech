#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2023                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import gmsh
import meshio
# import mshr

import dolfin_mech as dmech

################################################################################

def setPeriodic(coord, xmin, ymin, zmin, xmax, ymax, zmax, e=1e-6):
    # From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744
    
    smin = gmsh.model.getEntitiesInBoundingBox(
        xmin - e,
        ymin - e,
        zmin - e,
        (xmin + e) if (coord == 0) else (xmax + e),
        (ymin + e) if (coord == 1) else (ymax + e),
        (zmin + e) if (coord == 2) else (zmax + e),
        2)
    dx = (xmax - xmin) if (coord == 0) else 0
    dy = (ymax - ymin) if (coord == 1) else 0
    dz = (zmax - zmin) if (coord == 2) else 0

    for i in smin:
        bb = gmsh.model.getBoundingBox(i[0], i[1])
        bbe = [bb[0] - e + dx, bb[1] - e + dy, bb[2] - e + dz,
               bb[3] + e + dx, bb[4] + e + dy, bb[5] + e + dz]
        smax = gmsh.model.getEntitiesInBoundingBox(
            bbe[0], bbe[1], bbe[2],
            bbe[3], bbe[4], bbe[5])
        for j in smax:
            bb2 = list(gmsh.model.getBoundingBox(j[0], j[1]))
            bb2[0] -= dx; bb2[1] -= dy; bb2[2] -= dz;
            bb2[3] -= dx; bb2[4] -= dy; bb2[5] -= dz;
            if ((abs(bb2[0] - bb[0]) < e) and (abs(bb2[1] - bb[1]) < e) and
                (abs(bb2[2] - bb[2]) < e) and (abs(bb2[3] - bb[3]) < e) and
                (abs(bb2[4] - bb[4]) < e) and (abs(bb2[5] - bb[5]) < e)):
                gmsh.model.mesh.setPeriodic(
                    2,
                    [j[1]], [i[1]],
                    [1, 0, 0, dx,\
                     0, 1, 0, dy,\
                     0, 0, 1, dz,\
                     0, 0, 0, 1 ])

################################################################################

def run_HollowBox_Mesh(
        params : dict = {}):

    dim       = params.get("dim"); assert (dim in (2,3))
    xmin      = params.get("xmin", 0.)
    ymin      = params.get("ymin", 0.)
    zmin      = params.get("zmin", 0.)
    xmax      = params.get("xmax", 1.)
    ymax      = params.get("ymax", 1.)
    zmax      = params.get("zmax", 1.)
    xshift    = params.get("xshift", 0.)
    yshift    = params.get("yshift", 0.)
    zshift    = params.get("zshift", 0.)
    r0        = params.get("r0", 0.2)
    l         = params.get("l", 0.1)

    mesh_filebasename = params.get("mesh_filebasename", "mesh")

    ################################################################### Mesh ###

    gmsh.initialize()

    if (dim==2):
        box_tag   = 1
        hole1_tag = 2
        hole2_tag = 3
        hole3_tag = 4
        hole4_tag = 5
        rve_tag   = 6

        gmsh.model.occ.addRectangle(x=xmin+xshift, y=ymin+yshift, z=0., dx=xmax-xmin, dy=ymax-ymin, tag=box_tag)
        gmsh.model.occ.addDisk(xc=xmin, yc=ymin, zc=0., rx=r0, ry=r0, tag=hole1_tag)
        gmsh.model.occ.addDisk(xc=xmax, yc=ymin, zc=0., rx=r0, ry=r0, tag=hole2_tag)
        gmsh.model.occ.addDisk(xc=xmax, yc=ymax, zc=0., rx=r0, ry=r0, tag=hole3_tag)
        gmsh.model.occ.addDisk(xc=xmin, yc=ymax, zc=0., rx=r0, ry=r0, tag=hole4_tag)
        gmsh.model.occ.cut(objectDimTags=[(2, box_tag)], toolDimTags=[(2, hole1_tag), (2, hole2_tag), (2, hole3_tag), (2, hole4_tag)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=2, tags=[rve_tag])
        dmech.setPeriodic(coord=0, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)
        dmech.setPeriodic(coord=1, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)
        # gmsh.model.mesh.setPeriodic(dim=1, tags=[2], tagsMaster=[1], affineTransform=[1, 0, 0, xmax-xmin,\
        #                                                                               0, 1, 0, 0        ,\
        #                                                                               0, 0, 1, 0        ,\
        #                                                                               0, 0, 0, 1        ])
        # gmsh.model.mesh.setPeriodic(dim=1, tags=[4], tagsMaster=[3], affineTransform=[1, 0, 0, 0        ,\
        #                                                                               0, 1, 0, ymax-ymin,\
        #                                                                               0, 0, 1, 0        ,\
        #                                                                               0, 0, 0, 1        ])
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=2)
    if (dim==3):
        box_tag   = 1
        hole1_tag = 2
        hole2_tag = 3
        hole3_tag = 4
        hole4_tag = 5
        hole5_tag = 6
        hole6_tag = 7
        hole7_tag = 8
        hole8_tag = 9
        rve_tag   = 10

        gmsh.model.occ.addBox(x=xmin+xshift, y=ymin+yshift, z=zmin+zshift, dx=xmax-xmin, dy=ymax-ymin, dz=zmax-zmin, tag=box_tag)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymin, zc=zmin, radius=r0, tag=hole1_tag)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymin, zc=zmin, radius=r0, tag=hole2_tag)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymax, zc=zmin, radius=r0, tag=hole3_tag)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymax, zc=zmin, radius=r0, tag=hole4_tag)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymin, zc=zmax, radius=r0, tag=hole5_tag)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymin, zc=zmax, radius=r0, tag=hole6_tag)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymax, zc=zmax, radius=r0, tag=hole7_tag)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymax, zc=zmax, radius=r0, tag=hole8_tag)
        gmsh.model.occ.cut(objectDimTags=[(3, box_tag)], toolDimTags=[(3, hole1_tag), (3, hole2_tag), (3, hole3_tag), (3, hole4_tag), (3, hole5_tag), (3, hole6_tag), (3, hole7_tag), (3, hole8_tag)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=3, tags=[rve_tag])
        setPeriodic(coord=0, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)
        setPeriodic(coord=1, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)
        setPeriodic(coord=2, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)
        # gmsh.model.mesh.setPeriodic(dim=2, tags=[2], tagsMaster=[1], affineTransform=[1, 0, 0, xmax-xmin,\
        #                                                                               0, 1, 0, 0        ,\
        #                                                                               0, 0, 1, 0        ,\
        #                                                                               0, 0, 0, 1        ])
        # gmsh.model.mesh.setPeriodic(dim=2, tags=[4], tagsMaster=[3], affineTransform=[1, 0, 0, 0        ,\
        #                                                                               0, 1, 0, ymax-ymin,\
        #                                                                               0, 0, 1, 0        ,\
        #                                                                               0, 0, 0, 1        ])
        # gmsh.model.mesh.setPeriodic(dim=2, tags=[6], tagsMaster=[5], affineTransform=[1, 0, 0, 0        ,\
        #                                                                               0, 1, 0, 0        ,\
        #                                                                               0, 0, 1, zmax-zmin,\
        #                                                                               0, 0, 0, 1        ])
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=3)

    gmsh.write(mesh_filebasename+"-mesh.vtk")
    gmsh.finalize()

    mesh = meshio.read(mesh_filebasename+"-mesh.vtk")
    if (dim==2): mesh.points = mesh.points[:, :2]
    meshio.write(mesh_filebasename+"-mesh.xdmf", mesh)

    mesh = dolfin.Mesh()
    dolfin.XDMFFile(mesh_filebasename+"-mesh.xdmf").read(mesh)
    
    # if   (dim==2):
    #     geometry = mshr.Rectangle(dolfin.Point(xmin, ymin), dolfin.Point(xmax, xmax))\
    #              - mshr.Circle(dolfin.Point(x0, y0), r0)
    # elif (dim==3):
    #     geometry = mshr.Box(dolfin.Point(xmin, ymin, zmin), dolfin.Point(xmax, ymax, zmax))\
    #              - mshr.Sphere(dolfin.Point(x0, y0, z0), r0)
    # mesh = mshr.generate_mesh(geometry, 10)

    return mesh
