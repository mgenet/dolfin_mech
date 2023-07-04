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
###                                                                          ###
################################################################################

import dolfin
import gmsh
import meshio

################################################################################

def HollowBox_Mesh(
        res_basename,
        mesh_params={}):

    dim               = mesh_params.get("dim")
    width             = mesh_params.get("width")
    r0                = mesh_params.get("r0")
    centering         = mesh_params.get("centering")

    
    if centering:
        shift = width/2
    else:
        shift = 0
    xmin = 0.
    ymin = 0.
    zmin = 0.
    xmax = width
    ymax = width
    zmax = width
    l = width/20
    e = 1e-6


    ############################################################################ Mesh #########
       


    def setPeriodic(coord):
        # From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744
        smin = gmsh.model.getEntitiesInBoundingBox(xmin - e, ymin - e, zmin - e,
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
            smax = gmsh.model.getEntitiesInBoundingBox(bbe[0], bbe[1], bbe[2],
                                                        bbe[3], bbe[4], bbe[5])
            for j in smax:
                bb2 = list(gmsh.model.getBoundingBox(j[0], j[1]))
                bb2[0] -= dx; bb2[1] -= dy; bb2[2] -= dz;
                bb2[3] -= dx; bb2[4] -= dy; bb2[5] -= dz;
                if ((abs(bb2[0] - bb[0]) < e) and (abs(bb2[1] - bb[1]) < e) and
                    (abs(bb2[2] - bb[2]) < e) and (abs(bb2[3] - bb[3]) < e) and
                    (abs(bb2[4] - bb[4]) < e) and (abs(bb2[5] - bb[5]) < e)):
                    gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], [1, 0, 0, dx,\
                                                                    0, 1, 0, dy,\
                                                                    0, 0, 1, dz,\
                                                                    0, 0, 0, 1 ])

    gmsh.initialize()
    # gmsh.clear()

    if (dim==2):

        box_tag = 1
        hole_tag1 = 2
        hole_tag2 = 3
        hole_tag3 = 4
        hole_tag4 = 5
        rve_tag = 6

        gmsh.model.occ.addRectangle(x=xmin+shift, y=ymin+shift, z=0, dx=xmax-xmin, dy=ymax-ymin, tag=box_tag)
        gmsh.model.occ.addDisk(xc=xmin, yc=ymin, zc=0, rx=r0, ry=r0, tag=hole_tag1)
        gmsh.model.occ.addDisk(xc=xmax, yc=ymin, zc=0, rx=r0, ry=r0, tag=hole_tag2)
        gmsh.model.occ.addDisk(xc=xmax, yc=ymax, zc=0, rx=r0, ry=r0, tag=hole_tag3)
        gmsh.model.occ.addDisk(xc=xmin, yc=ymax, zc=0, rx=r0, ry=r0, tag=hole_tag4)
        gmsh.model.occ.cut(objectDimTags=[(2, box_tag)], toolDimTags=[(2, hole_tag1), (2, hole_tag2), (2, hole_tag3), (2, hole_tag4)], tag=rve_tag)

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=2, tags=[rve_tag])

        
        setPeriodic(0)
        setPeriodic(1)
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=2)

        gmsh.write(res_basename+"-mesh.vtk")
        gmsh.finalize()

        mesh = meshio.read(res_basename+"-mesh.vtk")
        mesh.points = mesh.points[:, :2]
        meshio.write(res_basename+"-mesh.xdmf", mesh)


    if (dim==3):

        box_tag = 1
        hole_tag1 = 2
        hole_tag2 = 3
        hole_tag3 = 4
        hole_tag4 = 5
        hole_tag5 = 6
        hole_tag6 = 7
        hole_tag7 = 8
        hole_tag8 = 9
        rve_tag = 10

        gmsh.model.occ.addBox(x=xmin+shift, y=ymin+shift, z=zmin+shift, dx=xmax-xmin, dy=ymax-ymin, dz=zmax-zmin, tag=box_tag)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymin, zc=zmin, radius=r0, tag=hole_tag1)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymin, zc=zmin, radius=r0, tag=hole_tag2)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymax, zc=zmin, radius=r0, tag=hole_tag3)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymax, zc=zmin, radius=r0, tag=hole_tag4)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymin, zc=zmax, radius=r0, tag=hole_tag5)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymin, zc=zmax, radius=r0, tag=hole_tag6)
        gmsh.model.occ.addSphere(xc=xmax, yc=ymax, zc=zmax, radius=r0, tag=hole_tag7)
        gmsh.model.occ.addSphere(xc=xmin, yc=ymax, zc=zmax, radius=r0, tag=hole_tag8)
        gmsh.model.occ.cut(objectDimTags=[(3, box_tag)], toolDimTags=[(3, hole_tag1), (3, hole_tag2), (3, hole_tag3), (3, hole_tag4), (3, hole_tag5), (3, hole_tag6), (3, hole_tag7), (3, hole_tag8)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=3, tags=[rve_tag])
        setPeriodic(0)
        setPeriodic(1)
        setPeriodic(2)

        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=3)

        gmsh.write(res_basename+"-mesh.vtk")
        gmsh.finalize()

        mesh = meshio.read(res_basename+"-mesh.vtk")
        mesh.points = mesh.points[:, :3]
        meshio.write(res_basename+"-mesh.xdmf", mesh)

    # gmsh.write(res_basename+"-mesh.vtk")
    # gmsh.finalize()

    # mesh = meshio.read(res_basename+"-mesh.vtk")
    # mesh.points = mesh.points[:, :3]
    # meshio.write(res_basename+"-mesh.xdmf", mesh)

    mesh = dolfin.Mesh()
    dolfin.XDMFFile(res_basename+"-mesh.xdmf").read(mesh)



    
    return mesh
