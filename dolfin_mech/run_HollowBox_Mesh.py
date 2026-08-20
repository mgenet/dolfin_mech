#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Haotian XIAO, 2024-2027                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import gmsh
import meshio
import numpy
import math

import dolfin_mech as dmech

################################################################################

def setPeriodic(dim, coord, xmin, ymin, zmin, xmax, ymax, zmax, e=1e-6):
    # From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744

    dx = (xmax - xmin) if (coord == 0) else 0.
    dy = (ymax - ymin) if (coord == 1) else 0.
    dz = (zmax - zmin) if (coord == 2) else 0.
    d = max(dx, dy, dz)
    e *= d

    smin = gmsh.model.getEntitiesInBoundingBox(
        xmin      - e, ymin      - e, zmin      - e,
        xmax - dx + e, ymax - dy + e, zmax - dz + e,
        dim-1)
    # print ("smin:",smin)
    for i in smin:
        bb = gmsh.model.getBoundingBox(*i)
        bbe = [bb[0] + dx, bb[1] + dy, bb[2] + dz,
               bb[3] + dx, bb[4] + dy, bb[5] + dz]
        smax = gmsh.model.getEntitiesInBoundingBox(
            bbe[0] - e, bbe[1] - e, bbe[2] - e,
            bbe[3] + e, bbe[4] + e, bbe[5] + e,
            dim-1)
        # print ("smax:",smax)
        for j in smax:
            bb2 = gmsh.model.getBoundingBox(*j)
            bb2e = [bb2[0] - dx, bb2[1] - dy, bb2[2] - dz,
                    bb2[3] - dx, bb2[4] - dy, bb2[5] - dz]
            if (numpy.linalg.norm(numpy.asarray(bb2e) - numpy.asarray(bb)) < e):
                gmsh.model.mesh.setPeriodic(
                    dim-1,
                    [j[1]], [i[1]],
                    [1, 0, 0, dx,\
                     0, 1, 0, dy,\
                     0, 0, 1, dz,\
                     0, 0, 0, 1 ])

################################################################################

def add_regular_hex_surface(occ, cx, cy, R, angle0=0., tag_base=None):
    pt_tags = []
    for k in range(6):
        theta = angle0 + k * math.pi / 3.
        x = cx + R * math.cos(theta)
        y = cy + R * math.sin(theta)
        if tag_base is None:
            pt = occ.addPoint(x, y, 0.)
        else:
            pt = occ.addPoint(x, y, 0., tag=tag_base + k)
        pt_tags.append(pt)

    line_tags = []
    for k in range(6):
        p1 = pt_tags[k]
        p2 = pt_tags[(k + 1) % 6]
        if tag_base is None:
            line = occ.addLine(p1, p2)
        else:
            line = occ.addLine(p1, p2, tag=tag_base + 100 + k)
        line_tags.append(line)

    if tag_base is None:
        loop = occ.addCurveLoop(line_tags)
        surf = occ.addPlaneSurface([loop])
    else:
        loop = occ.addCurveLoop(line_tags, tag=tag_base + 200)
        surf = occ.addPlaneSurface([loop], tag=tag_base + 300)

    return surf

################################################################################


def run_HollowBox_Mesh(params: dict = {}, return_porosity=False):

    dim    = params.get("dim"); assert (dim in (2,3))
    xmin   = params.get("xmin", 0.)
    ymin   = params.get("ymin", 0.)
    zmin   = params.get("zmin", 0.)
    xmax   = params.get("xmax", 1.)
    ymax   = params.get("ymax", 1.)
    zmax   = params.get("zmax", 1.)
    xshift = params.get("xshift", 0.)
    yshift = params.get("yshift", 0.)
    zshift = params.get("zshift", 0.)
    r0     = params.get("r0", 0.2)
    l      = params.get("l", 0.1)

    hole_shape            = params.get("hole_shape", "round")
    angle0                = params.get("angle0", numpy.pi / 2.0)
    add_center_hole       = params.get("add_center_hole", False)

    mesh_filebasename = params.get("mesh_filebasename", "mesh")

    gmsh.initialize()
    gmsh.model.add("HollowBox")

    if (dim == 2):
        box_tag = 1
        rve_tag = 100

        gmsh.model.occ.addRectangle(x=xmin+xshift, y=ymin+yshift, z=0., dx=xmax-xmin, dy=ymax-ymin, tag=box_tag)

        corner_centers = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]

        centers = list(corner_centers)

        if add_center_hole:
            xcenter = 0.5 * (xmin + xmax)
            ycenter = 0.5 * (ymin + ymax)
            centers.append((xcenter, ycenter))

        tool_tags = []

        if (hole_shape == "round"):
            for ihole, (cx, cy) in enumerate(centers):
                hole_tag = 2 + ihole
                gmsh.model.occ.addDisk(
                    xc=cx, yc=cy, zc=0.,
                    rx=r0, ry=r0,
                    tag=hole_tag)
                tool_tags.append((2, hole_tag))

        elif (hole_shape == "hex"):
            for ihole, (cx, cy) in enumerate(centers):
                hole_tag = add_regular_hex_surface(
                    gmsh.model.occ,
                    cx=cx,
                    cy=cy,
                    R=r0,
                    angle0=angle0,
                    tag_base=1000 + 100 * ihole)
                tool_tags.append((2, hole_tag))

        else:
            gmsh.finalize()
            raise ValueError("Unsupported hole_shape for dim=2. Use 'round' or 'hex'.")

        gmsh.model.occ.cut(
            objectDimTags=[(2, box_tag)],
            toolDimTags=tool_tags,
            tag=rve_tag)

        gmsh.model.occ.synchronize()
        #rve_tag = box_tag
        gmsh.model.addPhysicalGroup(dim=2, tags=[rve_tag])
        dmech.setPeriodic(dim=2, coord=0, xmin=xmin+xshift, ymin=ymin+yshift, zmin=0., xmax=xmax+xshift, ymax=ymax+yshift, zmax=0.)
        dmech.setPeriodic(dim=2, coord=1, xmin=xmin+xshift, ymin=ymin+yshift, zmin=0., xmax=xmax+xshift, ymax=ymax+yshift, zmax=0.)
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=2)

    if (dim==3):
        if (hole_shape != "round"):
            gmsh.finalize()
            raise ValueError("For dim=3, only hole_shape='round' is currently supported.")

        if add_center_hole:
            gmsh.finalize()
            raise ValueError("For dim=3, add_center_hole is currently not supported.")

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
        setPeriodic(dim=3, coord=0, xmin=xmin+xshift, ymin=ymin+yshift, zmin=zmin+zshift, xmax=xmax+xshift, ymax=ymax+yshift, zmax=zmax+zshift)
        setPeriodic(dim=3, coord=1, xmin=xmin+xshift, ymin=ymin+yshift, zmin=zmin+zshift, xmax=xmax+xshift, ymax=ymax+yshift, zmax=zmax+zshift)
        setPeriodic(dim=3, coord=2, xmin=xmin+xshift, ymin=ymin+yshift, zmin=zmin+zshift, xmax=xmax+xshift, ymax=ymax+yshift, zmax=zmax+zshift)
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=3)

    gmsh.write(mesh_filebasename+".vtk")
    gmsh.finalize()

    mesh = meshio.read(mesh_filebasename+".vtk")
    if (dim==2): mesh.points = mesh.points[:, :2]
    meshio.write(mesh_filebasename+".xdmf", mesh)

    mesh = dolfin.Mesh()
    dolfin.XDMFFile(mesh_filebasename+".xdmf").read(mesh)

    if return_porosity:
        dV = dolfin.Measure("dx", domain=mesh)
        skeleton_volume = dolfin.assemble(
            dolfin.Constant(1.0) * dV)

        if dim == 2:
            box_volume = (
                (xmax - xmin)
                * (ymax - ymin)
            )
        else:
            box_volume = (
                (xmax - xmin)
                * (ymax - ymin)
                * (zmax - zmin)
            )

        porosity = 1.0 - skeleton_volume / box_volume

        return mesh, porosity

    return mesh