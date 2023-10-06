#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2023                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import dolfin
import gmsh
import meshio
# import mshr
import sys

import myPythonLibrary as mypy
import dolfin_mech     as dmech

############################################################### test function ###

def microporohyperelasticity(
        dim,
        mesh_params,
        mat_params,
        bcs,
        step_params,
        load_params,
        res_basename,
        verbose):

    ################################################################### Mesh ###

    xmin = 0.
    ymin = 0.
    xmax = 1.
    ymax = 1.
    x0 = 0.3
    y0 = 0.3
    if (dim==3):
        zmin = 0.
        zmax = 1.
        z0 = 0.3
    r0 = 0.2
    l = 0.10
    e = 1e-6

    def setPeriodic(coord, xmin, ymin, zmin, xmax, ymax, zmax, e):
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
    box_tag = 1
    hole_tag = 2
    rve_tag = 3
    if (dim==2):
        gmsh.model.occ.addRectangle(x=xmin, y=ymin, z=0, dx=xmax-xmin, dy=ymax-ymin, tag=box_tag)
        gmsh.model.occ.addDisk(xc=x0, yc=y0, zc=0, rx=r0, ry=r0, tag=hole_tag)
        gmsh.model.occ.cut(objectDimTags=[(2, box_tag)], toolDimTags=[(2, hole_tag)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=2, tags=[rve_tag])
        # setPeriodic(coord=0, xmin=xmin, ymin=ymin, zmin=0., xmax=xmax, ymax=ymax, zmax=0., e=e)
        # setPeriodic(coord=1, xmin=xmin, ymin=ymin, zmin=0., xmax=xmax, ymax=ymax, zmax=0., e=e)
        gmsh.model.mesh.setPeriodic(dim=1, tags=[2], tagsMaster=[1], affineTransform=[1, 0, 0, xmax-xmin,\
                                                                                      0, 1, 0, 0        ,\
                                                                                      0, 0, 1, 0        ,\
                                                                                      0, 0, 0, 1        ])
        gmsh.model.mesh.setPeriodic(dim=1, tags=[4], tagsMaster=[3], affineTransform=[1, 0, 0, 0        ,\
                                                                                      0, 1, 0, ymax-ymin,\
                                                                                      0, 0, 1, 0        ,\
                                                                                      0, 0, 0, 1        ])
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=2)
    elif (dim==3):
        gmsh.model.occ.addBox(x=xmin, y=ymin, z=zmin, dx=xmax-xmin, dy=ymax-ymin, dz=zmax-zmin, tag=box_tag)
        gmsh.model.occ.addSphere(xc=x0, yc=y0, zc=z0, radius=r0, tag=hole_tag)
        gmsh.model.occ.cut(objectDimTags=[(3, box_tag)], toolDimTags=[(3, hole_tag)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=3, tags=[rve_tag])
        setPeriodic(coord=0, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax, e=e)
        setPeriodic(coord=1, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax, e=e)
        setPeriodic(coord=2, xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax, e=e)
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
    gmsh.write(res_basename+"-mesh.vtk")
    gmsh.finalize()

    mesh = meshio.read(res_basename+"-mesh.vtk")
    if (dim==2): mesh.points = mesh.points[:, :2]
    meshio.write(res_basename+"-mesh.xdmf", mesh)

    mesh = dolfin.Mesh()
    dolfin.XDMFFile(res_basename+"-mesh.xdmf").read(mesh)

    # if   (dim==2):
    #     geometry = mshr.Rectangle(dolfin.Point(xmin, ymin), dolfin.Point(xmax, xmax))\
    #              - mshr.Circle(dolfin.Point(x0, y0), r0)
    # elif (dim==3):
    #     geometry = mshr.Box(dolfin.Point(xmin, ymin, zmin), dolfin.Point(xmax, ymax, zmax))\
    #              - mshr.Sphere(dolfin.Point(x0, y0, z0), r0)
    # mesh = mshr.generate_mesh(geometry, 10)

    ################################################## Subdomains & Measures ###

    xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=xmin)
    xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=xmax)
    ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=ymin)
    ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=ymax)
    if (dim==3): zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmin)
    if (dim==3): zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmax)

    if (dim==2):
        sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, r0=r0)
    elif (dim==3):
        sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, z0=z0, r0=r0)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6
    sint_id = 9

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    boundaries_mf.set_all(0)
    xmin_sd.mark(boundaries_mf, xmin_id)
    xmax_sd.mark(boundaries_mf, xmax_id)
    ymin_sd.mark(boundaries_mf, ymin_id)
    ymax_sd.mark(boundaries_mf, ymax_id)
    if (dim==3): zmin_sd.mark(boundaries_mf, zmin_id)
    if (dim==3): zmax_sd.mark(boundaries_mf, zmax_id)
    sint_sd.mark(boundaries_mf, sint_id)

    if (verbose):
        xdmf_file_boundaries = dolfin.XDMFFile(res_basename+"-boundaries.xdmf")
        xdmf_file_boundaries.write(boundaries_mf)
        xdmf_file_boundaries.close()

    ################################################################ Problem ###

    problem = dmech.MicroPoroHyperelasticityProblem(
        mesh=mesh,
        mesh_bbox=[0., 1.]*dim,
        boundaries_mf=boundaries_mf,
        displacement_perturbation_degree=1,
        quadrature_degree=3,
        solid_behavior=mat_params,
        bcs=bcs)

    ################################################################ Loading ###

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)
    dt_max = step_params.get("dt_max", 1.)
    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min,
        dt_max=dt_max)

    load_type = load_params.get("type", "internal_pressure")
    if (load_type == "internal_pressure"):
        pf = load_params.get("pf", +0.2)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(sint_id),
            P_ini=0., P_fin=pf,
            k_step=k_step)
        for k in range(dim):
            for l in range (dim):
                problem.add_macroscopic_stress_component_constraint_operator(
                    i=k, j=l,
                    sigma_bar_ij_ini=0.0, sigma_bar_ij_fin=0.0,
                    pf_ini=0.0, pf_fin=pf,
                    k_step=k_step)
    elif (load_type == "macroscopic_stretch"):
        problem.add_macroscopic_stretch_component_penalty_operator(
            i=0, j=0,
            U_bar_ij_ini=0.0, U_bar_ij_fin=0.5,
            pen_val=1e3,
            k_step=k_step)
        for k in range(dim):
            for l in range (dim):
                if (k!=0 or l!=0):
                    problem.add_macroscopic_stress_component_constraint_operator(
                        i=k, j=l,
                        sigma_bar_ij_ini=0.0, sigma_bar_ij_fin=0.0,
                        pf_ini=0.0, pf_fin=0.0,
                        k_step=k_step)
    elif (load_type == "macroscopic_stress"):
        problem.add_macroscopic_stress_component_constraint_operator(
            i=0, j=0,
            sigma_bar_ij_ini=0.0, sigma_bar_ij_fin=0.5,
            pf_ini=0.0, pf_fin=0.0,
            k_step=k_step)
        for k in range(dim):
            for l in range (dim):
                if (k!=0 or l!=0):
                    problem.add_macroscopic_stress_component_constraint_operator(
                        i=k, j=l,
                        sigma_bar_ij_ini=0.0, sigma_bar_ij_fin=0.0,
                        pf_ini=0.0, pf_fin=0.0,
                        k_step=k_step)

    ################################################# Quantities of Interest ###

    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_stress_qois()
    problem.add_hydrostatic_pressure_qois()

    ################################################################# Solver ###

    solver = dmech.NonlinearSolver(
        problem=problem,
        parameters={
            "sol_tol":[1e-6]*len(problem.subsols),
            "n_iter_max":32},
        relax_type="constant",
        write_iter=0)

    integrator = dmech.TimeIntegrator(
        problem=problem,
        solver=solver,
        parameters={
            "n_iter_for_accel":4,
            "n_iter_for_decel":16,
            "accel_coeff":2,
            "decel_coeff":2},
        print_out=res_basename*verbose,
        print_sta=res_basename*verbose,
        write_qois=res_basename+"-qois",
        write_qois_limited_precision=1,
        write_sol=res_basename*verbose)

    success = integrator.integrate()
    assert (success),\
        "Integration failed. Aborting."

    integrator.close()

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1,
    tester_numpy_tolerance=2*1e-2)

dim_lst  = []
dim_lst += [2]
dim_lst += [3]
for dim in dim_lst:

    bcs_lst  = []
    bcs_lst += ["kubc"]
    bcs_lst += ["pbc"]
    for bcs in bcs_lst:

        load_lst  = []
        load_lst += ["internal_pressure"]
        load_lst += ["macroscopic_stretch"]
        load_lst += ["macroscopic_stress"]
        for load in load_lst:

            print("dim =",dim)
            print("bcs =",bcs)
            print("load =",load)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-bcs="+str(bcs)
            res_basename += "-load="+str(load)

            microporohyperelasticity(
                dim=dim,
                mesh_params={},
                mat_params={"model":"CGNHMR", "parameters":{"E":1.0, "nu":0.3}},
                bcs=bcs,
                step_params={"dt_ini":1e-1, "dt_min":1e-3},
                load_params={"type":load},
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)
