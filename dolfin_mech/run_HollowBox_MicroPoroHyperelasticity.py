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

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

def run_HollowBox_MicroPoroHyperelasticity(
        dim,
        mesh=None,
        mesh_params=None,
        mat_params={},
        bcs="pbc",
        step_params={},
        load_params={},
        res_basename="run_HollowBox_MicroPoroHyperelasticity",
        verbose=0):

    assert ((mesh is not None) ^ (mesh_params is not None))
    if (mesh is None):
        mesh = dmech.run_HollowBox_Mesh(
            params=mesh_params)

    coord = mesh.coordinates()
    xmax = max(coord[:,0]); xmin = min(coord[:,0])
    ymax = max(coord[:,1]); ymin = min(coord[:,1])
    if (dim==2):    
        bbox = [xmin, xmax, ymin, ymax]
        vertices = numpy.array([[xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax]])
        a1 = vertices[1,:]-vertices[0,:] # first vector generating periodicity
        a2 = vertices[3,:]-vertices[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        tol = 1E-8
        assert numpy.linalg.norm(vertices[2, :]-vertices[3, :] - a1) <= tol
        assert numpy.linalg.norm(vertices[2, :]-vertices[1, :] - a2) <= tol
    elif (dim==3):    
        zmax = max(coord[:,2]); zmin = min(coord[:,2])
        bbox = [xmin, xmax, ymin, ymax, zmin, zmax]
        vertices = numpy.array([[xmin, ymin, zmin],
                                [xmax, ymin, zmin],
                                [xmax, ymax, zmin],
                                [xmin, ymax, zmin],
                                [xmin, ymin, zmax],
                                [xmax, ymin, zmax],
                                [xmax, ymax, zmax],
                                [xmin, ymax, zmax]])

    ################################################## Subdomains & Measures ###

    tol = 1E-8
    xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmin, tol=tol)
    xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmax, tol=tol)
    ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymin, tol=tol)
    ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymax, tol=tol)
    if (dim==3): zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmin, tol=tol)
    if (dim==3): zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmax, tol=tol)

    # if (dim==2):
    #     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, r0=r0)
    # elif (dim==3):
    #     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, z0=z0, r0=r0)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6
    # sint_id = 9

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    boundaries_mf.set_all(0)

    xmin_sd.mark(boundaries_mf, xmin_id)
    xmax_sd.mark(boundaries_mf, xmax_id)
    ymin_sd.mark(boundaries_mf, ymin_id)
    ymax_sd.mark(boundaries_mf, ymax_id)
    if (dim==3): zmin_sd.mark(boundaries_mf, zmin_id)
    if (dim==3): zmax_sd.mark(boundaries_mf, zmax_id)
    # sint_sd.mark(boundaries_mf, sint_id)

    if (verbose):
        xdmf_file_boundaries = dolfin.XDMFFile(res_basename+"-boundaries.xdmf")
        xdmf_file_boundaries.write(boundaries_mf)
        xdmf_file_boundaries.close()

    ################################################################ Problem ###

    problem = dmech.MicroPoroHyperelasticityProblem(
        mesh=mesh,
        mesh_bbox=bbox,
        vertices=vertices,
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

    pf = load_params.get("pf", 0.)
    problem.add_surface_pressure_loading_operator(
        measure=problem.dS(0),
        P_ini=0., P_fin=pf,
        k_step=k_step)

    for i in range(dim):
     for j in range (dim):
        U_bar_ij     = load_params.get("U_bar_"+str(i)+str(j)    , None)
        sigma_bar_ij = load_params.get("sigma_bar_"+str(i)+str(j), None)
        assert ((U_bar_ij is not None) ^ (sigma_bar_ij is not None))
        if (U_bar_ij is not None):
            problem.add_macroscopic_stretch_component_penalty_operator(
                i=i, j=j,
                U_bar_ij_ini=0., U_bar_ij_fin=U_bar_ij,
                pen_val=1e6,
                k_step=k_step)
        elif (sigma_bar_ij is not None):
            problem.add_macroscopic_stress_component_constraint_operator(
                i=i, j=j,
                sigma_bar_ij_ini=0., sigma_bar_ij_fin=sigma_bar_ij,
                pf_ini=0., pf_fin=pf,
                k_step=k_step)

    ################################################# Quantities of Interest ###

    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()
    problem.add_macroscopic_solid_hydrostatic_pressure_qoi()
    problem.add_macroscopic_stress_qois()

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
