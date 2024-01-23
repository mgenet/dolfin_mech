#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import math
import numpy

import dolfin_mech as dmech

################################################################################

def run_Ball_Hyperelasticity(
        incomp=0,
        mesh_params={},
        mat_params={},
        step_params={},
        load_params={},
        res_basename="run_Ball_Hyperelasticity",
        write_vtus_with_preserved_connectivity=False,
        verbose=0):

    ################################################################### Mesh ###

    X0 = mesh_params.get("X0", 0.5)
    Y0 = mesh_params.get("Y0", 0.5)
    Z0 = mesh_params.get("Z0", 0.5)
    R  = mesh_params.get("R" , 0.3)

    mesh, boundaries_mf, S_id = dmech.run_Ball_Mesh(
        params=mesh_params)

    ################################################################ Problem ###

    if (incomp):
        displacement_degree = 2 # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
    else:
        displacement_degree = 1

    problem = dmech.HyperelasticityProblem(
        mesh=mesh,
        define_facet_normals=1,
        boundaries_mf=boundaries_mf,
        displacement_degree=displacement_degree,
        quadrature_degree="default",
        w_incompressibility=incomp,
        elastic_behavior=mat_params)

    ########################################## Boundary conditions & Loading ###

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)

    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min)

    load_type = load_params.get("type", "disp")

    if (load_type == "disp"): # MG20220813: It would be possible to impose the spatially varying displacement directly through an expression, but this would need to be implemented within Constraint, e.g. with a TimeVaryingExpression.
        surface_nodes_coords = [node_coords for node_coords in mesh.coordinates() if dolfin.near((node_coords[0]-X0)**2 + (node_coords[1]-Y0)**2 + (node_coords[2]-Z0)**2, R**2, eps=1e-3)]
        dR = load_params.get("dR", +0.1)
        for X in surface_nodes_coords:
            eR = numpy.subtract(X, [X0, Y0, Z0])
            eR /= numpy.linalg.norm(eR)
            U = dR * eR
            # X_sd = dolfin.AutoSubDomain(lambda x, on_boundary: dolfin.near(x[0], X[0], eps=1e-3) and dolfin.near(x[1], X[1], eps=1e-3)) # MG20220813: OMG this behaves so weird!
            X_sd = dolfin.CompiledSubDomain("near(x[0], x0) && near(x[1], y0) && near(x[2], z0)", x0=X[0], y0=X[1], z0=X[2])
            problem.add_constraint(
                V=problem.get_displacement_function_space(),
                sub_domain=X_sd,
                val_ini=[0.,0.,0.], val_fin=U,
                k_step=k_step,
                method="pointwise")

    ################################################# Quantities of Interest ###

    problem.add_point_displacement_qoi(
        name="U",
        coordinates=[X0+R, Y0, Z0],
        component=0)

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
        write_sol=res_basename*verbose,
        write_vtus=res_basename*verbose,
        write_vtus_with_preserved_connectivity=write_vtus_with_preserved_connectivity)

    success = integrator.integrate()
    assert (success),\
        "Integration failed. Aborting."

    integrator.close()
