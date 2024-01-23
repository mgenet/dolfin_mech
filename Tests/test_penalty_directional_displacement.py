#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import dolfin
import sys

import myPythonLibrary as mypy
import dolfin_mech     as dmech

############################################################## test function ###

def test_directional_displacement_penalty(
    dim,
    incomp,
    pointwise,
    res_basename,
    verbose=0):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.run_RivlinCube_Mesh(dim=dim)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.run_RivlinCube_Mesh(dim=dim)

    if   (dim==2):
        x0_sd = dolfin.CompiledSubDomain("near(x[0], x0) && near(x[1], y0)", x0=1/2, y0=1/2)
    elif (dim==3):
        x0_sd = dolfin.CompiledSubDomain("near(x[0], x0) && near(x[1], y0) && near(x[2], z0)", x0=1/2, y0=1/2, z0=1/2)

    x0_id = 1

    x0_mf = dolfin.MeshFunction("size_t", mesh, 0) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    x0_mf.set_all(0)
    x0_sd.mark(x0_mf, x0_id)

    # xdmf_file_boundaries = dolfin.XDMFFile(res_basename+"-boundaries.xdmf")
    # xdmf_file_boundaries.write(x0_mf)
    # xdmf_file_boundaries.close()

    ################################################################ Problem ###

    material_parameters = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp),
        "PS":0}

    if (incomp):
        problem = dmech.ElasticityProblem(
            mesh=mesh,
            points_mf=x0_mf,
            define_facet_normals=1,
            boundaries_mf=boundaries_mf,
            displacement_degree=2, # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
            quadrature_degree="default",
            w_incompressibility=1,
            elastic_behavior={"model":"H_dev", "parameters":material_parameters})
    else:
        problem = dmech.ElasticityProblem(
            mesh=mesh,
            points_mf=x0_mf,
            define_facet_normals=1,
            boundaries_mf=boundaries_mf,
            displacement_degree=1,
            quadrature_degree="default",
            w_incompressibility=0,
            elastic_behavior={"model":"H", "parameters":material_parameters})

    ########################################## Boundary conditions & Loading ###

    problem.add_directional_displacement_penalty_operator(
        measure=problem.dS(xmin_id),
        N_val=[1.]+[0.]*(dim-1),
        pen_val=1.)
    problem.add_directional_displacement_penalty_operator(
        measure=problem.dS(ymin_id),
        N_val=[0.,1.]+[0.]*(dim-2),
        pen_val=1.)
    if (dim == 3): problem.add_directional_displacement_penalty_operator(
        measure=problem.dS(zmin_id),
        N_val=[0.,0.,1.],
        pen_val=1.)

    if (pointwise): problem.add_directional_displacement_penalty_operator(
        measure=problem.dP(x0_id),
        N_val=[1.]+[0.]*(dim-1),
        pen_val=10.)

    k_step = problem.add_step(
        Deltat=1.,
        dt_ini=1.,
        dt_min=1.)

    problem.add_constraint(
        V=problem.get_displacement_function_space().sub(0),
        sub_domains=boundaries_mf,
        sub_domain_id=xmax_id,
        val_ini=0., val_fin=1.,
        k_step=k_step)

    ################################################# Quantities of Interest ###

    problem.add_global_strain_qois()
    problem.add_global_stress_qois()
    if (incomp): problem.add_global_pressure_qoi()

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

if (__name__ == "__main__"):

    res_folder = sys.argv[0][:-3]
    test = mypy.Test(
        res_folder=res_folder,
        perform_tests=1,
        stop_at_failure=1,
        clean_after_tests=1)

    dim_lst  = []
    dim_lst += [2]
    dim_lst += [3]
    for dim in dim_lst:

        incomp_lst  = []
        incomp_lst += [0]
        incomp_lst += [1]
        for incomp in incomp_lst:

            if (incomp == 0):
                pointwise_lst  = []
                pointwise_lst += [0]
                pointwise_lst += [1]
            elif (incomp == 1):
                pointwise_lst  = []
                pointwise_lst += [0]
                # pointwise_lst += [1] # MG20211219: vertex integrals only work if solution only has dofs on vertices…
            for pointwise in pointwise_lst:

                print("dim =",dim)
                print("incomp =",incomp)
                print("pointwise =",pointwise)

                res_basename  = sys.argv[0][:-3]
                res_basename += "-dim="+str(dim)
                res_basename += "-incomp="+str(incomp)
                res_basename += "-pointwise="+str(pointwise)

                test_directional_displacement_penalty(
                    dim=dim,
                    incomp=incomp,
                    pointwise=pointwise,
                    res_basename=res_folder+"/"+res_basename,
                    verbose=0)

                test.test(res_basename)
