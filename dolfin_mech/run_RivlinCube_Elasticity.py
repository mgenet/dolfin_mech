#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from curses import use_default_colors
import dolfin

import dolfin_mech as dmech

################################################################################

def run_RivlinCube_Elasticity(
        dim           : int  = 3                      ,
        incomp        : bool = 0                      ,
        multimaterial : bool = 0                      ,
        cube_params   : dict = {}                     ,
        mat_params    : dict = {}                     ,
        step_params   : dict = {}                     ,
        const_params  : dict = {}                     ,
        load_params   : dict = {}                     ,
        res_basename  : str  = "run_RivlinCube_Elasticity",
        verbose       : bool = 0                      ):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.run_RivlinCube_Mesh(dim=dim, params=cube_params)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.run_RivlinCube_Mesh(dim=dim, params=cube_params)

    if (multimaterial):
        mat1_sd = dolfin.CompiledSubDomain("x[0] <= x0", x0=0.5)
        mat2_sd = dolfin.CompiledSubDomain("x[0] >= x0", x0=0.5)

        mat1_id = 1
        mat2_id = 2

        domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
        domains_mf.set_all(0)
        mat1_sd.mark(domains_mf, mat1_id)
        mat2_sd.mark(domains_mf, mat2_id)
    else:
        domains_mf = None

    ################################################################ Problem ###

    if (incomp):
        displacement_degree = 2 # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
        w_incompressibility = 1
    else:
        displacement_degree = 1
        w_incompressibility = 0

    quadrature_degree = "default"
    # quadrature_degree = "full"

    if (multimaterial):
        elastic_behavior = None
        if (incomp):
            mat1_mod = "H_dev"
            mat2_mod = "H_dev"
        else:
            mat1_mod = "H"
            mat2_mod = "H"
        mat1_params = {
            "E":1.,
            "nu":0.5*(incomp)+0.3*(1-incomp)}

        mat2_params = {
            "E":10.,
            "nu":0.5*(incomp)+0.3*(1-incomp)}
        elastic_behaviors=[
                {"subdomain_id":mat1_id, "model":mat1_mod, "parameters":mat1_params, "suffix":"1"},
                {"subdomain_id":mat2_id, "model":mat2_mod, "parameters":mat2_params, "suffix":"2"}]
    else:
        elastic_behavior = mat_params
        elastic_behaviors = None

    problem = dmech.ElasticityProblem(
        mesh=mesh,
        domains_mf=domains_mf,
        define_facet_normals=1,
        boundaries_mf=boundaries_mf,
        displacement_degree=displacement_degree,
        quadrature_degree=quadrature_degree,
        w_incompressibility=w_incompressibility,
        elastic_behavior=elastic_behavior,
        elastic_behaviors=elastic_behaviors)

    ########################################## Boundary conditions & Loading ###

    const_type = const_params.get("type", "sym")

    if (const_type in ("symx", "sym")):
        problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
    if (const_type in ("symy", "sym")) and (dim >= 2):
        problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
    if (const_type in ("symz", "sym")) and (dim >= 3):
        problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    if (const_type in ("blox")):
        problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=[0.]*dim)
    if (const_type in ("bloy")):
        problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=[0.]*dim)
    if (const_type in ("bloz")):
        problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=[0.]*dim)

    load_type = load_params.get("type", "disp")

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)

    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min)

    if (load_type == "disp"):
        u = load_params.get("u", 1.)
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(0),
            sub_domains=boundaries_mf,
            sub_domain_id=xmax_id,
            val_ini=0., val_fin=u,
            k_step=k_step)
    elif (load_type == "volu"):
        f = load_params.get("f", 1.)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
    elif (load_type == "surf"):
        f = load_params.get("f", 1.)
        problem.add_surface_force0_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
    elif (load_type == "pres"):
        p = load_params.get("p", -1.)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
    elif (load_type == "pgra"):
        X0 = load_params.get("X0", [0.5]*dim)
        N0 = load_params.get("N0", [1.]+[0.]*(dim-1))
        P0 = load_params.get("P0", -1.0)
        DP = load_params.get("DP", -0.5)
        problem.add_surface_pressure_gradient0_loading_operator(
            measure=problem.dS(),
            X0_val=X0,
            N0_val=N0,
            P0_ini=0., P0_fin=P0,
            DP_ini=0., DP_fin=DP,
            k_step=k_step)
    elif (load_type == "tens"):
        gamma = load_params.get("gamma", 0.01)
        problem.add_surface_tension0_loading_operator(
            measure=problem.dS,
            gamma_ini=0.0, gamma_fin=gamma,
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
