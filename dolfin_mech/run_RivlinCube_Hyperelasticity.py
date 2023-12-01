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

def run_RivlinCube_Hyperelasticity(
        dim                                    : int  = 3                               ,
        inverse                                : bool = 0                               ,
        incomp                                 : bool = 0                               ,
        multimaterial                          : bool = 0                               ,
        cube_params                            : dict = {}                              ,
        mat_params                             : dict = {}                              ,
        step_params                            : dict = {}                              ,
        const_params                           : dict = {}                              ,
        load_params                            : dict = {}                              ,
        move                                   : dict = {}                              ,
        get_results                            : bool = 0                               ,
        res_basename                           : str  = "run_RivlinCube_Hyperelasticity",
        write_vtus_with_preserved_connectivity : bool = False                           ,
        verbose                                : bool = 0                               ):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.run_RivlinCube_Mesh(dim=dim, params=cube_params)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.run_RivlinCube_Mesh(dim=dim, params=cube_params)

    if move.get("move", False) == True :
        Umove = move.get("U")
        dolfin.ALE.move(mesh, Umove)

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

    if (inverse):
        problem_type = dmech.InverseHyperelasticityProblem
    else:
        problem_type = dmech.HyperelasticityProblem

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
            mat1_mod = "NHMR"
            mat2_mod = "NHMR"
        else:
            mat1_mod = "CGNHMR"
            mat2_mod = "CGNHMR"
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

    problem = problem_type(
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

    n_steps = step_params.get("n_steps", 1)
    Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.)/n_steps]*n_steps)
    dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.)/n_steps]*n_steps)
    dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.)/n_steps]*n_steps)

    load_type = load_params.get("type", "disp")

    if   (load_type == "disp"):
        u_lst = load_params.get("u_lst", [(k_step+1)*load_params.get("u", +0.5)/n_steps for k_step in range(n_steps)])
    elif (load_type in ("volu0", "volu")):
        f_lst = load_params.get("f_lst", [(k_step+1)*load_params.get("f", +0.5)/n_steps for k_step in range(n_steps)])
    elif (load_type in ("surf0", "surf")):
        f_lst = load_params.get("f_lst", [(k_step+1)*load_params.get("f", +1.0)/n_steps for k_step in range(n_steps)])
    elif (load_type in ("pres0", "pres0_multi", "pres0_inertia", "pres")):
        p_lst = load_params.get("f_lst", [(k_step+1)*load_params.get("f", -0.5)/n_steps for k_step in range(n_steps)])
    elif (load_type in ("pgra0", "pgra")):
        X0_lst = load_params.get("X0_lst", [load_params.get("X0",      [0.5]* dim   )]*n_steps)
        N0_lst = load_params.get("N0_lst", [load_params.get("X0", [1.]+[0.0]*(dim-1))]*n_steps)
        P0_lst = load_params.get("P0_lst", [(k_step+1)*load_params.get("P0", -0.50)/n_steps for k_step in range(n_steps)])
        DP_lst = load_params.get("DP_lst", [(k_step+1)*load_params.get("P0", -0.25)/n_steps for k_step in range(n_steps)])
    elif (load_type == "tens"):
        gamma_lst = load_params.get("gamma_lst", [(k_step+1)*load_params.get("gamma", 0.01)/n_steps for k_step in range(n_steps)])
            
    for k_step in range(n_steps):

        Deltat = Deltat_lst[k_step]
        dt_ini = dt_ini_lst[k_step]
        dt_min = dt_min_lst[k_step]

        k_step = problem.add_step(
            Deltat=Deltat,
            dt_ini=dt_ini,
            dt_min=dt_min)

        if (load_type == "disp"):
            u = u_lst[k_step]
            u_old = u_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_constraint(
                V=problem.get_displacement_function_space().sub(0),
                sub_domains=boundaries_mf,
                sub_domain_id=xmax_id,
                val_ini=u_old, val_fin=u,
                k_step=k_step)
        elif (load_type == "volu0"):
            f = f_lst[k_step]
            f_old = f_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_volume_force0_loading_operator(
                measure=problem.dV,
                F_ini=[f_old]+[0.]*(dim-1), F_fin=[f]+[0.]*(dim-1),
                k_step=k_step)
        elif (load_type == "volu"):
            f = f_lst[k_step]
            f_old = f_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_volume_force_loading_operator(
                measure=problem.dV,
                F_ini=[f_old]+[0.]*(dim-1), F_fin=[f]+[0.]*(dim-1),
                k_step=k_step)
        elif (load_type == "surf0"):
            f = f_lst[k_step]
            f_old = f_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_force0_loading_operator(
                measure=problem.dS(xmax_id),
                F_ini=[f_old]+[0.]*(dim-1), F_fin=[f]+[0.]*(dim-1),
                k_step=k_step)
        elif (load_type == "surf"):
            f = f_lst[k_step]
            f_old = f_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_force_loading_operator(
                measure=problem.dS(xmax_id),
                F_ini=[f_old]+[0.]*(dim-1), F_fin=[f]+[0.]*(dim-1),
                k_step=k_step)
        elif (load_type == "pres0"):
            p = p_lst[k_step]
            p_old = p_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(xmax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
        elif (load_type == "pres0_multi"):
            p = p_lst[k_step]
            p_old = p_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(xmax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(ymax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            if (dim==3): problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(zmax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
        elif (load_type == "pres0_inertia"):
            p = p_lst[k_step]
            p_old = p_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_inertia_operator(
                measure=problem.dV,
                rho_val=1e-2,
                k_step=k_step)
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(xmin_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(xmax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(ymin_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(ymax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            if (dim==3): problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(zmin_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
            if (dim==3): problem.add_surface_pressure0_loading_operator(
                measure=problem.dS(zmax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
        elif (load_type == "pres"):
            p = p_lst[k_step]
            p_old = p_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_pressure_loading_operator(
                measure=problem.dS(xmax_id),
                P_ini=p_old, P_fin=p,
                k_step=k_step)
        elif (load_type == "pgra0"):
            X0 = X0_lst[k_step]
            N0 = N0_lst[k_step]
            P0 = P0_lst[k_step]
            DP = DP_lst[k_step]
            P0_old = P0_lst[k_step-1] if (k_step > 0) else 0.
            DP_old = DP_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_pressure_gradient0_loading_operator(
                measure=problem.dS(),
                X0_val=X0,
                N0_val=N0,
                P0_ini=P0_old, P0_fin=P0,
                DP_ini=DP_old, DP_fin=DP,
                k_step=k_step)
        elif (load_type == "pgra"):
            X0 = X0_lst[k_step]
            N0 = N0_lst[k_step]
            P0 = P0_lst[k_step]
            DP = DP_lst[k_step]
            P0_old = P0_lst[k_step-1] if (k_step > 0) else 0.
            DP_old = DP_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_pressure_gradient_loading_operator(
                measure=problem.dS(),
                X0_val=X0,
                N0_val=N0,
                P0_ini=P0_old, P0_fin=P0,
                DP_ini=DP_old, DP_fin=DP,
                k_step=k_step)
        elif (load_type == "tens"):
            gamma = gamma_lst[k_step]
            gamma_old = gamma_lst[k_step-1] if (k_step > 0) else 0.
            problem.add_surface_tension_loading_operator(
                measure=problem.dS,
                gamma_ini=gamma_old, gamma_fin=gamma,
                k_step=k_step)

    ################################################# Quantities of Interest ###

    problem.add_global_strain_qois()
    problem.add_global_stress_qois()
    if (incomp): problem.add_global_pressure_qoi()
    if (inverse==0) and (dim==2): problem.add_global_out_of_plane_stress_qois()

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

    if (get_results):
        return (problem.get_displacement_subsol().func, dolfin.Measure("dx", domain=mesh))
