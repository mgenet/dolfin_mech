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

def RivlinCube_Elasticity(
        dim=3,
        incomp=0,
        multimaterial=0,
        cube_params={},
        mat_params={},
        step_params={},
        load_params={},
        res_basename="RivlinCube_Elasticity",
        estimation_gap=False,
        initialisation_estimation=[],
        verbose=0):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)

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
        displacement_degree = 1 # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
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

    load_type = load_params.get("type", "disp")


    boundary_conditions = []
    if ("inertia" not in load_type):
        problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
        problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
        problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
        # problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
        # if (dim==3):
            # problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
        # boundary_conditions.append([[0.]*3, problem.dS(xmin_id) ])
    
    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)

    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min)
    
    # problem.add_inertia_operator(
    #     measure=problem.dV,
    #     rho_val=float(1e-6),
    #     k_step=k_step)

    surface_forces = []
    volume_forces = []

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
    boundaries_mf.set_all(0)

    if (load_type == "disp"):
        u = load_params.get("u", 1.)
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(0),
            sub_domains=boundaries_mf,
            sub_domain_id=xmax_id,
            val_ini=0., val_fin=u,
            k_step=k_step)
    elif (load_type == "volu+pres"):
        f = load_params.get("f", 1.)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
        volume_forces.append([[f]+[0.]*(dim-1), problem.dV])
        p = load_params.get("p", -1.)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            # measure=problem.dS(0),
            P_ini=0, P_fin=p,
            k_step=k_step)
        surface_forces.append([p,problem.dS(xmax_id)])
        # surface_forces.append([p,problem.dS(0)])
    elif (load_type == "volu"):
        f = load_params.get("f", 1.)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
        volume_forces.append([[f]+[0.]*(dim-1), problem.dV])
    elif (load_type == "surf"):
        f = load_params.get("f", 1.)
        problem.add_surface_force0_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
        surface_forces.append([[f]+[0.]*(dim-1), problem.dS(xmax_id)])
    elif (load_type == "pres"):
        p = load_params.get("p", -1.)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            # measure=problem.dS,
            P_ini=0, P_fin=p,
            k_step=k_step)
        surface_forces.append([p,problem.dS(xmax_id)])
        # surface_forces.append([p,problem.dS])
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
    #     surface_forces.append([gamma,problem.dS])
    # surface_forces.append([0,problem.dS(0)])
    # print("ds", dolfin.assemble(dolfin.Constant(1)*problem.dS))
    # print("ds1", dolfin.assemble(dolfin.Constant(1)*problem.dS(2)))
    # print("ds0", dolfin.assemble(dolfin.Constant(1)*problem.dS(0)))

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
    
    if estimation_gap:
        # for foi in problem.fois:
            # if foi.name == "sigma":
                # sigma_func = foi.func
        # print("div(sigma)", dolfin.assemble(dolfin.inner(dolfin.div(sigma_func), dolfin.div(sigma_func))*problem.dV))
        # sigma_t = dolfin.dot(sigma_func, problem.mesh_normals)
        # norm = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*problem.dS(0))/dolfin.assemble(dolfin.Constant(1)*problem.dS(0)) )**(1/2)
        # norm = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*problem.dS(xmax_id))/dolfin.assemble(dolfin.Constant(1)*problem.dS(xmax_id)) )**(1/2)
        # print("sigma n - t", norm)
        # u = problem.get_subsols_func_lst()[0]
        # print("u estimation=", problem.get_subsols_func_lst()[0].vector())
        # print("gradu", dolfin.grad(u))
        kinematics = problem.kinematics
        # kinematics = dmech.LinearizedKinematics(u=problem.get_subsols_func_lst()[0], u_old=None)
        # print("surface force is", surface_forces)
        dmech.EquilibriumGap(problem=problem, kinematics=kinematics, material_model=elastic_behavior["model"], material_parameters=elastic_behavior["parameters"], initialisation_estimation=initialisation_estimation, surface_forces=surface_forces, volume_forces=volume_forces, boundary_conditions=boundary_conditions, inverse=1, U=problem.get_displacement_subsol().func)
        
        
        

    integrator.close()

    return(problem.get_subsols_func_lst()[0], problem.dV)
