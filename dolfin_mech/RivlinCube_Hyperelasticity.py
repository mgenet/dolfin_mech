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
import meshio

import dolfin_mech as dmech

################################################################################

def RivlinCube_Hyperelasticity(
        dim=3,
        inverse=0,
        incomp=0,
        multimaterial=0,
        cube_params={},
        mat_params={},
        step_params={},
        load_params={},
        move={},
        initialisation_estimation=[],
        get_results=0,
        const_params={},
        res_basename="RivlinCube_Hyperelasticity",
        estimation_virtual_fields=0,
        verbose=0):

    ################################################################### Mesh ###

    refine=True
    u_from_field = False
    boundary_conditions = []
    if u_from_field:
        ### read displacement field form data
        # for i in range(1, 11):
        #     number = str(i).zfill(2)
        mesh_meshio = meshio.read("/Users/peyrault/Seafile/PhD_Alice/Articles/Article_identification_methods/DolfinWarp-Alice/generate_images/square-compx-h=0.1_020.vtu")
        u_meshio = mesh_meshio.point_data["U"]
        u_meshio = u_meshio.tolist()
        u_meshio = [item for sublist in u_meshio for item in sublist[:2]]
        # cube_params = {"X0":0.2, "Y0":0.2, "X1":0.8, "Y1":0.8, "l":0.1}

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params, refine=refine)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params, refine=refine)


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
        displacement_degree=displacement_degree, # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
        quadrature_degree=quadrature_degree,
        w_incompressibility=w_incompressibility,
        elastic_behavior=elastic_behavior,
        elastic_behaviors=elastic_behaviors)

    ########################################## Boundary conditions & Loading ###

    load_type = load_params.get("type", "disp")

    # if estimation_virtual_fields:
    #     problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=[0.,0.,0.])
    # elif ("inertia" not in load_type):
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
    #     if (dim==3):
    #         problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)

    const_type = const_params.get("type", "blox")

    const_type = "blox"

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

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)

    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min)
    
    surface_forces = []
    volume_forces = []


    if (load_type == "disp"):
        u = load_params.get("u", 0.5)
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(0),
            sub_domains=boundaries_mf,
            sub_domain_id=xmax_id,
            val_ini=0., val_fin=u,
            k_step=k_step)
    elif (load_type == "volu0"):
        f = load_params.get("f", 0.5)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
    elif (load_type == "volu"):
        f = load_params.get("f", 0.5)
        problem.add_volume_force_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
    elif (load_type == "surf0"):
        f = load_params.get("f", 1.)
        problem.add_surface_force0_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
    elif (load_type == "surf"):
        f = load_params.get("f", 1.0)
        problem.add_surface_force_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.]*dim, F_fin=[f]+[0.]*(dim-1),
            k_step=k_step)
        surface_forces.append([f, problem.dS(xmax_id)])
    elif (load_type == "pres0"):
        p = load_params.get("p", -0.5)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
    elif (load_type == "pres0_multi"):
        p = load_params.get("p", -0.5)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
    elif (load_type == "pres0_inertia"):
        p = load_params.get("p", -0.5)
        problem.add_inertia_operator(
            measure=problem.dV,
            rho_val=1e-2,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmin_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(ymin_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(zmin_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0,P_fin=p,
            k_step=k_step)
    elif (load_type == "pres"):
        p = load_params.get("p", -0.5)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0, P_fin=p,
            k_step=k_step)
        surface_forces.append([p,problem.dS(xmax_id)])
    elif (load_type == "pgra0"):
        X0 = load_params.get("X0", [0.5]*dim)
        N0 = load_params.get("N0", [1.]+[0.]*(dim-1))
        P0 = load_params.get("P0", -0.5)
        DP = load_params.get("DP", -0.25)
        problem.add_surface_pressure_gradient0_loading_operator(
            measure=problem.dS(),
            X0_val=X0,
            N0_val=N0,
            P0_ini=0., P0_fin=P0,
            DP_ini=0., DP_fin=DP,
            k_step=k_step)
    elif (load_type == "pgra"):
        X0 = load_params.get("X0", [0.5]*dim)
        N0 = load_params.get("N0", [1.]+[0.]*(dim-1))
        P0 = load_params.get("P0", -0.5)
        DP = load_params.get("DP", -0.25)
        problem.add_surface_pressure_gradient_loading_operator(
            measure=problem.dS(),
            X0_val=X0,
            N0_val=N0,
            P0_ini=0., P0_fin=P0,
            DP_ini=0., DP_fin=DP,
            k_step=k_step)
    elif (load_type == "tens"):
        gamma = load_params.get("gamma", 0.01)
        problem.add_surface_tension_loading_operator(
            measure=problem.dS,
            gamma_ini=0., gamma_fin=gamma,
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
        write_sol=res_basename*verbose)

    success = integrator.integrate()
    assert (success),\
        "Integration failed. Aborting."

    integrator.close()

    # print("computing cost function")

    kinematics = dmech.Kinematics(U=problem.get_displacement_subsol().func)

    parameters = {'E': 1, 'nu':0.3}

    material   = dmech.material_factory(kinematics, mat_params["model"], parameters) #mat_params["parameters"])
    sigma= material.sigma 
    N = problem.mesh_normals
    nf = dolfin.dot(N, dolfin.inv(kinematics.F))
    nf_norm = dolfin.sqrt(dolfin.inner(nf,nf))
    sigma_t = dolfin.dot(sigma, nf/nf_norm) + dolfin.Constant(0.3)*nf/nf_norm 
    Sref = problem.dS(xmax_id)

    norm_sigma_t = (((1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*kinematics.J*nf_norm*Sref)/dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) ))**(1/2))**2

    # print("norm_sigma_t=", norm_sigma_t)

    estimation_gap = False
    if estimation_gap:
        print("in estimation gap")

        fe_u = dolfin.VectorElement(
                family="CG",
                cell=mesh.ufl_cell(),
                degree=1)
            
        U_fs = dolfin.FunctionSpace(mesh, fe_u)
        U=dolfin.Function(U_fs)


        U.vector().set_local(u_meshio)

        kinematics = dmech.Kinematics(U=U)
        
        # kinematics = dmech.LinearizedKinematics(u=problem.get_subsols_func_lst()[0], u_old=None)
        # print("surface force is", surface_forces)
        # dmech.EquilibriumGap(problem=problem, kinematics=kinematics, material_model=elastic_behavior["model"], material_parameters=elastic_behavior["parameters"], initialisation_estimation=initialisation_estimation, surface_forces=surface_forces, volume_forces=volume_forces, boundary_conditions=boundary_conditions, inverse=1, U=problem.get_displacement_subsol().func)
        dmech.EquilibriumGap(problem=problem, kinematics=kinematics, material_model=elastic_behavior["model"], material_parameters=elastic_behavior["parameters"], initialisation_estimation=initialisation_estimation, surface_forces=surface_forces, volume_forces=volume_forces, boundary_conditions=boundary_conditions, inverse=0, U=U)
        

    if estimation_virtual_fields:
        if inverse:
            kinematics = dmech.InverseKinematics(u=problem.get_displacement_subsol().func)
        else:
            kinematics = dmech.Kinematics(U=problem.get_displacement_subsol().func)

        dmech.VirtualFields(problem=problem, kinematics=kinematics, material_model=None, material_parameters=mat_params["parameters"], inverse=inverse, U=problem.get_displacement_subsol().func)

    return(problem.get_displacement_subsol().func, problem.dV)

    # if get_results:
    #     return(problem.get_displacement_subsol().func, dolfin.Measure("dx", domain=mesh))
