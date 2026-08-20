################################################################################
###                                                                          ###
### Created by Haotian XIAO, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy
import os
import json

import dolfin_mech as dmech

def run_HollowBox_MicroPoroflow(
        dim=2,
        bcs="pbc",
        mesh_params={},
        mat_params={},
        flow_params={},
        step_params={},
        load_params={},
        porosity_params={},
        res_basename={},
        verbose=1):

    # ------------------------- Mesh ------------------------- #
    mesh, mesh_porosity = dmech.run_HollowBox_Mesh(params=mesh_params,return_porosity=True)

    metadata = {
        "mesh_porosity": float(mesh_porosity),
        "mesh_params": mesh_params,
        "res_basename": str(res_basename),
    }

    metadata_filename = str(res_basename) + "-metadata.json"
    metadata_folder = os.path.dirname(metadata_filename)

    if metadata_folder:
        os.makedirs(metadata_folder, exist_ok=True)

    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)

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
        tol = 1e-8
        assert numpy.linalg.norm(vertices[2,:]-vertices[3,:] - a1) <= tol # check if UC vertices form indeed a parallelogram
        assert numpy.linalg.norm(vertices[2,:]-vertices[1,:] - a2) <= tol # check if UC vertices form indeed a parallelogram
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

    tol = 1e-8
    xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmin, tol=tol)
    xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmax, tol=tol)
    ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymin, tol=tol)
    ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymax, tol=tol)
    if (dim==3): zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmin, tol=tol)
    if (dim==3): zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmax, tol=tol)

    # if (dim==2):
    #     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, r0=r0)
    # elif (dim==3):
    #      sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, z0=z0, r0=r0)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6
    # sint_id = 9

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
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

    points_mf = dolfin.MeshFunction("size_t", mesh, 0)
    points_mf.set_all(0)
    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_mf.set_all(0) 


    # ------------------- Porosity Init ---------------------- #
    poro_type = porosity_params.get("type", "constant")
    poro_val = porosity_params.get("val", 0.5)

    porosity_fun = None
    if poro_type == "function_constant":
        poro_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.Function(poro_fs)
        porosity_fun.vector()[:] = poro_val
        poro_val = None
    elif poro_type == "random":
        poro_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.Function(poro_fs)
        porosity_fun.vector()[:] = numpy.random.uniform(low=0.4, high=0.6, size=porosity_fun.vector().size())
        poro_val = None

    # ---------------------- Problem ------------------------- #
    problem = dmech.MicroPoroFlowHyperelasticityProblem(
        mesh=mesh,
        domains_mf=domains_mf,
        boundaries_mf=boundaries_mf,
        points_mf=points_mf,
        displacement_perturbation_degree=2,
        quadrature_degree = 6,
        bcs=bcs,
        porosity_init_val=poro_val,
        porosity_init_fun=porosity_fun,
        flow_params=flow_params,
        skel_behavior=mat_params["skel"],
        bulk_behavior=mat_params["bulk"],
        pore_behavior=mat_params["pore"])
    
    # -------------------- Time Step ------------------------- #
    n_steps = step_params.get("n_steps", 1)
    Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.)/n_steps]*n_steps)
    dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.)/n_steps]*n_steps)
    dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.)/n_steps]*n_steps)
    dt_max_lst = step_params.get("dt_max_lst", [step_params.get("dt_max", 1.)/n_steps]*n_steps)

    load_params = {} if load_params is None else load_params

    load_params_solid  = load_params.get("solid", {})
    load_params_liquid = load_params.get("liquid", {})
    load_params_air    = load_params.get("air", {})

    # --- solid loading lists ---

    U_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]
    sigma_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]

    for i in range(dim):
        for j in range(dim):
            U_bar_ij_lst[i][j] = load_params_solid.get(
                f"U_bar_{i}{j}_lst",
                [load_params_solid.get(f"U_bar_{i}{j}", None) for k_step in range(n_steps)]
            )
            sigma_bar_ij_lst[i][j] = load_params_solid.get(
                f"sigma_bar_{i}{j}_lst",
                [load_params_solid.get(f"sigma_bar_{i}{j}", None) for k_step in range(n_steps)]
            )

    gamma_lst = load_params_solid.get(
        "gamma_lst",
        [(k_step+1) * load_params_solid.get("gamma", 0.0) / n_steps for k_step in range(n_steps)]
    )

    tension_params = load_params_solid.get("tension_params", {})

    # --- liquid loading lists ---

    pl_bar_ini_lst = load_params_liquid.get("pl_bar_ini_lst", [0.0] * n_steps)
    pl_bar_fin_lst = load_params_liquid.get("pl_bar_fin_lst", [0.0] * n_steps)

    grad_p_bar_x_ini_lst = load_params_liquid.get("grad_p_bar_x_ini_lst", [0.0] * n_steps)
    grad_p_bar_x_fin_lst = load_params_liquid.get("grad_p_bar_x_fin_lst", [0.0] * n_steps)

    grad_p_bar_y_ini_lst = load_params_liquid.get("grad_p_bar_y_ini_lst", [0.0] * n_steps)
    grad_p_bar_y_fin_lst = load_params_liquid.get("grad_p_bar_y_fin_lst", [0.0] * n_steps)

    Theta_in_ini_lst  = load_params_liquid.get("Theta_in_ini_lst",  [0.0] * n_steps)
    Theta_in_fin_lst  = load_params_liquid.get("Theta_in_fin_lst",  [0.0] * n_steps)
    Theta_out_ini_lst = load_params_liquid.get("Theta_out_ini_lst", [0.0] * n_steps)
    Theta_out_fin_lst = load_params_liquid.get("Theta_out_fin_lst", [0.0] * n_steps)

    # --- air loading lists ---

    pf_lst = load_params_air.get(
        "pf_lst",
        [(k_step+1) * load_params_air.get("pf", 0.0) / n_steps for k_step in range(n_steps)]
    )


    for k_step in range(n_steps):

        Deltat = Deltat_lst[k_step]
        dt_ini = dt_ini_lst[k_step]
        dt_min = dt_min_lst[k_step]
        dt_max = dt_max_lst[k_step]

        k_step = problem.add_step(
            Deltat=Deltat,
            dt_ini=dt_ini,
            dt_min=dt_min,
            dt_max=dt_max)

        #   air pressure loading

        pf = pf_lst[k_step]
        pf_old = pf_lst[k_step-1] if (k_step > 0) else 0.

        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(0),
            P_ini=pf_old, P_fin=pf,
            k_step=k_step)

        for i in range(dim):
         for j in range (dim):
            U_bar_ij = U_bar_ij_lst[i][j][k_step]
            U_bar_ij_old = U_bar_ij_lst[i][j][k_step-1] if (k_step > 0) else 0.
            sigma_bar_ij = sigma_bar_ij_lst[i][j][k_step]
            sigma_bar_ij_old = sigma_bar_ij_lst[i][j][k_step-1] if (k_step > 0) else 0.
            assert ((U_bar_ij is not None) or (sigma_bar_ij is not None))
            if (U_bar_ij is not None):
                problem.add_macroscopic_stretch_component_penalty_operator(
                    i=i, j=j,
                    U_bar_ij_ini=U_bar_ij_old, U_bar_ij_fin=U_bar_ij,
                    pen_val=1e6,
                    k_step=k_step)
            elif (sigma_bar_ij is not None):
                problem.add_macroscopic_stress_component_constraint_operator(
                    i=i, j=j,
                    sigma_bar_ij_ini=sigma_bar_ij_old, sigma_bar_ij_fin=sigma_bar_ij,
                    pf_ini=pf_old, pf_fin=pf,
                    k_step=k_step)
        
        problem.add_surface_area_operator(
            measure=problem.dS(0),
            k_step=k_step)
        
        gamma = gamma_lst[k_step]
        gamma_old = gamma_lst[k_step-1] if (k_step > 0) else 0.
        problem.add_surface_tension_loading_operator(
            measure=problem.dS(0),
            gamma_ini=gamma_old, gamma_fin=gamma,
            tension_params=tension_params,
            k_step=k_step)
        
        # ---- flow loadings (pull from your manual arrays using k) ----
        pl_bar_ini = pl_bar_ini_lst[k_step]
        pl_bar_fin = pl_bar_fin_lst[k_step]

        grad_p_bar_ini = (grad_p_bar_x_ini_lst[k_step], grad_p_bar_y_ini_lst[k_step])
        grad_p_bar_fin = (grad_p_bar_x_fin_lst[k_step], grad_p_bar_y_fin_lst[k_step])

        Theta_in_ini  = Theta_in_ini_lst[k_step]
        Theta_in_fin  = Theta_in_fin_lst[k_step]
        Theta_out_ini = Theta_out_ini_lst[k_step]
        Theta_out_fin = Theta_out_fin_lst[k_step]


        k_l   = flow_params.get("k_l", dolfin.Constant(1.0) * dolfin.Identity(dim))


        problem.add_Darcy_operator(
            # --- kinematics / fields ---
            kinematics=problem.kinematics,
            U=problem.displacement_perturbation_subsol.subfunc,
            U_test=problem.displacement_perturbation_subsol.dsubtest,
            X=problem.X,
            X_0=problem.X_0,

            # --- macro loads ---
            grad_p_bar_ini=grad_p_bar_ini,
            grad_p_bar_fin=grad_p_bar_fin,
            pl_bar_ini=pl_bar_ini,
            pl_bar_fin=pl_bar_fin,
            Theta_in_ini=Theta_in_ini,   Theta_in_fin=Theta_in_fin,
            Theta_out_ini=Theta_out_ini, Theta_out_fin=Theta_out_fin,

            # --- material ---
            k_l0=k_l,   
            use_kozeny_carman=flow_params.get("use_kozeny_carman", False),  
            # --- ids ---
            subdomain_id=None,
            inlet_id=None,
            outlet_id=None,

            # --- step ---
            k_step=k_step,
        )

    # -------------------- Quantities of Interest ------------- #
    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()
    problem.add_macroscopic_stress_qois()
    problem.add_fluid_pressure_qoi()
    problem.add_interfacial_surface_qois()
    problem.add_darcy_qois() 

    # -------------------- Solver & Integrator ---------------- #
    solver = dmech.NonlinearSolver(
    problem=problem,
    parameters={
        "sol_tol": [1e-6]*len(problem.subsols),
        "n_iter_max": 32,
        "linear_solver_type": "dolfin",
        "linear_solver_name": "umfpack",
    },
    relax_type="constant",
    write_iter=0)


    integrator = dmech.TimeIntegrator(
        problem=problem,
        solver=solver,
        parameters={
            "n_iter_for_accel": 4,
            "n_iter_for_decel": 16,
            "accel_coeff": 2,
            "decel_coeff": 2},
        print_out=res_basename*verbose,
        print_sta=res_basename*verbose,
        write_qois=res_basename+"-qois",
        write_sol=res_basename,
        write_vtus=0,
        write_vtus_with_preserved_connectivity=0)

    success = integrator.integrate()
    assert success, "Integration failed. Aborting."

    
    integrator.close()
