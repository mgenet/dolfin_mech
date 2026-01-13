import dolfin
import math
import numpy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))
from fenics import *
import dolfin_mech as dmech
from dolfin_mech.Problem_Hyperelasticity_MicroPoroFlow import MicroPoroFlowHyperelasticityProblem

import myPythonLibrary as mypy



def run_MicroPoroFlowHyperelasticity(
        dim=2,
        bcs="pbc",
        mesh_params={},
        mat_params={},
        flow_params={},
        flow_loading_params={},
        step_params={},
        load_params={},
        porosity_params={},
        res_basename={},
        verbose=1):

    # ------------------------- Mesh ------------------------- #
    # mesh = dolfin.Mesh()
    # with dolfin.XDMFFile("./mesh/mesh.xdmf") as infile:
    #      infile.read(mesh)

    mesh = dmech.run_HollowBox_Mesh(params=mesh_params)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    # print("Reading facet mesh...")
    # with XDMFFile("./mesh/facet_mesh.xdmf") as infile:
    #     # "name_to_read" must match the name used when writing the XDMF
    #     infile.read(mvc, "name_to_read")
    #     print("Facet mesh read.")

    # 3. Convert MeshValueCollection to a MeshFunction for use in Measures
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

            


    # 4. Use in your Variational Problem
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries_mf.set_all(0)
    points_mf = dolfin.MeshFunction("size_t", mesh, 0)
    points_mf.set_all(0)
    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_mf.set_all(0)  # default domain

        # Define tags
    tag_plane=1
    tag_left=2
    tag_right=3
    tag_top=4
    tag_bottom=5
    tag_inclusions=6

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
    problem = MicroPoroFlowHyperelasticityProblem(
        mesh=mesh,
        domains_mf=domains_mf,
        boundaries_mf=boundaries_mf,
        points_mf=points_mf,
        displacement_perturbation_degree=2,
        quadrature_degree = 6,
        bcs="pbc",
        porosity_init_val=poro_val,
        porosity_init_fun=porosity_fun,
        flow_params=flow_params,
        skel_behavior=mat_params["skel"],
        bulk_behavior=mat_params["bulk"],
        pore_behavior=mat_params["pore"])
    
    # dx_in   = self.get_subdomain_measure(inlet_id)          
    # dx_out  = self.get_subdomain_measure(outlet_id)   
    


    # -------------------- Time Step ------------------------- #
    n_steps = step_params.get("n_steps", 1)
    Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.)/n_steps]*n_steps)
    dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.)/n_steps]*n_steps)
    dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.)/n_steps]*n_steps)
    dt_max_lst = step_params.get("dt_max_lst", [step_params.get("dt_max", 1.)/n_steps]*n_steps)
    
    # gamma = load_params.get("gamma", 0.0)

    U_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]
    sigma_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]
    for i in range(dim):
     for j in range (dim):
        U_bar_ij_lst[i][j] = load_params.get("U_bar_"+str(i)+str(j)+"_lst", [load_params.get("U_bar_"+str(i)+str(j), None) for k_step in range(n_steps)])
        sigma_bar_ij_lst[i][j] = load_params.get("sigma_bar_"+str(i)+str(j)+"_lst", [load_params.get("sigma_bar_"+str(i)+str(j), None) for k_step in range(n_steps)])
    pf_lst = load_params.get("pf_lst", [(k_step+1)*load_params.get("pf", 0)/n_steps for k_step in range(n_steps)])
    gamma_lst = load_params.get("gamma_lst", [(k_step+1)*load_params.get("gamma", 0)/n_steps for k_step in range(n_steps)])
    tension_params = load_params.get("tension_params", {})

    # --- flow loading lists ---
    grad_p_bar_lst = flow_loading_params.get("grad_p_bar_lst", [[0.0]*n_steps for _ in range(dim)])
    Theta_in_lst   = flow_loading_params.get("Theta_in_lst",  [0.0]*n_steps)
    Theta_out_lst  = flow_loading_params.get("Theta_out_lst", [0.0]*n_steps)


    # ---- grad_p_bar (x,y) ----
    grad_p_bar_ini = {}
    grad_p_bar_fin = {}
    for k_step in range(n_steps):
        for d in range(dim):
            grad_p_bar     = grad_p_bar_lst[d][k_step]
            grad_p_bar_old = grad_p_bar_lst[d][k_step-1] if (k_step > 0) else 0.0

            grad_p_bar_ini[d] = grad_p_bar_old
            grad_p_bar_fin[d] = grad_p_bar

        # ---- Theta_in / Theta_out ----
        Theta_ini = {}
        Theta_fin = {}

        for name, Theta_lst in {
            "in":  Theta_in_lst,
            "out": Theta_out_lst,
        }.items():
            Theta     = Theta_lst[k_step]
            Theta_old = Theta_lst[k_step-1] if (k_step > 0) else 0.0

            Theta_ini[name] = Theta_old
            Theta_fin[name] = Theta


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
        
        #grad_p_bar_ini = (grad_p_bar_ini[0], grad_p_bar_ini[1])
        grad_p_bar_ini=(0.01,0.01)
        grad_p_bar_fin = (grad_p_bar_fin[0], grad_p_bar_fin[1])

        rho_l = flow_params.get("rho_l", dolfin.Constant(1.0))
        K_l   = flow_params.get("K_l", dolfin.Constant(1.0) * dolfin.Identity(dim))
       
        pl_bar = flow_params.get("pl_bar", dolfin.Constant(0.0))

        problem.add_Darcy_operator(
            kinematics=problem.kinematics,
            U=problem.displacement_perturbation_subsol.subfunc,
            U_test=problem.displacement_perturbation_subsol.dsubtest,
            X=problem.X,
            X_0=problem.X_0,
            unknown_porosity_test=problem.porosity_subsol.dsubtest,
            K_l=K_l,
            rho_l=rho_l,
            pl_bar=pl_bar,
            grad_p_bar_ini=grad_p_bar_ini,
            grad_p_bar_fin=grad_p_bar_fin,
            Theta_in_ini=Theta_ini["in"],   Theta_in_fin=Theta_fin["in"],
            Theta_out_ini=Theta_ini["out"], Theta_out_fin=Theta_fin["out"],
            subdomain_id=None,
            inlet_id=3,
            outlet_id=4,
            k_step=k_step
        )




    # -------------------- Quantities of Interest ------------- #
    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()
    #problem.add_macroscopic_solid_hydrostatic_pressure_qoi()
    problem.add_macroscopic_stress_qois()
    problem.add_fluid_pressure_qoi()
    problem.add_interfacial_surface_qois()

    # Retrieve pressure field (Function)
    p = problem.pl_subsol.subfunc

    # Darcy velocity expression
    #velocity_expr = - problem.rho_l * problem.K_l * dolfin.grad(p)
    velocity_expr = -  dolfin.grad(p)

    # Function space: vector CG space
    velocity_fs = dolfin.VectorFunctionSpace(problem.mesh, "CG", 1)

    # Register as a Field Of Interest
    problem.add_foi(expr=velocity_expr, fs=velocity_fs, name="DarcyVelocity")
    # -------------------- Solver & Integrator ---------------- #
    solver = dmech.NonlinearSolver(
        problem=problem,
        parameters={"sol_tol": [1e-6]*len(problem.subsols), "n_iter_max": 32},
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
        print_out=1,#res_basename*verbose,
        print_sta=1,#res_basename*verbose,
        write_qois=res_basename+"-qois",
        write_sol=res_basename,
        write_vtus=0,
        write_vtus_with_preserved_connectivity=0)

    success = integrator.integrate()
    assert success, "Integration failed. Aborting."

    
    integrator.close()

# ----------------- Run with Options -----------------

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.2,
    "c2":0.4,
    "kappa":1e2,
    "eta":1e-5}


res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=0,
    stop_at_failure=1,
    clean_after_tests=0,
    tester_numpy_tolerance=1e-2)

dim_lst  = [ ]
dim_lst += [2]
# dim_lst += [3]
for dim in dim_lst:

    bcs_lst  = [      ]
    #bcs_lst += ["kubc"]
    bcs_lst += ["pbc" ]
    for bcs in bcs_lst:

        load_lst  = [                     ]
        load_lst += ["internal_pressure"  ]
        #load_lst += ["macroscopic_stretch"]
        #load_lst += ["macroscopic_stress" ]
        for load in load_lst:

            print("dim =",dim)
            print("bcs =",bcs)
            print("load =",load)
            print(res_folder)

            #res_basename  = sys.argv[0][:-3]
            res_basename = "-dim="+str(dim)
            res_basename += "-bcs="+str(bcs)
            res_basename += "-load="+str(load)

            load_params = {}

            grad_p_bar_x_lst = [
                1.0,   # step 0
            ]

            grad_p_bar_y_lst = [
                1.0,
            ]
            Theta_in_lst = [
                0.0,
            ]   
            Theta_out_lst = [
                0.0,
            ]

            flow_loading_params = {
                # 2D: d=0 -> x, d=1 -> y
                "grad_p_bar_lst": [
                    grad_p_bar_x_lst,
                    grad_p_bar_y_lst
                ],
                "Theta_in_lst":  Theta_in_lst,
                "Theta_out_lst": Theta_out_lst,
            }
            if (load == "internal_pressure"):
                load_params["pf"] = +0.2
                for i in range(dim):
                 for j in range (dim):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
            elif (load == "macroscopic_stretch"):
                load_params["pf"] = 0.
                load_params["U_bar_00"] = 0.5
                for i in range(dim):
                 for j in range (dim):
                  if ((i != 0) or (j != 0)):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
            elif (load == "macroscopic_stress"):
                load_params["pf"] = 0.
                for i in range(dim):
                 for j in range (dim):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
                load_params["sigma_bar_00"] = 0.5
                

            run_MicroPoroFlowHyperelasticity(
                dim=dim,
                mesh_params={"dim":dim, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":-0.3, "yshift":-0.3, "zshift":-0.3, "r0":0.2, "l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
                mat_params={
                        "skel": {"parameters": mat_params, "scaling": "no"},
                        "bulk": {"parameters": mat_params, "scaling": "no"},
                        "pore": {"parameters": mat_params, "scaling": "no"}
                    },
                flow_params={
                    "rho_l": 1.0,
                    "K_l": dolfin.Constant(((1e-12, 0.0),
                        (0.0, 1e-12))),
                    "pl_bar": 1.0
                    },
                flow_loading_params=flow_loading_params,
                porosity_params={
                    "type": "constant",  # can be "constant", "function_constant", or "random"
                    "val": 0.3
                },  
                
                bcs=bcs,
                step_params={"dt_ini":1e-1, "dt_min":1e-3},
                load_params=load_params,
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)
