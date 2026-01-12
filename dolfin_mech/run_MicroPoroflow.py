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
        step_params={},
        load_params={},
        porosity_params={},
        res_basename="run_MicroPoroflow",
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

    # ---------------- Boundary Conditions ------------------- #
    # -------------------- Pressure BCs ----------------------- #
    # tol = 0.1e-6
    # coords = mesh.coordinates()
    # x_min = coords[:, 0].min()
    # x_max = coords[:, 0].max()
    # y_min = coords[:, 1].min()
    # y_max = coords[:, 1].max()
    
    # x_max_surface = dolfin.CompiledSubDomain("near(x[0], x_top, tol)", x_top=x_max, tol=tol)
    # x_min_surface = dolfin.CompiledSubDomain("near(x[0], x_top, tol)", x_top=x_min, tol=tol)
    # y_min_surface = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=y_min, tol=tol)
    # y_max_surface = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=y_max, tol=tol)

    # pressure_space = problem.pl_subsol.fs


    # problem.add_constraint(
    #     V=pressure_space,
    #     sub_domain=x_max_surface,
    #     val=1.0,
    #     k_step=k_step,
    #     method='pointwise'
    # )



    # problem.add_constraint(
    #     V=pressure_space,
    #     sub_domains=boundaries,
    #     sub_domain_id=x_max_surface,   
    #     val_ini=0.0,
    #     val_fin=1.0,
    #     k_step=k_step
    # )



    

    # -------------------- Quantities of Interest ------------- #
    #problem.add_point_displacement_qoi(name="U", coordinates=[X0+R, Y0], component=0)
    #problem.add_qoi(name="U_field", expr=problem.get_displacement_subsol().subfunc)

    #p = problem.get_subsol("pressure").subfunc
    ##problem.add_qoi(name="pressure", expr=p)
    #problem.add_qoi(name="avg_pressure", expr=p * problem.dV)

    #velocity = - rho_l * K_l * dolfin.grad(p)
    #V = dolfin.VectorFunctionSpace(problem.mesh, "CG", 1)
    #problem.add_foi(expr=velocity, fs=V, name="velocity")

    #problem.add_qoi(name="sigma_bulk", expr=problem.get_foi("sigma_bulk"))
    # problem.add_point_displacement_qoi(
    #    name="U",
    #    coordinates=[X0+R, Y0],
    #    component=0)

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
        write_qois=1,#res_basename+"-qois",
        write_sol=1,#res_basename*verbose,
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
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1,
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
                    "macro_grad_p": dolfin.Constant((1.0, 0.0)),
                    "pl_bar": 0.0
                    },
                porosity_params={
                    "type": "constant",  # can be "constant", "function_constant", or "random"
                    "val": 0.3
                },  
                bcs=bcs,
                step_params={"dt_ini":1e-1, "dt_min":1e-3},
                load_params=load_params,
                res_basename=res_folder+"/"+res_basename,
                verbose=1)

            test.test(res_basename)
