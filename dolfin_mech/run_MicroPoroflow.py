import math
import numpy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))
import dolfin_mech as dmech
from fenics import *
import dolfin
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
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile("/Users/xiao/PhD/dolfin_mech_HX2/mesh/voronoi_2D_batch_circle/mesh_phi0p0091_RVE.xdmf") as infile:
         infile.read(mesh)

    #mesh = dmech.run_HollowBox_Mesh(params=mesh_params)

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries_mf.set_all(0)
    points_mf = dolfin.MeshFunction("size_t", mesh, 0)
    points_mf.set_all(0)
    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_mf.set_all(0)  # default domain


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

    pl_bar_ini_lst = flow_loading_params.get("pl_bar_ini_lst", [0.0]*n_steps)
    pl_bar_fin_lst = flow_loading_params.get("pl_bar_fin_lst", [0.0]*n_steps)

    grad_p_bar_x_ini_lst = flow_loading_params.get("grad_p_bar_x_ini_lst", [0.0]*n_steps)
    grad_p_bar_x_fin_lst = flow_loading_params.get("grad_p_bar_x_fin_lst", [0.0]*n_steps)

    grad_p_bar_y_ini_lst = flow_loading_params.get("grad_p_bar_y_ini_lst", [0.0]*n_steps)
    grad_p_bar_y_fin_lst = flow_loading_params.get("grad_p_bar_y_fin_lst", [0.0]*n_steps)

    Theta_in_ini_lst  = flow_loading_params.get("Theta_in_ini_lst",  [0.0]*n_steps)
    Theta_in_fin_lst  = flow_loading_params.get("Theta_in_fin_lst",  [0.0]*n_steps)
    Theta_out_ini_lst = flow_loading_params.get("Theta_out_ini_lst", [0.0]*n_steps)
    Theta_out_fin_lst = flow_loading_params.get("Theta_out_fin_lst", [0.0]*n_steps)


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
            k_l0=k_l,   # rename: baseline intrinsic permeability tensor
            use_kozeny_carman=flow_params.get("use_kozeny_carman", False),  # whether to use Kozeny-Carman relative permeability (True) or a constant factor (False)

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
    "kappa":1,
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
Ex_values = [0.0, 0.1, 0.2]
pf_values = [0.0, 0.03,0.06]
#Ex_values = [0.0, 0.1, 0.2] 
#pf_values = [0]
p_bar_lst = [0.0, 0.1, 0.1]#[0.0, 0.1, 0.2]

pl_bar_ini_lst = [0.0, 0.0]
pl_bar_fin_lst = [0.0, 0.0]

grad_p_bar_x_ini_lst = [0.0, 0.001]
grad_p_bar_x_fin_lst = [0.001, 0.001]

grad_p_bar_y_ini_lst = [0.0, 0.001]
grad_p_bar_y_fin_lst = [0.001, 0.001]

Theta_in_lst = [0.0,0]   
Theta_out_lst = [0.0,0]


flow_loading_params = {
    # pressure (scalar)
    "pl_bar_ini_lst": pl_bar_ini_lst,
    "pl_bar_fin_lst": pl_bar_fin_lst,

    # pressure gradient (2D)
    "grad_p_bar_x_ini_lst": grad_p_bar_x_ini_lst,
    "grad_p_bar_x_fin_lst": grad_p_bar_x_fin_lst,
    "grad_p_bar_y_ini_lst": grad_p_bar_y_ini_lst,
    "grad_p_bar_y_fin_lst": grad_p_bar_y_fin_lst,

    # Theta (keep simple: scalar ini/fin per step)
    "Theta_in_ini_lst":  [0.0, 0.0],  
    "Theta_in_fin_lst":  Theta_in_lst, 
    "Theta_out_ini_lst": [0.0, 0.0],
    "Theta_out_fin_lst": Theta_out_lst,
}
for dim in dim_lst:

    bcs_lst  = [      ]
    #bcs_lst += ["kubc"]
    bcs_lst += ["pbc" ]
    for bcs in bcs_lst:

        load_lst  = [                     ]
        #load_lst += ["K_vs_U"]
        load_lst += ["K_vs_pf"]
        for load in load_lst:

            if load == "K_vs_U":
                load_params = {}
                def set_sigma_bar_all_zero(except00=False):
                    for i in range(dim):
                        for j in range(dim):
                            if except00 and (i == 0 and j == 0):
                                continue
                            load_params[f"sigma_bar_{i}{j}"] = 0.0

                for pf in pf_values:

                    print("dim =",dim)
                    print("bcs =",bcs)
                    print("load =",load)
                    print("pf   =",pf)


                    #res_basename  = sys.argv[0][:-3]
                    res_basename = "-dim="+str(dim)
                    res_basename += "-bcs="+str(bcs)
                    res_basename += "-load="+str(load)
                    res_basename += "-pf="+str(pf)

                    load_params["pf_lst"] = [pf,pf]

                    load_params["U_bar_00_lst"] = [0,0.3]
                    for i in range(dim):
                        for j in range(dim):
                            if ((i != 0) or (j != 0)):
                                load_params["sigma_bar_"+str(i)+str(j)] = 0.

                    # load_params["U_bar_00_lst"] = [0, 0.3]
                    # load_params["U_bar_11_lst"] = [0, 0.3]

                    # for i in range(dim):
                    #     for j in range(dim):
                    #         if (i, j) not in [(0, 0), (1, 1)]:
                    #             load_params["sigma_bar_"+str(i)+str(j)] = 0.

                    # for i in range(dim):
                    #     for j in range(dim):
                    #         load_params[f"U_bar_{i}{j}_lst"] = [0.0, 0.0]


                    # for i in range(dim):
                    #     for j in range(dim):
                    #         if ((i != 0) or (j != 0)):
                    #             load_params["sigma_bar_"+str(i)+str(j)] = 0.

                    run_MicroPoroFlowHyperelasticity(
                        dim=dim,
                        mesh_params={"dim":dim, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":-0.5, "yshift":-0.5, "zshift":-0.5, "r0":0.2, "l":0.05, "mesh_filebasename":res_folder+"/"+"mesh"},
                        mat_params={
                                "skel": {"parameters": mat_params, "scaling": "no"},
                                "bulk": {"parameters": mat_params, "scaling": "no"},
                                "pore": {"parameters": mat_params, "scaling": "no"}
                            },
                        flow_params={ 
                            "k_l": dolfin.Constant(((1e-6, 0.0),
                                (0.0, 1e-6))),
                            "use_kozeny_carman": False
                            },
                        flow_loading_params=flow_loading_params,
                        porosity_params={
                            "type": "constant",  # can be "constant", "function_constant", or "random"
                            "val": 0.3
                        },  
                        
                        bcs=bcs,
                        step_params = {
                            "n_steps": 2,
                            "Deltat_lst": [1e-2, 1e-1],     
                            "dt_ini_lst": [2e-3, 1e-3],     
                            "dt_min_lst": [2e-3, 1e-4],     
                            "dt_max_lst": [2e-3, 5e-3],     
                        },
                        load_params=load_params,
                        res_basename=res_folder+"/"+res_basename,
                        verbose=0)

                    test.test(res_basename)


            if load == "K_vs_pf":
                for Ex in Ex_values:

                    load_params = {}
                    load_params["U_bar_00_lst"] = [Ex, Ex]      
                    pf_target = 0.2
                    load_params["pf_lst"] = [0.0, pf_target]     

                    for i in range(dim):
                        for j in range(dim):
                            if (i, j) != (0, 0):
                                load_params[f"sigma_bar_{i}{j}"] = 0.0

                    res_basename  = f"-dim={dim}-bcs={bcs}-load=K_vs_pf-Ex={Ex}"

                    run_MicroPoroFlowHyperelasticity(
                        dim=dim,
                        mesh_params={"dim":dim, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":-0.5, "yshift":-0.5, "zshift":-0.5, "r0":0.2, "l":0.05, "mesh_filebasename":res_folder+"/"+"mesh"},
                        mat_params={
                                "skel": {"parameters": mat_params, "scaling": "no"},
                                "bulk": {"parameters": mat_params, "scaling": "no"},
                                "pore": {"parameters": mat_params, "scaling": "no"}
                            },
                        flow_params={ 
                            "k_l": dolfin.Constant(((1e-6, 0.0),
                                (0.0, 1e-6))),
                            "use_kozeny_carman": False,
                            },
                        flow_loading_params=flow_loading_params,
                        porosity_params={
                            "type": "constant",  # can be "constant", "function_constant", or "random"
                            "val": 0.3
                        },  
                        
                        bcs=bcs,
                        step_params = {
                            "n_steps": 2,
                            "Deltat_lst": [1e-2, 1e-1],     
                            "dt_ini_lst": [5e-3, 1e-3],     
                            "dt_min_lst": [5e-3, 1e-4],     
                            "dt_max_lst": [5e-3, 5e-3],     
                        },
                        load_params=load_params,
                        res_basename=res_folder+"/"+res_basename,
                        verbose=0)
                    test.test(res_basename)

                
