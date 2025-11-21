#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# import dolfin
# import numpy
# import sys

# import myPythonLibrary as mypy
# from pathlib import Path
# local_path = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(local_path))
# import dolfin_mech     as dmech

# sys.path.remove(str(local_path))
import dolfin
import math
import numpy
import myPythonLibrary as mypy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))

import dolfin_mech as dmech
from dolfin_mech.Problem_Darcy_MicroPoro import MicroPoroDarcyProblem





def run_HollowBox_MicroPoroFlow(
        dim,
        mesh=None,
        mesh_params=None,
        displacement_perturbation_degree=1,
        quadrature_degree=3,
        mat_params={},
        bcs="pbc",
        step_params={},
        load_params={},
        res_basename="run_HollowBox_MicroPoroHyperelasticity",
        add_p_hydro_and_Sigma_VM_FoI=False,
        write_qois_limited_precision=True,
        verbose=0):
    
        # ------------------------- Mesh ------------------------- #
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile("./mesh/voronoi_2D_RVE.xdmf") as infile:
        infile.read(mesh)

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    points_mf = dolfin.MeshFunction("size_t", mesh, 0)
    try:
        with dolfin.XDMFFile("./mesh/mesh-2D_rectangle_w_voronoi_inclusions.xdmf") as infile:
            infile.read(boundaries_mf, "boundaries")
            infile.read(points_mf, "points")
    except:
        boundaries_mf = None
        points_mf = None

    assert ((mesh is not None) or (mesh_params is not None))
    if (mesh is None):
        mesh = dmech.run_HollowBox_Mesh(
            params=mesh_params)

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
    #     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, z0=z0, r0=r0)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6
    # sint_id = 9

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unsigned int, but more robust wrt architecture and os
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



    ###



    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_mf.set_all(0)  # default domain
    r = 0.05  # adjust as needed

    # inlet_center = [0.13, 0.5]
    # outlet_center = [0.42, 0.0]

    inlet_center = [-0.05, 0.45]
    outlet_center = [0.45, -0.005]

    # Mark cells near inlet
    class InletDomain(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] - inlet_center[0])**2 + (x[1] - inlet_center[1])**2 < r**2

    # Mark cells near outlet
    class OutletDomain(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] - outlet_center[0])**2 + (x[1] - outlet_center[1])**2 < r**2

    inlet_sub = InletDomain()
    outlet_sub = OutletDomain()

    inlet_id = 3
    outlet_id = 4
    inlet_sub.mark(domains_mf, inlet_id)
    outlet_sub.mark(domains_mf, outlet_id)
    import numpy as np  
    arr = domains_mf.array()



    ################################################################ Problem ###

    problem = MicroPoroDarcyProblem(
        mesh=mesh,
        domains_mf = domains_mf,
        mesh_bbox=bbox,
        vertices=vertices,
        boundaries_mf=boundaries_mf,
        displacement_perturbation_degree=displacement_perturbation_degree,
        quadrature_degree=quadrature_degree,
        solid_behavior=mat_params,
        bcs=bcs)
    


    ################################################################ Loading ###

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

        problem.add_Darcy_operator(kinematics=problem.kinematics,
                K_l=dolfin.Constant(1.0) * dolfin.Identity(2),
                rho_l=dolfin.Constant(1),
                Theta_in=dolfin.Constant(0.0),
                Theta_out=dolfin.Constant(0.0),
                subdomain_id=None,    # where grad(p)·grad(p) is integrated
                inlet_id=3,
                outlet_id=4,
                k_step=k_step)

        
        ######################################################## constaints ###
        x_min = mesh.coordinates()[:, 0].min()
        x_max = mesh.coordinates()[:, 0].max()
        y_max = mesh.coordinates()[:, 1].max()
        y_min = mesh.coordinates()[:, 1].min()
        #print(x_min,y_max)
        

        top_line = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=ymax, tol=tol)
        bot_line = dolfin.CompiledSubDomain("near(x[1], y_bot, tol)", y_bot=ymin, tol=tol)
        pressure_space = problem.get_subsol_function_space("pressure")
        tol = 0.001
        p0_point = dolfin.CompiledSubDomain(
            "near(x[0], x_min, tol) && near(x[1], y_max, tol)",
            x_min=x_min, y_max=y_max, tol=tol)




        # problem.add_constraint(
        #     V=pressure_space,
        #     sub_domain=p0_point,
        #     val=0.0)


        # problem.add_constraint(
        # V=pressure_space,
        # sub_domain=top_line,
        # val=1.0)


        # problem.add_constraint(
        # V=pressure_space,
        # sub_domain=bot_line,
        # val=-1.0)

        





################################################################ Fields of Interest ###
    # Darcy velocity expression
    #velocity_expr = - problem.rho_l * problem.K_l * dolfin.grad(p)
    # p = problem.get_subsol("p_tot").subfunc
    # velocity_expr = -  dolfin.grad(p)

    # # Function space: vector CG space
    # velocity_fs = dolfin.VectorFunctionSpace(problem.mesh, "CG", 1)

    # # Register as a Field Of Interest
    # problem.add_foi(expr=velocity_expr, fs=velocity_fs, name="DarcyVelocity")


    #v_macro, K_macro_col = problem.compute_macro_flow()






    ################################################# Quantities of Interest ###

    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()

    #problem.add_macroscopic_solid_hydrostatic_pressure_qoi()
    problem.add_fluid_pressure_qoi()
    problem.add_macroscopic_stress_qois()
    problem.add_interfacial_surface_qois()

    if (add_p_hydro_and_Sigma_VM_FoI):
        for operator in problem.operators:
            if hasattr(operator, "material"):
                material = operator.material
                break
        problem.add_foi(expr=material.p_hydro, fs=problem.sfoi_fs, name="p_hydro")
        problem.add_foi(expr=material.Sigma_VM, fs=problem.sfoi_fs, name="Sigma_VM")


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
        write_qois_limited_precision=write_qois_limited_precision,
        write_sol=res_basename*verbose)

    success = integrator.integrate()
    assert (success),\
        "Integration failed. Aborting."

    integrator.close()

    return problem


####################################################################### test ###

# res_folder = sys.argv[0][:-3]
# test = mypy.Test(
#     res_folder=res_folder,
#     perform_tests=0,
#     stop_at_failure=1,
#     clean_after_tests=0,
#     tester_numpy_tolerance=1e-2)

# dim_lst  = [ ]
# dim_lst += [2]
# # dim_lst += [3]
# for dim in dim_lst:

#     bcs_lst  = [      ]
#     #bcs_lst += ["kubc"]
#     bcs_lst += ["pbc" ]
#     for bcs in bcs_lst:

#         load_lst  = [                     ]
#         load_lst += ["internal_pressure"  ]
#         load_lst += ["macroscopic_stretch"]
#         load_lst += ["macroscopic_stress" ]
        
#         for load in load_lst:

#             print("dim =",dim)
#             print("bcs =",bcs)
#             print("load =",load)

#             #res_basename  = sys.argv[0][:-3]
#             res_basename = "-dim="+str(dim)
#             res_basename += "-bcs="+str(bcs)
#             res_basename += "-load="+str(load)

#             load_params = {}
#             if (load == "internal_pressure"):
#                 load_params["pf"] = +0.2
#                 for i in range(dim):
#                  for j in range (dim):
#                     load_params["sigma_bar_"+str(i)+str(j)] = 0.
#             elif (load == "macroscopic_stretch"):
#                 load_params["pf"] = 0.
#                 load_params["U_bar_00"] = 0.5
#                 for i in range(dim):
#                  for j in range (dim):
#                   if ((i != 0) or (j != 0)):
#                     load_params["sigma_bar_"+str(i)+str(j)] = 0.
#             elif (load == "macroscopic_stress"):
#                 load_params["pf"] = 0.
#                 for i in range(dim):
#                  for j in range (dim):
#                     load_params["sigma_bar_"+str(i)+str(j)] = 0.
#                 load_params["sigma_bar_00"] = 0.5

            

#             mat_params = {
#                 "alpha":0.16,
#                 "gamma":0.5,
#                 "c1":0.2,
#                 "c2":0.4,
#                 "kappa":1e2,
#                 "eta":1e-5}


#             run_HollowBox_MicroPoroFlow(
#                 dim=dim,
#                 mesh_params={"dim":dim, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":-0.3, "yshift":-0.3, "zshift":-0.3, "r0":0.2, "l":0.04, "mesh_filebasename":res_folder+"/"+"mesh"},
#                 mat_params=mat_params,
#                 bcs=bcs,
#                 step_params={"dt_ini":1e-1, "dt_min":1e-3},
#                 load_params=load_params,
#                 res_basename=res_folder+"/"+res_basename,
#                 #res_basename=res_basename,
#                 verbose=1)

#             test.test(res_basename)

###############################################################################
## TEST CODE FINISHED
###############################################################################