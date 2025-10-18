import dolfin
import math
import numpy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))

import dolfin_mech as dmech
from dolfin_mech.Problem_Hyperelasticity_PoroFlow import PoroFlowHyperelasticityProblem

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.2,
    "c2":0.4,
    "kappa":1e2,
    "eta":1e-5}



def run_PoroDisc_Coupled(
        mesh_params={},
        mat_params={},
        step_params={},
        load_params={},
        porosity_params={},
        res_basename="run_PoroDisc_Coupled",
        verbose=1):

    # ------------------------- Mesh ------------------------- #
    X0 = mesh_params.get("X0", 0.5)
    Y0 = mesh_params.get("Y0", 0.5)
    R  = mesh_params.get("R", 0.3)

    #mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.run_RivlinCube_Mesh(dim=2, params=cube_params)
    mesh, boundaries_mf, S_id, points_mf, x1_sd, x2_sd, x3_sd, x4_sd = dmech.run_Disc_Mesh(params=mesh_params)

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

    # ------------------- Domain Init -------------------------#

    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_mf.set_all(0)  # default domain

    # Define radius around each "hole"
    r = 0.05  # adjust as needed

    inlet_center = [0.5, 0.65]
    outlet_center = [0.5, 0.4]

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
    #inlet_sub = (dolfin.CompiledSubDomain((x[0] - inlet_center[0])**2 + (x[1] - inlet_center[1])**2 < r**2))
    inlet_id = 3
    outlet_id = 4
    inlet_sub.mark(domains_mf, inlet_id)
    outlet_sub.mark(domains_mf, outlet_id)

    # ---------------------- Problem ------------------------- #
    problem = PoroFlowHyperelasticityProblem(
        mesh=mesh,
        define_facet_normals=True,
        domains_mf=domains_mf,
        boundaries_mf=boundaries_mf,
        points_mf=points_mf,
        displacement_degree=1,
        quadrature_degree = 6,
        porosity_init_val=poro_val,
        porosity_init_fun=porosity_fun,
        skel_behavior=mat_params["skel"],
        bulk_behavior=mat_params["bulk"],
        pore_behavior=mat_params["pore"],
        gradient_operators=None)
    

    # ------------------ Darcy + Pressure Constraint --------- #
    #K_l = dolfin.Constant(1e-12)
    #rho_l = dolfin.Constant(1000.0)

    #problem.add_Darcy_operator(K_l=K_l, rho_l=rho_l, Theta_0=dolfin.Constant(0.0))

    # -------------------- Time Step ------------------------- #
    Deltat = step_params.get("Deltat", 0.1)
    dt_ini = step_params.get("dt_ini", 0.1)
    dt_min = step_params.get("dt_min", 1e-4)

    k_step = problem.add_step(Deltat=Deltat, dt_ini=dt_ini, dt_min=dt_min)



 
    pressure_space = problem.get_subsol_function_space("pressure")


    problem.add_Darcy_operator(
    K_l=dolfin.Constant(1),
    rho_l=dolfin.Constant(1.0),
    Theta_in=dolfin.Constant(1000.0),
    Theta_out=dolfin.Constant(1000.0),
    subdomain_id=None,    # where grad(p)·grad(p) is integrated
    inlet_id=inlet_id,
    outlet_id=outlet_id,
    k_step=k_step
)
    

    # ---------------- Boundary Conditions ------------------- #
    #problem.add_pf_operator(measure=problem.dV, pf_ini=0, pf_fin=1, k_step=k_step)
    # problem.add_constraint(
    #     V=problem.get_displacement_function_space(),
    #     sub_domain=dolfin.CompiledSubDomain("near(x[0], 0.5) && near(x[1], 0.5)"),
    #     val=[0.0, 0.0],
    #     method="pointwise")

    # dR = load_params.get("dR", 0.1)
    # surface_nodes_coords = [  
    #     X for X in mesh.coordinates()
    #     if dolfin.near((X[0]-X0)**2 + (X[1]-Y0)**2, R**2, eps=1e-3)
    # ]
    # for X in surface_nodes_coords:
    #     X_inplane = numpy.array(X) - numpy.array([X0, Y0])
    #     R_mag = numpy.linalg.norm(X_inplane)
    #     T = math.atan2(X_inplane[1], X_inplane[0])
    #     r = R_mag + dR
    #     x_inplane = numpy.array([r * math.cos(T), r * math.sin(T)])
    #     x = numpy.array([X0, Y0]) + x_inplane
    #     U = x - X
    #     X_sd = dolfin.CompiledSubDomain("near(x[0], x0) && near(x[1], y0)", x0=X[0], y0=X[1])
    #     problem.add_constraint(
    #         V=problem.get_displacement_function_space(),
    #         sub_domain=X_sd,
    #         val_ini=[0.0, 0.0],
    #         val_fin=U,
    #         k_step=k_step,
    #         method="pointwise")

    # -------------------- Pressure BCs ----------------------- #

    tol = 0.1
    # X0 = mesh_params.get("X0", 0.5)
    # Y0 = mesh_params.get("Y0", 0.5)
    # R  = mesh_params.get("R", 0.3)

    # Inlet: at the center (X0, Y0)
    inlet = dolfin.CompiledSubDomain("on_boundary && x[0] < X0 - tol", X0=X0, tol=tol)

    # Outlet: on the outer edge of the disc (distance = R)
    outlet = dolfin.CompiledSubDomain("on_boundary && x[0] > X0 + tol", X0=X0, tol=tol)


    

    # pressure_space = problem.get_subsol_function_space("pressure")

    # print(pressure_space)

    # problem.add_constraint(
    # V=pressure_space,
    # sub_domain=inlet,
    # val_ini=0.0,
    # val_fin=1.0,
    # k_step=k_step,  # ← your current time step
    # method="pointwise")

    # problem.add_constraint(
    # V=pressure_space,
    # sub_domain=outlet,
    # val_ini=0.0,
    # val_fin=-1.0,
    # k_step=k_step,  # ← your current time step
    # method="pointwise"
    # )
    problem.add_constraint(V=pressure_space, sub_domain=inlet,  val=0.0)
    
    problem.add_constraint(V=pressure_space, sub_domain=outlet, val=0.0)


    # facet_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # facet_markers.set_all(0)

    # inlet.mark(facet_markers, 1)
    # outlet.mark(facet_markers, 2)

    # print("Facet marker values:", set(facet_markers.array()))
    # dolfin.File("results/facet_markers.pvd") << facet_markers



    surface_nodes_coords = [
    node_coords for node_coords in mesh.coordinates()
    if dolfin.near((node_coords[0] - X0)**2 + (node_coords[1] - Y0)**2, R**2, eps=1e-3)]

    dR = load_params.get("dR", 0.05 * R)  

    for X in surface_nodes_coords:
        X_inplane = numpy.array(X) - numpy.array([X0, Y0])
        R_mag = numpy.linalg.norm(X_inplane)
        T = math.atan2(X_inplane[1], X_inplane[0])
        r_new = R_mag + dR  # apply radial compression
        x_inplane = numpy.array([r_new * math.cos(T), r_new * math.sin(T)])
        x = numpy.array([X0, Y0]) + x_inplane
        U = x - X  # displacement to be applied

        X_sd = dolfin.CompiledSubDomain(
            "near(x[0], x0) && near(x[1], y0)", x0=X[0], y0=X[1]
        )

        problem.add_constraint(
            V=problem.get_displacement_function_space(),
            sub_domain=X_sd,
            val_ini=[0.0, 0.0],
            val_fin=U,
            k_step=k_step,
            method="pointwise"
        )


    center_sd = dolfin.CompiledSubDomain("near(x[0], X0) && near(x[1], Y0)", X0=X0-R, Y0=Y0)

    problem.add_constraint(
        V=problem.get_displacement_function_space(),
        sub_domain=center_sd,
        val=[0.0, 0.0],
        method="pointwise"
    )

    center_sd2 = dolfin.CompiledSubDomain("near(x[0], X0) && near(x[1], Y0)", X0=X0+R, Y0=Y0)

    problem.add_constraint(
        V=problem.get_displacement_function_space(),
        sub_domain=center_sd2,
        val=[0.0, 0.0],
        method="pointwise"
    )


    #boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    #pressure = problem.get_subsol("pressure").func


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
    p = problem.get_subsol("pressure").subfunc

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
        write_sol=res_basename,#res_basename*verbose,
        write_vtus=res_basename*verbose,
        write_vtus_with_preserved_connectivity=True)

    success = integrator.integrate()
    assert success, "Integration failed. Aborting."

    
    integrator.close()

# ----------------- Run with Options -----------------

run_PoroDisc_Coupled(
    mat_params={
        "skel": {"parameters": mat_params, "scaling": "no"},
        "bulk": {"parameters": mat_params, "scaling": "no"},
        "pore": {"parameters": mat_params, "scaling": "no"}
    },
    mesh_params={
        "X0": 0.5,
        "Y0": 0.5,
        "R": 0.3,
        "l": 0.03,
        "mesh_filebasename": "results/mesh"
    },
    step_params={
        "Deltat": 1.0,
        "dt_ini": 0.2,
        "dt_min": 0.0001
    },
    load_params={
        "dR": 0.05
    },
    porosity_params={
        "type": "constant",  # can be "constant", "function_constant", or "random"
        "val": 0.3
    },
    res_basename="results/run_PoroDisc_Coupled",
    verbose=0
    
)