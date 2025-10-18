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
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile("./mesh/mesh-2D_rectangle_w_voronoi_inclusions.xdmf") as infile:
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


    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_mf.set_all(0)  # default domain

    # Define radius around each "hole"
    r = 0.02  # adjust as needed

    inlet_center = [0.13, 0.5]
    outlet_center = [0.42, 0.0]

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

    tol = 1e-5

    # ---------------------- Problem ------------------------- #
    problem = PoroFlowHyperelasticityProblem(
        mesh=mesh,
        define_facet_normals=True,
        domains_mf = domains_mf,
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


    Deltat = step_params.get("Deltat", 0.1)
    dt_ini = step_params.get("dt_ini", 0.1)
    dt_min = step_params.get("dt_min", 1e-4)

    k_step = problem.add_step(Deltat=Deltat, dt_ini=dt_ini, dt_min=dt_min)
   


    # -------------------- Pressure BCs ----------------------- #




   



    problem.add_Darcy_operator(
    K_l=dolfin.Constant(1),
    rho_l=dolfin.Constant(1),
    Theta_in=dolfin.Constant(1000.0),
    Theta_out=dolfin.Constant(1000.0),
    subdomain_id=None,    # where grad(p)·grad(p) is integrated
    inlet_id=inlet_id,
    outlet_id=outlet_id,
    k_step=k_step
)

    # facet_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # facet_markers.set_all(0)

    # inlet.mark(facet_markers, 1)
    # outlet.mark(facet_markers, 2)

    # print("Facet marker values:", set(facet_markers.array()))
    # dolfin.File("results/facet_markers.pvd") << facet_markers




    coords = mesh.coordinates()
    x_min = coords[:, 0].min()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()

    bottom_left = dolfin.CompiledSubDomain(
    "near(x[1], y0, tol)",
    y0=y_min, tol=tol
    )

    problem.add_constraint(
    V=problem.get_displacement_function_space(),
    sub_domain=bottom_left,
    val=[0.0, 0.0],  # fully fixed in x and y
    method="pointwise"
    )






    top_line = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=y_max, tol=tol)
    bot_line = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=y_min, tol=tol)

    problem.add_constraint(
    V=problem.get_displacement_function_space().sub(1),  # x-component only
    sub_domain=top_line,
    val_ini=0.0,
    val_fin=0.1,
    k_step=k_step,
    method="pointwise"
    )
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    pressure = problem.get_subsol("pressure").func


    x_min = mesh.coordinates()[:, 0].min()
    x_max = mesh.coordinates()[:, 0].max()

 

    inlet = dolfin.CompiledSubDomain("near(x[0], x_min, tol)", x_min=x_min, tol=tol)
    outlet = dolfin.CompiledSubDomain("near(x[0], x_max, tol)", x_max=x_max, tol=tol)

    pressure_space = problem.get_subsol_function_space("pressure")

    problem.add_constraint(
        V=pressure_space,
        sub_domain=top_line,
        val=1.0,  # or any inlet pressure
    
    )

    problem.add_constraint(
        V=pressure_space,
        sub_domain=bot_line,
        val=0.0,   # or any outlet pressure
     
    )

    # problem.add_constraint(
    #     V=pressure_space,
    #     sub_domain=inlet,
    #     val=0.0,  # or any inlet pressure
    
    # )

    # problem.add_constraint(
    #     V=pressure_space,
    #     sub_domain=outlet,
    #     val=0.0,  # or any inlet pressure
    
    # )
    
    
#     bbox = [x_min, x_max, y_min, y_max]  # Use actual values

#     bbox = [
#     mesh.coordinates()[:, 0].min(),  # xmin
#     mesh.coordinates()[:, 0].max(),  # xmax
#     mesh.coordinates()[:, 1].min(),  # ymin
#     mesh.coordinates()[:, 1].max()   # ymax
# ]
#     pbc = dmech.PeriodicSubDomain(dim=2, bbox=bbox)
#     V_pressure = dolfin.FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)

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