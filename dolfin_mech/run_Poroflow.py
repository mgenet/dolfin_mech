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
from dolfin_mech.Problem_Hyperelasticity_PoroFlow import PoroFlowHyperelasticityProblem



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
    with dolfin.XDMFFile("./mesh/mesh_refine.xdmf") as infile:
         infile.read(mesh)

    #mesh = dmech.run_HollowBox_Mesh(params=mesh_params)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    print("Reading facet mesh...")
    with XDMFFile("./mesh/facet_mesh_refine.xdmf") as infile:
        # "name_to_read" must match the name used when writing the XDMF
        infile.read(mvc, "name_to_read")
        print("Facet mesh read.")

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
        pore_behavior=mat_params["pore"])
    
    # dx_in   = self.get_subdomain_measure(inlet_id)          
    # dx_out  = self.get_subdomain_measure(outlet_id)   
    
    # -------------------- Time Step ------------------------- #
    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 0.1)
    dt_min = step_params.get("dt_min", 0.1)

    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min)

    # ---------------- Boundary Conditions ------------------- #
    # -------------------- Pressure BCs ----------------------- #
    tol = 0.1e-6
    coords = mesh.coordinates()
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    
    x_max_surface = dolfin.CompiledSubDomain("near(x[0], x_top, tol)", x_top=x_max, tol=tol)
    x_min_surface = dolfin.CompiledSubDomain("near(x[0], x_top, tol)", x_top=x_min, tol=tol)
    y_min_surface = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=y_min, tol=tol)
    y_max_surface = dolfin.CompiledSubDomain("near(x[1], y_top, tol)", y_top=y_max, tol=tol)

    pressure_space = problem.pl_subsol.fs


    problem.add_constraint(
        V=pressure_space,
        sub_domains=boundaries,
        sub_domain_id=tag_left,   
        val=0.0,
        k_step=k_step
    )

    problem.add_constraint(
        V=pressure_space,
        sub_domains=boundaries,
        sub_domain_id=tag_right,   
        val=0.0,
        k_step=k_step
    )

    problem.add_constraint(
        V=pressure_space,
        sub_domains=boundaries,
        sub_domain_id=tag_inclusions,   
        val_ini=0.0,
        val_fin=1.0,
        k_step=k_step
    )


    problem.add_constraint(
        V=problem.displacement_subsol.fs.sub(0),
        sub_domains=boundaries,
        sub_domain_id=tag_left,   
        val_ini=0.0,
        val_fin=-0.0005,
        k_step=k_step
    )

    problem.add_constraint(
        V=problem.displacement_subsol.fs.sub(0),
        sub_domains=boundaries,
        sub_domain_id=tag_right,   
        val=0.0,
        k_step=k_step
    )

    problem.add_constraint(
        V=problem.displacement_subsol.fs.sub(1),
        sub_domains=boundaries,
        sub_domain_id=tag_bottom,   
        val=0.0,
        k_step=k_step
    )

    problem.add_constraint(
        V=problem.displacement_subsol.fs.sub(1),
        sub_domains=boundaries,
        sub_domain_id=tag_top,   
        val=0.0,
        k_step=k_step
    )

    

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
        write_sol=res_basename,#res_basename*verbose,
        write_vtus=res_basename*verbose,
        write_vtus_with_preserved_connectivity=True)

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



run_PoroDisc_Coupled(
    mat_params={
        "skel": {"parameters": mat_params, "scaling": "no"},
        "bulk": {"parameters": mat_params, "scaling": "no"},
        "pore": {"parameters": mat_params, "scaling": "no"}
    },
    mesh_params = {
        "dim": 2,

        # square domain
        "xmin": 0.0,
        "ymin": 0.0,
        "xmax": 1.0,
        "ymax": 1.0,

        # optional shift (usually 0)
        "xshift": 0.0,
        "yshift": 0.0,

        # hole radius at corners
        "r0": 0.1,

        # target mesh size
        "l": 0.05,

        # output
        "mesh_filebasename": "results/mesh"
    },

    step_params={
        "Deltat": 1.0,
        "dt_ini": 0.1,
        "dt_min": 0.0001
    },
    load_params={
        "dR": 0.05
    },
    porosity_params={
        "type": "constant",  # can be "constant", "function_constant", or "random"
        "val": 0.3
    },
    res_basename="results/run_PoroBox",
    verbose=0
    
)