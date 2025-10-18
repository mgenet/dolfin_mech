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
from dolfin_mech.Problem_Hyperelasticity_MicroPoroFlowV2 import MicroPoroFlowHyperelasticityProblem
mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.2,
    "c2":0.4,
    "kappa":1e2,
    "eta":1e-5}

def _inspect_ops(problem, k, k_step, tag):
    try:
        step_obj = problem.steps[k_step] if isinstance(problem.steps, (list, tuple, dict)) else None
        if step_obj is None:
            print(f"[WARN] step {k}: k_step={k_step} not found in problem.steps")
            return
        
        # 如果是 dict，走 dict 的
        if isinstance(step_obj, dict):
            ops = step_obj.get("operators", [])
        else:
            # 否则当成 Step 对象（有 .operators 属性）
            ops = getattr(step_obj, "operators", [])
        
        print(f"[CHK] step {k} ({tag}): #operators attached = {len(ops)}")
        for idx, op in enumerate(ops):
            print(f"   - op[{idx}] type = {type(op).__name__}")
    except Exception as e:
        print(f"[WARN] step {k}: cannot inspect operators ({e})")

def run_PoroDisc_Coupled(
        mesh_params={},
        mat_params={},
        step_params={},
        load_params={},
        porosity_params={},
        res_basename="run_PoroDisc_Coupled",
        bcs="pbc",
        verbose=1):

    # ------------------------- Mesh ------------------------- #
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile("./mesh/voronoi_2D_thick.xdmf") as infile:
        infile.read(mesh)

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    points_mf = dolfin.MeshFunction("size_t", mesh, 0)

    coords = mesh.coordinates()
    x_max = max(coords[:,0]); x_min = min(coords[:,0])
    y_max = max(coords[:,1]); y_min = min(coords[:,1])

    bbox = [x_min, x_max, y_min, y_max]

    vertices = numpy.array([[x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max]])
    a1 = vertices[1,:]-vertices[0,:] # first vector generating periodicity
    a2 = vertices[3,:]-vertices[0,:] # second vector generating periodicity
    tol = 1e-8
    assert numpy.linalg.norm(vertices[2,:]-vertices[3,:] - a1) <= tol # check if UC vertices form indeed a parallelogram
    assert numpy.linalg.norm(vertices[2,:]-vertices[1,:] - a2) <= tol # check if UC vertices form indeed a parallelogram    
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
    r = 0.2  # adjust as needed

    inlet_center = [0.35, 0.59]
    outlet_center = [0.57, 0.42]

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

    inlet_id = 1
    outlet_id = 2
    inlet_sub.mark(domains_mf, inlet_id)
    outlet_sub.mark(domains_mf, outlet_id)

    tol = 1e-5
    print("[CHK] unique domain tags:", numpy.unique(domains_mf.array()))

    # ---------------------- Problem ------------------------- #
    problem = MicroPoroFlowHyperelasticityProblem(
        mesh=mesh,
        mesh_bbox=bbox,
        vertices=vertices,
        domains_mf=domains_mf,
        boundaries_mf=boundaries_mf,
        displacement_perturbation_degree=1,
        quadrature_degree=6,
        bcs=bcs,        
        porosity_init_val=poro_val,
        porosity_init_fun=porosity_fun,
        skel_behavior=mat_params["skel"],
        bulk_behavior=mat_params["bulk"],
        pore_behavior=mat_params["pore"],
        )


    Deltat = step_params.get("Deltat", 0.1)
    dt_ini = step_params.get("dt_ini", 0.1)
    dt_min = step_params.get("dt_min", 1e-4)

    #k_step = problem.add_step(Deltat=Deltat, dt_ini=dt_ini, dt_min=dt_min)
   


    # --------------------  BCs ----------------------- #

    ####### Loading (BC) ########

    # ------------------- Loading Loop ------------------- #
    n_steps = step_params.get("n_steps", 1)
    Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.0)/n_steps]*n_steps)
    dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.0)/n_steps]*n_steps)
    dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.0)/n_steps]*n_steps)
    dt_max_lst = step_params.get("dt_max_lst", [step_params.get("dt_max", 1.0)/n_steps]*n_steps)

    # 剪切加载参数（U_bar_01）
    U_bar_01_lst = load_params.get("U_bar_01_lst",
                                [(k+1)*load_params.get("U_bar_01", 0.0)/n_steps for k in range(n_steps)])

    # Darcy 驱动参数
    p_in  = load_params.get("p_in", 0.0)
    p_out = load_params.get("p_out", 0.0)
    inlet_id  = load_params.get("inlet_id", 1)
    outlet_id = load_params.get("outlet_id", 2)

    # 从 mesh_bbox 里提取一个角点坐标
    x_min, y_min = problem.mesh_bbox[0], problem.mesh_bbox[2]

    for k in range(n_steps):

        Deltat = Deltat_lst[k]
        dt_ini = dt_ini_lst[k]
        dt_min = dt_min_lst[k]
        dt_max = dt_max_lst[k]

        k_step = problem.add_step(
            Deltat=Deltat,
            dt_ini=dt_ini,
            dt_min=dt_min,
            dt_max=dt_max)

        # ------------------- 剪切加载 (U_bar_01) -------------------
        U_bar_01     = U_bar_01_lst[k]
        U_bar_01_old = U_bar_01_lst[k-1] if (k > 0) else 0.0

        problem.add_macroscopic_stretch_component_penalty_operator(
            i=0, j=1,    # U_bar_01 = γ （剪切）
            U_bar_ij_ini=U_bar_01_old,
            U_bar_ij_fin=U_bar_01,
            pen_val=1e5,   # 建议比 1e6 稍小，先稳定起来
            k_step=k_step)
        
        

        for (i,j) in [(0,0),(1,0),(1,1)]:
            problem.add_macroscopic_stretch_component_penalty_operator(
                i=i, j=j,
                U_bar_ij_ini=0.0,
                U_bar_ij_fin=0.0,
                pen_val=1e5,
                k_step=k_step)
            

        # ------------------- Darcy 流动 (p_in / p_out) -------------------
        # problem.add_Darcy_operator(
        #     K_l=dolfin.Constant(1.0),
        #     rho_l=dolfin.Constant(1.0),
        #     Theta_in=dolfin.Constant(p_in),
        #     Theta_out=dolfin.Constant(p_out),
        #     inlet_id=inlet_id,
        #     outlet_id=outlet_id,
        #     k_step=k_step)
        
        # 取压力空间
        Qp = problem.get_subsol_function_space("pressure")

        # 选一个参考点（例如左下角）
        coords = problem.mesh.coordinates()
        x0, y0 = x_min,y_min

        class PRef(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return dolfin.near(x[0], x0, 1e-4) and dolfin.near(x[1], y0, 1e-4)

        # 施加点约束
        problem.add_constraint(V=Qp, sub_domain=PRef(), val=0.0, method="pointwise")

        V_u = problem.get_displacement_perturbation_function_space()
        problem.add_constraint(
            V=V_u,     # sub(0) → x方向位移
            sub_domain=PRef(),
            val=[0.0,0.0],
            method="pointwise"
        )


        # ---- 表面压力加载（外边界）----
        # 1) 取本步/上一步的目标值（用 k，不要用 k_step）
        pf_lst = load_params.get("pf_lst", [(k_step+1)*load_params.get("pf", 0)/n_steps for k_step in range(n_steps)])
        pf     = pf_lst[k]
        pf_old = pf_lst[k-1] if (k > 0) else 0.0


        # 2) 选择积分 measure
        #   - 有边界标签时：用 problem.ds(tag)；比如 tag=10 是你想压的那条边
        #   - 暂时没标签用于快速验证：整条外边界 problem.ds
        #surf_measure = problem.dS           # 或者 problem.ds(你的边界tag)

        # # 3) 加载算子
        # problem.add_surface_pressure_loading_operator(
        #     measure=surf_measure,
        #     P_ini=pf_old,
        #     P_fin=pf,
        #     k_step=k_step
        # )

 

    # -------------------- Quantities of Interest ------------- #
    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()
    #problem.add_macroscopic_solid_hydrostatic_pressure_qoi()
    #problem.add_macroscopic_stress_qois()
    #problem.add_fluid_pressure_qoi()
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
    },
    step_params={
        "Deltat": 1.0,
        "dt_ini": 0.2,
        "dt_min": 0.0001
    },
    load_params = {
        "n_steps": 2,   # 两步加载
        "pf" : 0,

        # --- Step 1: 剪切加载 ---
        "U_bar_01_lst": [0.0, 0.0],   # γ = 0.05 保持不变

        # --- Step 2: Darcy 流动 ---
        "p_in": 0.0,      # 入口压力
        "p_out": 0.0,     # 出口压力
        "inlet_id": 1,    # 网格 inlet 边界 id
        "outlet_id": 2    # 网格 outlet 边界 id
    },
    porosity_params={
        "type": "constant",  # can be "constant", "function_constant", or "random"
        "val": 0.3
    },
    res_basename="results/run_PoroDisc_Coupled",
    verbose=0
)

