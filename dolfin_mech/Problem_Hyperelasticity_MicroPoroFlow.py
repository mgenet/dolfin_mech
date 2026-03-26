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

import operator
import dolfin
import numpy

import dolfin_mech as dmech
from .Problem                 import Problem
from .Problem_Hyperelasticity_MicroPoro import MicroPoroHyperelasticityProblem,_SigmaAggregatorMaterial
from .Problem_Hyperelasticity_Poro import PoroHyperelasticityProblem

################################################################################
class MicroPoroFlowHyperelasticityProblem(MicroPoroHyperelasticityProblem, PoroHyperelasticityProblem):

    def __init__(self,
            w_solid_incompressibility=False,
            mesh=None,
            mesh_bbox=None,
            vertices=None,
            domains_mf=None,
            boundaries_mf=None,
            points_mf=None,
            displacement_perturbation_degree=None,
            solid_pressure_degree=None,
            porosity_known="Phis0",
            bcs="pbc",
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None,
            quadrature_degree=None,
            foi_degree=1,
            flow_params={},
            skel_behavior=None,
            skel_behaviors=[],
            bulk_behavior=None,
            bulk_behaviors=[],
            pore_behavior=None,
            pore_behaviors=[],
            w_pressure_balancing_gravity=0):

        Problem.__init__(self)

        self.w_solid_incompressibility = w_solid_incompressibility
        self.vertices = vertices


        self.set_mesh(
            mesh=mesh,
            define_spatial_coordinates=1,
            define_facet_normals=1,
            compute_bbox=(mesh_bbox is None))
        self.X_0 = [0.]*self.dim
        for k_dim in range(self.dim):
            self.X_0[k_dim] = dolfin.assemble(self.X[k_dim] * self.dV)/self.mesh_V0
        self.X_0 = dolfin.Constant(self.X_0)
        if (mesh_bbox is not None):
            self.mesh_bbox = mesh_bbox
        d = [0]*self.dim
        for k_dim in range(self.dim):
            d[k_dim] = self.mesh_bbox[2*k_dim+1] - self.mesh_bbox[2*k_dim+0]

        self.V0 = numpy.prod(d) 
        self.Vs0 = self.mesh_V0
        self.Vf0 = self.V0 - self.Vs0

        self.set_measures(
            domains=domains_mf,
            boundaries=boundaries_mf,
            points=points_mf)

        assert (porosity_known in ("Phis0", "phis"))
        self.set_known_and_unknown_porosity(porosity_known)

        assert (porosity_init_val is None) or (porosity_init_fun is None)
        self.init_known_porosity(
            porosity_init_val=porosity_init_val,
            porosity_init_fun=porosity_init_fun)
        
        self.w_pressure_balancing_gravity = w_pressure_balancing_gravity

        self.set_subsols(
            displacement_perturbation_degree=displacement_perturbation_degree,
            solid_pressure_degree=solid_pressure_degree,
            porosity_degree=porosity_degree,
            porosity_init_val=porosity_init_val,
            porosity_init_fun=porosity_init_fun)
        self.set_solution_finite_element()
        if (bcs == "pbc"):
            periodic_sd = dmech.PeriodicSubDomain(self.dim, self.mesh_bbox, self.vertices)
            self.set_solution_function_space(constrained_domain=periodic_sd)
        else:
            self.set_solution_function_space()
        self.set_solution_functions()

        self.U_bar      = dolfin.dot(self.macroscopic_stretch_subsol.subfunc , self.X-self.X_0)
        self.U_bar_old  = dolfin.dot(self.macroscopic_stretch_subsol.func_old, self.X-self.X_0)
        self.U_bar_test = dolfin.dot(self.macroscopic_stretch_subsol.dsubtest, self.X-self.X_0)

        self.U_tot      = self.U_bar      + self.displacement_perturbation_subsol.subfunc
        self.U_tot_old  = self.U_bar_old  + self.displacement_perturbation_subsol.func_old
        self.U_tot_test = self.U_bar_test + self.displacement_perturbation_subsol.dsubtest

        self.set_quadrature_degree(
            quadrature_degree=quadrature_degree)

        self.set_foi_finite_elements_DG(
            degree=foi_degree)
        self.set_foi_function_spaces()

        self.add_foi(
            expr=self.U_bar,
            fs=self.vfoi_fs,
            name="U_bar",
            update_type="project")
        self.add_foi(
            expr=self.U_tot,
            fs=self.vfoi_fs,
            name="U_tot",
            update_type="project")

        self.set_kinematics()

        assert (skel_behavior is     None) or (len(skel_behaviors)==0),\
            "Cannot provide both skel_behavior & skel_behaviors. Aborting."
        assert (skel_behavior is not None) or (len(skel_behaviors) >0),\
            "Need to provide skel_behavior or skel_behaviors. Aborting."
        if (skel_behavior is not None):
            skel_behaviors = [skel_behavior]
        self.add_Wskel_operators(skel_behaviors)

        assert (bulk_behavior is     None) or (len(bulk_behaviors)==0),\
            "Cannot provide both bulk_behavior & bulk_behaviors. Aborting."
        assert (bulk_behavior is not None) or (len(bulk_behaviors) >0),\
            "Need to provide bulk_behavior or bulk_behaviors. Aborting."
        if (bulk_behavior is not None):
            bulk_behaviors = [bulk_behavior]  
        self.kappa_val = bulk_behaviors[0]["parameters"]["kappa"]   

        # self.add_macroscopic_stretch_symmetry_operator()
        self.add_macroscopic_stretch_symmetry_penalty_operator(pen_val=1e6)

        self.add_pressure_liquid_perturbation_zero_mean_operator()

        if (bcs == "kubc"):
            self.add_kubc()
        elif (bcs == "pbc"):
            pinpoint_sd = dmech.PinpointSubDomain(coords=mesh.coordinates()[-1], tol=1e-3)
            self.add_constraint(
                V=self.displacement_perturbation_subsol.fs, 
                val=[0.]*self.dim,
                sub_domain=pinpoint_sd,
                method='pointwise')


    def set_subsols(self,
            displacement_perturbation_degree=None,
            solid_pressure_degree=None,
            liquid_pressure_perturbation_degree=None,
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None):

        self.add_macroscopic_stretch_subsol(
            symmetry=None) # MG20220425: True does not work, cf. https://fenicsproject.discourse.group/t/writing-symmetric-tensor-function-fails/1136/2 & https://bitbucket.org/fenics-project/dolfin/issues/1065/cannot-store-symmetric-tensor-values

        self.add_displacement_perturbation_subsol(
            degree=displacement_perturbation_degree)
        
        if (self.w_solid_incompressibility):
            if (solid_pressure_degree is None):
                solid_pressure_degree = displacement_perturbation_degree-1
            self.add_pressure_subsol(
                degree=solid_pressure_degree)
        
        # self.add_macroscopic_stress_lagrange_multiplier_subsol()

        self.add_surface_area_subsol()
        
        if (liquid_pressure_perturbation_degree is None):
            liquid_pressure_perturbation_degree = displacement_perturbation_degree -1
        self.add_pressure_liquid_perturbation_subsol(degree=liquid_pressure_perturbation_degree)
        self.add_pressure_liquid_perturbation_zero_mean_subsol()


    def get_pressure_liquid_name(self):
        return "p_l_perturbation"

    def add_pressure_liquid_perturbation_subsol(self, degree):
        self.pl_perturbation_subsol = self.add_scalar_subsol(
            name=self.get_pressure_liquid_name(),
            family="CG",
            degree=degree
        )
    def add_pressure_liquid_perturbation_zero_mean_subsol(self):
        self.lambda_pl_perturbation_zero_mean_subsol = self.add_scalar_subsol(
            name="lambda_"+self.get_pressure_liquid_name()+"_zero_mean",
            family="R",
            degree=0
        )

    def add_pressure_liquid_perturbation_zero_mean_operator(self):
        p      = self.pl_perturbation_subsol.subfunc
        p_test = self.pl_perturbation_subsol.dsubtest


        lam      = self.lambda_pl_perturbation_zero_mean_subsol.subfunc
        lam_test = self.lambda_pl_perturbation_zero_mean_subsol.dsubtest

        operator = dmech.ZeroMeanPressureOperator(
            p=p,
            p_test=p_test,
            lam=lam,
            lam_test=lam_test,
            measure=self.dV
        )
        return self.add_operator(operator)

    def add_Darcy_operator(self,
                        # --- kinematics / fields ---
                        kinematics,
                        U,
                        U_test,
                        X,
                        X_0,

                        # --- macro loads ---
                        grad_p_bar_ini,
                        grad_p_bar_fin,
                        pl_bar_ini,
                        pl_bar_fin,
                        Theta_in_ini,
                        Theta_in_fin,
                        Theta_out_ini,
                        Theta_out_fin,

                        # --- material ---
                        k_l0,               # 2x2 tensor baseline intrinsic permeability (current config)
                        use_kozeny_carman,  # whether to use Kozeny-Carman relative permeability (True) or a constant factor (False)

                        # --- domain / boundary ids ---
                        subdomain_id,
                        inlet_id,
                        outlet_id,

                        # --- step index ---
                        k_step):

        # Pressure perturbation unknown and its test function
        p_tilde = self.pl_perturbation_subsol.subfunc
        p_test  = self.pl_perturbation_subsol.dsubtest

        dx     = self.get_subdomain_measure(subdomain_id)
        dx_in  = self.get_subdomain_measure(inlet_id)   if inlet_id  is not None else None
        dx_out = self.get_subdomain_measure(outlet_id)  if outlet_id is not None else None

        operator = dmech.DarcyFlowOperator(
            kinematics=kinematics,
            U=U,
            U_test=U_test,
            X=X,
            X_0=X_0,
            p_tilde=p_tilde,
            p_test=p_test,
            grad_p_bar_ini=grad_p_bar_ini,
            grad_p_bar_fin=grad_p_bar_fin,
            pl_bar_ini=pl_bar_ini,
            pl_bar_fin=pl_bar_fin,
            use_kozeny_carman=use_kozeny_carman,
            k_l0=k_l0,
            Phis0=self.Phis0,
            kappa_val=self.kappa_val,
            dx=dx,
            dx_in=dx_in,
            dx_out=dx_out,
            Theta_in_ini=Theta_in_ini,
            Theta_in_fin=Theta_in_fin,
            Theta_out_ini=Theta_out_ini,
            Theta_out_fin=Theta_out_fin,
        )

        self.add_foi(expr=operator.Phis_expr,   fs=self.sfoi_fs, name="Phis",   update_type="project")
        self.add_foi(expr=operator.Phif_expr,   fs=self.sfoi_fs, name="Phif",   update_type="project")

        self.add_foi(expr=operator.pl_affine,   fs=self.sfoi_fs, name="pl_affine", update_type="project")
        self.add_foi(expr=operator.pl_bar,      fs=self.sfoi_fs, name="pl_bar",    update_type="project")
        self.add_foi(expr=operator.pl_tot,      fs=self.sfoi_fs, name="pl_tot",    update_type="project")

        # --- tensor fields ---
        # K_l: pull-back permeability used in the reference weak form
        self.add_foi(expr=operator.K_l,       fs=self.mfoi_fs, name="K_l_ref",   update_type="project")

        # k_l_intr: intrinsic permeability tensor in current configuration
        self.add_foi(expr=operator.k_l_intr,  fs=self.mfoi_fs, name="k_l_intr",  update_type="project")

        # --- fluxes (vectors) ---
        # q_l: current Darcy flux, Q_l: Piola/reference flux
        self.add_foi(expr=operator.q_l,       fs=self.vfoi_fs, name="q_l",       update_type="project")
        self.add_foi(expr=operator.Q_l,       fs=self.vfoi_fs, name="Q_l",       update_type="project")

        return self.add_operator(operator=operator, k_step=k_step)
        
    def add_darcy_qois(self, symmetric=False):

        if not self.steps:
            raise RuntimeError("No steps available.")

        step = self.steps[-1]
        darcy_op = None
        for op in step.operators:
            if hasattr(op, "Q_l") and hasattr(op, "grad_p_bar") and hasattr(op, "pl_bar"):
                darcy_op = op
                break
        if darcy_op is None:
            raise RuntimeError("Cannot find Darcy operator in current step.")

        dx = getattr(darcy_op, "dx_measure", None)
        if dx is None:
            raise RuntimeError("Darcy operator has no dx_measure.")

        Area_ref = dolfin.assemble(1.0 * dx)

        # Reference averages over the Darcy subdomain
        # Average over RVE box
        self.add_qoi(name="Q_l_avg_x",        expr=darcy_op.Q_l[0] * dx,       norm=self.V0)
        self.add_qoi(name="Q_l_avg_y",        expr=darcy_op.Q_l[1] * dx,       norm=self.V0)

        # Average over solid skeleton 
        self.add_qoi(name="pl_bar_avg",       expr=darcy_op.pl_bar * dx,       norm=Area_ref)
        self.add_qoi(name="grad_p_bar_avg_x", expr=darcy_op.grad_p_bar[0] * dx, norm=Area_ref)
        self.add_qoi(name="grad_p_bar_avg_y", expr=darcy_op.grad_p_bar[1] * dx, norm=Area_ref)

    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WskelPoroOperator(
            kinematics=self.kinematics,
            U=self.displacement_perturbation_subsol.subfunc,
            U_test=self.displacement_perturbation_subsol.dsubtest,
            Phis0=self.Phis0,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)
