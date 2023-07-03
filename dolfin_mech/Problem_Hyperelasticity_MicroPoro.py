#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2021-2023                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class MicroPoroHyperelasticityProblem(HyperelasticityProblem):



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
            quadrature_degree=None,
            foi_degree=0,
            solid_behavior=None,
            bcs="kubc"): # "kubc" or "pbc"

        HyperelasticityProblem.__init__(self)

        self.w_solid_incompressibility = w_solid_incompressibility

        if (mesh is not None):
            self.set_mesh(
                mesh=mesh,
                define_spatial_coordinates=1,
                define_facet_normals=1,
                compute_bbox=(mesh_bbox is None))
            self.X_0 = dolfin.Constant(tuple(["0."]*self.dim))
            if (mesh_bbox is not None):
                self.mesh_bbox = mesh_bbox
            d = [0]*self.dim
            for k_dim in range(self.dim):
                d[k_dim] = self.mesh_bbox[2*k_dim+1] - self.mesh_bbox[2*k_dim+0]
            self.mesh_bbox_V0 = numpy.prod(d)
            self.Vf0 = self.mesh_bbox_V0 - self.mesh_V0

            self.set_measures(
                domains=domains_mf,
                boundaries=boundaries_mf,
                points=points_mf)

            self.set_subsols(
                displacement_perturbation_degree=displacement_perturbation_degree,
                solid_pressure_degree=solid_pressure_degree)
            self.set_solution_finite_element()
            if (bcs == "pbc"):
                periodic_sd = dmech.PeriodicSubDomain(self.dim, self.mesh_bbox)
                self.set_solution_function_space(constrained_domain=periodic_sd)
            else:
                self.set_solution_function_space()
            self.set_solution_functions()

            self.U_bar      = dolfin.dot(self.get_macroscopic_stretch_subsol().subfunc , self.X-self.X_0)
            self.U_bar_old  = dolfin.dot(self.get_macroscopic_stretch_subsol().func_old, self.X-self.X_0)
            self.U_bar_test = dolfin.dot(self.get_macroscopic_stretch_subsol().dsubtest, self.X-self.X_0)

            self.U_tot      = self.U_bar      + self.get_displacement_perturbation_subsol().subfunc
            self.U_tot_old  = self.U_bar_old  + self.get_displacement_perturbation_subsol().func_old
            self.U_tot_test = self.U_bar_test + self.get_displacement_perturbation_subsol().dsubtest

            self.set_quadrature_degree(
                quadrature_degree=quadrature_degree)

            self.set_foi_finite_elements_DG(
                degree=foi_degree)
            self.set_foi_function_spaces()

            self.add_foi(
                expr=self.U_bar,
                fs=self.get_displacement_perturbation_function_space().collapse(),
                name="U_bar",
                update_type="project")
            self.add_foi(
                expr=self.U_tot,
                fs=self.get_displacement_perturbation_function_space().collapse(),
                name="U_tot",
                update_type="project")

            self.set_kinematics()

            self.add_elasticity_operator(
                solid_behavior_model=solid_behavior["model"],
                solid_behavior_parameters=solid_behavior["parameters"])
            if (self.w_solid_incompressibility):
                self.add_hydrostatic_pressure_operator()
                self.add_incompressibility_operator()

            # self.add_macroscopic_stretch_symmetry_operator()
            self.add_macroscopic_stretch_symmetry_penalty_operator(pen_val=1e6)

            self.add_macrosocpic_stress_operator()
            # self.add_macrosocpic_stress_operator(pen_val=1e6)

            if (bcs == "kubc"):
                self.add_kubc()
            elif (bcs == "pbc"):
                pinpoint_sd = dmech.PinpointSubDomain(coords=mesh.coordinates()[-1], tol=1e-3)
                self.add_constraint(
                    V=self.get_displacement_perturbation_function_space(), 
                    val=[0.]*self.dim,
                    sub_domain=pinpoint_sd,
                    method='pointwise')



    def get_macroscopic_stretch_name(self):

        return "U_bar"


    
    def add_macroscopic_stretch_subsol(self,
            degree=0,
            symmetry=None,
            init_val=None):

        self.add_tensor_subsol(
            name=self.get_macroscopic_stretch_name(),
            family="R",
            degree=degree,
            symmetry=symmetry,
            init_val=init_val)



    def get_macroscopic_stretch_subsol(self):

        return self.get_subsol(self.get_macroscopic_stretch_name())



    def get_macroscopic_stretch_function_space(self):

        return self.get_subsol_function_space(name=self.get_macroscopic_stretch_name())



    def get_displacement_perturbation_name(self):

        return "U_tilde"



    def add_displacement_perturbation_subsol(self,
            degree):

        self.displacement_perturbation_degree = degree
        self.add_vector_subsol(
            name=self.get_displacement_perturbation_name(),
            family="CG",
            degree=self.displacement_perturbation_degree)



    def get_displacement_perturbation_subsol(self):

        return self.get_subsol(self.get_displacement_perturbation_name())



    def get_displacement_perturbation_function_space(self):

        return self.get_subsol_function_space(name=self.get_displacement_perturbation_name())



    def get_solid_pressure_name(self):

        return "p_s"



    def add_solid_pressure_subsol(self,
            degree):

        self.solid_pressure_degree = degree
        if (degree == 0):
            self.add_scalar_subsol(
                name=self.get_solid_pressure_name(),
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name=self.get_solid_pressure_name(),
                family="CG",
                degree=self.solid_pressure_degree)



    def get_solid_pressure_subsol(self):

        assert (self.w_solid_incompressibility),\
            "There is no solid pressure subsol. Aborting."
        return self.get_subsol(self.get_solid_pressure_name())



    def get_solid_pressure_function_space(self):

        assert (self.w_solid_incompressibility),\
            "There is no solid pressure function space. Aborting."
        return self.get_subsol_function_space(name=self.get_solid_pressure_name())



    def get_macroscopic_stress_name(self):

        return "sigma_bar"



    def add_macroscopic_stress_subsol(self,
            degree=0,
            symmetry=None,
            init_val=None):

        self.add_tensor_subsol(
            name=self.get_macroscopic_stress_name(),
            family="R",
            degree=degree,
            symmetry=symmetry,
            init_val=init_val)



    def get_macroscopic_stress_subsol(self):

        return self.get_subsol(self.get_macroscopic_stress_name())



    def get_macroscopic_stress_function_space(self):

        return self.get_subsol_function_space(name=self.get_macroscopic_stress_name())



    def set_subsols(self,
            displacement_perturbation_degree=None,
            solid_pressure_degree=None):

        self.add_macroscopic_stretch_subsol(
            symmetry=None) # MG20220425: True does not work, cf. https://fenicsproject.discourse.group/t/writing-symmetric-tensor-function-fails/1136/2 & https://bitbucket.org/fenics-project/dolfin/issues/1065/cannot-store-symmetric-tensor-values

        self.add_displacement_perturbation_subsol(
            degree=displacement_perturbation_degree)

        if (self.w_solid_incompressibility):
            if (solid_pressure_degree is None):
                solid_pressure_degree = displacement_perturbation_degree-1
            self.add_pressure_subsol(
                degree=solid_pressure_degree)

        self.add_macroscopic_stress_subsol()



    def set_kinematics(self):

        self.kinematics = dmech.Kinematics(
            U=self.U_tot,
            U_old=self.U_tot_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F_tot", update_type="project")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J_tot", update_type="project")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C_tot", update_type="project")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E_tot", update_type="project")



    def add_elasticity_operator(self,
            solid_behavior_model,
            solid_behavior_parameters):

        operator = dmech.HyperElasticityOperator(
            U=self.sol_func,
            U_test=self.dsol_test,
            kinematics=self.kinematics,
            material_model=solid_behavior_model,
            material_parameters=solid_behavior_parameters,
            measure=self.dV,
            formulation="ener")
        self.add_foi(expr=operator.material.Sigma, fs=self.mfoi_fs, name="Sigma", update_type="project")
        self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma", update_type="project")
        return self.add_operator(operator)



    def add_macroscopic_stretch_symmetry_operator(self):

        operator = dmech.TensorSymmetryOperator(
            tensor=self.get_macroscopic_stretch_subsol().subfunc,
            tensor_test=self.get_macroscopic_stretch_subsol().dsubtest,
            measure=self.dV)
        return self.add_operator(operator)



    def add_macroscopic_stretch_symmetry_penalty_operator(self,
            **kwargs):

        operator = dmech.MacroscopicStretchSymmetryPenaltyOperator(
            U_bar=self.get_macroscopic_stretch_subsol().subfunc,
            U_bar_test=self.get_macroscopic_stretch_subsol().dsubtest,
            sol=self.sol_func,
            sol_test=self.dsol_test,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator)



    def add_macroscopic_stretch_component_penalty_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.MacroscopicStretchComponentPenaltyOperator(
            U_bar=self.get_macroscopic_stretch_subsol().subfunc,
            U_bar_test=self.get_macroscopic_stretch_subsol().dsubtest,
            sol=self.sol_func,
            sol_test=self.dsol_test,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator, k_step=k_step)



    def add_macrosocpic_stress_operator(self,
            **kwargs):

        for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
            if hasattr(operator, "material"):
                material = operator.material
                break

        operator = dmech.MacroscopicStressOperator(
            mesh_V0=self.mesh_V0,
            mesh_bbox_V0=self.mesh_bbox_V0,
            sigma_bar=self.get_macroscopic_stress_subsol().subfunc,
            sigma_bar_test=self.get_macroscopic_stress_subsol().dsubtest,
            sol=self.sol_func,
            sol_test=self.dsol_test,
            material=material,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator)



    def add_macroscopic_stress_component_penalty_operator(self,
            k_step=None,
            **kwargs):

        for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
            if hasattr(operator, "material"):
                material = operator.material
                break

        operator = dmech.MacroscopicStressComponentPenaltyOperator(
            sigma_bar=self.get_macroscopic_stress_subsol().subfunc,
            sigma_bar_test=self.get_macroscopic_stress_subsol().dsubtest,
            sol=self.sol_func,
            sol_test=self.dsol_test,
            material=material,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator, k_step=k_step)



    def add_surface_pressure_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.SurfacePressureLoadingOperator(
            U_test=self.U_tot_test,
            kinematics=self.kinematics,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)



    def add_kubc(self,
            xmin_id=1, xmax_id=2,
            ymin_id=3, ymax_id=4,
            zmin_id=5, zmax_id=6):

        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(0),
            sub_domains=self.boundaries,
            sub_domain_id=xmin_id,
            val=0.)
        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(0),
            sub_domains=self.boundaries,
            sub_domain_id=xmax_id,
            val=0.)
        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(1),
            sub_domains=self.boundaries,
            sub_domain_id=ymin_id,
            val=0.)
        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(1),
            sub_domains=self.boundaries,
            sub_domain_id=ymax_id,
            val=0.)
        if (self.dim==3):
            self.add_constraint(
                V=self.get_displacement_perturbation_function_space().sub(2),
                sub_domains=self.boundaries,
                sub_domain_id=zmin_id,
                val=0.)
            self.add_constraint(
                V=self.get_displacement_perturbation_function_space().sub(2),
                sub_domains=self.boundaries,
                sub_domain_id=zmax_id,
                val=0.)



    def add_macroscopic_tensor_qois(self,
            basename,
            get_subsol,
            symmetric=False):

        self.add_qoi(
            name=basename+"_XX",
            expr=get_subsol().subfunc[0,0],
            point=self.mesh.coordinates()[0],
            update_type="direct")
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"_YY",
                expr=get_subsol().subfunc[1,1],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"_ZZ",
                    expr=get_subsol().subfunc[2,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"_XY",
                expr=get_subsol().subfunc[0,1],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if not (symmetric): self.add_qoi(
                name=basename+"_YX",
                expr=get_subsol().subfunc[1,0],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"_YZ",
                    expr=get_subsol().subfunc[1,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                if not (symmetric): self.add_qoi(
                    name=basename+"_ZY",
                    expr=get_subsol().subfunc[2,1],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                self.add_qoi(
                    name=basename+"_ZX",
                    expr=get_subsol().subfunc[2,0],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                if not (symmetric): self.add_qoi(
                    name=basename+"_XZ",
                    expr=get_subsol().subfunc[0,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")



    def add_macroscopic_stretch_qois(self):
        self.add_macroscopic_tensor_qois(
            basename="U_bar",
            get_subsol=self.get_macroscopic_stretch_subsol)



    def add_macroscopic_stress_qois(self):
        self.add_macroscopic_tensor_qois(
            basename="sigma_bar",
            get_subsol=self.get_macroscopic_stress_subsol)
