#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem_Hyperelasticity_Poro import PoroHyperelasticityProblem

################################################################################

class InversePoroHyperelasticityProblem(PoroHyperelasticityProblem):



    def __init__(self, *args, **kwargs):

        PoroHyperelasticityProblem.__init__(self, *args, **kwargs)



    def set_known_and_unknown_porosity(self,
            porosity_known):
        
        self.porosity_known = porosity_known
        if (self.porosity_known == "phis"):
            self.porosity_unknown = "phis0"
        elif (self.porosity_known == "Phis0"):
            self.porosity_unknown = "phis"



    def get_deformed_center_of_mass(self):
        
        M = dolfin.assemble(getattr(self, self.porosity_known)*self.dV)
        center_of_mass = numpy.empty(self.dim)
        for k_dim in range(self.dim):
            center_of_mass[k_dim] = dolfin.assemble(getattr(self, self.porosity_known)*self.x[k_dim]*self.dV)/M
        return center_of_mass



    def set_kinematics(self):

        self.kinematics = dmech.InverseKinematics(
            u=self.displacement_subsol.subfunc,
            u_old=self.displacement_subsol.func_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")



    def set_porosity_fields(self):

        if (self.porosity_known == "phis"):
            self.phis0 = self.porosity_subsol.subfunc
            self.Phis0 = self.phis0*self.kinematics.J
        elif (self.porosity_known == "Phis0"):
            self.phis = self.porosity_subsol.subfunc
            self.phis0 = self.Phis0/self.kinematics.J



    def add_local_porosity_fois(self):

        if (self.porosity_known == "phis"): self.add_foi(
            expr=self.phis,
            fs=self.porosity_subsol.fs.collapse(),
            name="phis")
        self.add_foi(
            expr=1. - self.phis,
            fs=self.porosity_subsol.fs.collapse(),
            name="phif")

        if (self.porosity_known == "Phis0"): self.add_foi(
            expr=self.phis0,
            fs=self.porosity_subsol.fs.collapse(),
            name="phis0")
        self.add_foi(
            expr=1/self.kinematics.J - self.phis0,
            fs=self.porosity_subsol.fs.collapse(),
            name="phif0")

        self.add_foi(
            expr=self.kinematics.J * self.phis0, # MG20250908: Todo: check!
            fs=self.porosity_subsol.fs.collapse(),
            name="Phis0")
        self.add_foi(
            expr=1. - self.kinematics.J * self.phis0, # MG20250908: Todo: check!
            fs=self.porosity_subsol.fs.collapse(),
            name="Phif0")



    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.InverseWskelPoroOperator(
            kinematics=self.kinematics,
            u_test=self.displacement_subsol.dsubtest,
            phis0=self.phis0,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wbulk_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.InverseWbulkPoroOperator(
            kinematics=self.kinematics,
            u_test=self.displacement_subsol.dsubtest,
            phis=self.phis,
            phis0=self.phis0,
            unknown_porosity_test=self.porosity_subsol.dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wpore_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.InverseWporePoroOperator(
            kinematics=self.kinematics,
            phis=self.phis,
            phis0=self.phis0,
            unknown_porosity_test=self.porosity_subsol.dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_pressure_balancing_gravity0_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PressureBalancingGravity0LoadingOperator(
            x = self.x,
            x0 = self.deformed_center_of_mass_subsol.subfunc,
            x0_test = self.deformed_center_of_mass_subsol.dsubtest,
            n = self.mesh_normals,
            u_test = self.displacement_subsol.dsubtest, 
            lmbda = self.lmbda_subsol.subfunc,
            lmbda_test = self.lmbda_subsol.dsubtest,
            p = self.pressure_balancing_gravity_subsol.subfunc,
            p_test = self.pressure_balancing_gravity_subsol.dsubtest,
            gamma = self.gamma_subsol.subfunc,
            gamma_test = self.gamma_subsol.dsubtest,
            mu = self.mu_subsol.subfunc,
            mu_test= self.mu_subsol.dsubtest,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)



    def add_global_porosity_qois(self):
            
        self.add_qoi(
            name="phis",
            expr=self.phis * self.dV)
            
        self.add_qoi(
            name="phif",
            expr=(1. - self.phis) * self.dV)
            
        self.add_qoi(
            name="phis0",
            expr=self.phis0 * self.dV)
            
        self.add_qoi(
            name="phif0",
            expr=(1/self.kinematics.J - self.phis0) * self.dV)

        self.add_qoi(
            name="Phis0",
            expr=(self.kinematics.J * self.phis0) * self.dV)

        self.add_qoi(
            name="Phif0",
            expr=(1 - self.kinematics.J * self.phis0) * self.dV)
