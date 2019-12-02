#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin
import numpy
# import math

import dolfin_cm as dcm
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class NewFPoroWporProblem(HyperelasticityProblem):



    def __init__(self,
            eta,
            kappa):

        HyperelasticityProblem.__init__(self,w_incompressibility=False)
        self.eta = eta
        self.kappa = kappa



    def add_porosity_subsol(self,
            degree):

        if (degree == 0):
            self.add_scalar_subsol(
                name="Phi",
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name="Phi",
                family="CG",
                degree=degree)



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        self.add_porosity_subsol(
            degree=U_degree-1)



    def get_porosity_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Phi")



    def set_porosity_energy(self):

        # if self.subsols["Phi"].subfunc < self.phi0:
            # dWpordJ = - self.eta * (self.phi0 / self.subsols["Phi"].subfunc)**2 * exp(-self.subsols["Phi"].subfunc / (self.phi0 - self.subsols["Phi"].subfunc)) / (self.phi0 - self.subsols["Phi"].subfunc)
        # dWpordJ = - self.eta * (self.phi0 / (self.kinematics.Je * self.subsols["Phi"].subfunc))**2 * dolfin.exp(-self.kinematics.Je * self.subsols["Phi"].subfunc / (self.phi0 - self.kinematics.Je * self.subsols["Phi"].subfunc)) / (self.phi0 - self.kinematics.Je * self.subsols["Phi"].subfunc)
        dWpordJ = 0
        # else:
        #     dWpordJ = 0
        self.dWpordJ = (1 - self.phi0) * dWpordJ



    def set_bulk_energy(self):

        dWbulkdJs = self.kappa * (1. / (1. - self.phi0) - 1./self.kinematics.Js)
        # dWbulkdJs = self.kappa * (1. / (1. - self.phi0) - 1./(self.kinematics.Je * (1-self.subsols["Phi"].subfunc)))
        self.dWbulkdJs = (1 - self.phi0) * dWbulkdJs



    def set_phi0(self,
            config_porosity='ref'):

        if self.config_porosity == 'ref':
            coef = self.porosity_given
        elif self.config_porosity == 'deformed':
            coef = Nan

        self.phi0 = coef



    def set_kinematics(self):

        HyperelasticityProblem.set_kinematics(self)

        self.set_phi0(self.config_porosity)

        if self.config_porosity == 'ref':
            self.kinematics.Js = self.kinematics.Je * (1 - self.subsols["Phi"].subfunc)
        else:
            self.kinematics.Js = Nan



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        self.set_kinematics()

        HyperelasticityProblem.set_materials(self,
                elastic_behavior=elastic_behavior,
                elastic_behavior_dev=elastic_behavior_dev,
                elastic_behavior_bulk=elastic_behavior_bulk,
                subdomain_id=subdomain_id)

        self.set_porosity_energy()
        self.set_bulk_energy()



    def set_variational_formulation(self,
            normal_penalties=[],
            directional_penalties=[],
            surface_tensions=[],
            surface0_loadings=[],
            pressure0_loadings=[],
            volume0_loadings=[],
            surface_loadings=[],
            pressure_loadings=[],
            volume_loadings=[],
            dt=None):

        self.Pi = sum([subdomain.Psi * self.dV(subdomain.id) for subdomain in self.subdomains])
        # print (self.Pi)

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test);

        for loading in pressure_loadings:
            T = dolfin.dot(
               -loading.val * self.mesh_normals,
                dolfin.inv(self.kinematics.Ft))
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                T,
                self.subsols["U"].dsubtest) * loading.measure

        self.res_form += dolfin.inner(
            self.dWpordJ * self.kinematics.Je * self.kinematics.Ce_inv,
            dolfin.derivative(
                    self.kinematics.Et,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest)) * self.dV

        self.res_form += dolfin.inner(
                self.dWbulkdJs - self.dWpordJ,
                self.subsols["Phi"].dsubtest) * self.dV

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_Phi_qois(self):

        basename = "PHI_"
        Phi = self.subsols["Phi"].subfunc

        self.add_qoi(
            name=basename,
            expr=Phi / self.mesh_V0 * self.dV)



    def add_Js_qois(self):

        basename = "Js_"

        self.add_qoi(
            name=basename,
            expr=self.kinematics.Js / self.mesh_V0 * self.dV)
