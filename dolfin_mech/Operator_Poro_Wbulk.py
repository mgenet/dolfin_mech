#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class WbulkPoroOperator(Operator):

    def __init__(self,
            kinematics,
            U_test,
            Phis0,
            Phis,
            Phis_test,
            material_parameters,
            material_scaling,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WbulkLungElasticMaterial(
            Phis=Phis,
            Phis0=Phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        dE_test = dolfin.derivative(
            self.kinematics.E, self.kinematics.U, U_test)
        self.res_form = dolfin.inner(
            self.material.dWbulkdPhis * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure

################################################################################

class InverseWbulkPoroOperator(Operator):

    def __init__(self,
            kinematics,
            u_test,
            phis,
            phis0,
            phis0_test,
            material_parameters,
            material_scaling,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WbulkLungElasticMaterial(
            Phis=self.kinematics.J * phis,
            Phis0=self.kinematics.J * phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=self.kinematics.J * phis0)
        self.measure = measure

        epsilon_test = dolfin.sym(dolfin.grad(u_test))
        self.res_form = self.material.dWbulkdPhis * dolfin.tr(epsilon_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * phis0_test * self.measure
