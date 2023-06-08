#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class WporePoroOperator(Operator):

    def __init__(self,
            kinematics,
            Phis0,
            Phis,
            Phis_test,
            material_parameters,
            material_scaling,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WporeLungElasticMaterial(
            Phif=self.kinematics.J - Phis,
            Phif0=1-Phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        self.res_form = - self.material.dWporedPhif * Phis_test * self.measure

################################################################################

class InverseWporePoroOperator(Operator):

    def __init__(self,
            kinematics,
            phis,
            phis0,
            phis0_test,
            material_parameters,
            material_scaling,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WporeLungElasticMaterial(
            Phif=self.kinematics.J * (1 - phis),
            Phif0=1 - self.kinematics.J * phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=self.kinematics.J * phis0)
        self.measure = measure

        self.res_form = - self.material.dWporedPhif * phis0_test * self.measure
