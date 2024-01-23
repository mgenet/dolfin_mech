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

class LinearizedElasticityOperator(Operator):

    def __init__(self,
            kinematics,
            u_test,
            material_model,
            material_parameters,
            measure):

        self.kinematics = kinematics
        self.material   = dmech.material_factory(kinematics, material_model, material_parameters)
        self.measure    = measure

        epsilon_test = dolfin.sym(dolfin.grad(u_test))
        self.res_form = dolfin.inner(self.material.sigma, epsilon_test) * self.measure
