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

class LinearizedHydrostaticPressureOperator(Operator):

    def __init__(self,
            kinematics,
            u_test,
            p,
            measure):

        self.kinematics = kinematics
        self.p          = p
        self.measure    = measure

        epsilon_test = dolfin.sym(dolfin.grad(u_test))
        self.res_form = -self.p * dolfin.tr(epsilon_test) * self.measure
