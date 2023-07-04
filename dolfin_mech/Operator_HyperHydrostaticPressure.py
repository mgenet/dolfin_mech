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

class HyperHydrostaticPressureOperator(Operator):

    def __init__(self,
            kinematics,
            U_test,
            P,
            measure):

        self.kinematics = kinematics
        self.P          = P
        self.measure    = measure

        dJ_test = dolfin.derivative(
            self.kinematics.J,
            self.kinematics.U,
            U_test)

        self.res_form = - self.P * dJ_test * self.measure
