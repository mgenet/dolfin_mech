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

class HyperIncompressibilityOperator(Operator):

    def __init__(self,
            kinematics,
            P_test,
            measure):

        self.kinematics = kinematics
        self.measure    = measure

        self.res_form = - (self.kinematics.J - 1) * P_test * self.measure
