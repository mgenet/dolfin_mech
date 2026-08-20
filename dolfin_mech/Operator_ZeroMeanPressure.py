#coding=utf8

################################################################################
###                                                                          ###
### Created by Haotian Xiao, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class ZeroMeanPressureOperator(Operator):

    def __init__(self,
            p, p_test,
            lam, lam_test,
            measure):

        self.measure = measure
        V0 = dolfin.assemble(1.0 * measure)

        self.res_form  = lam_test * (p      / V0) * self.measure
        self.res_form += lam      * (p_test / V0) * self.measure
