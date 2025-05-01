#coding=utf8

################################################################################
###                                                                          ###
### Created by Haotian XIAO, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class DarcyFlowOperator(Operator):

    def __init__(self,
            p,
            p_test,
            K_l,
            rho_l,
            Theta_0,
            measure):

        self.measure = measure

        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        self.res_form = (
            rho_l * dolfin.dot(K_l * grad_p, grad_p_test) * self.measure
            - Theta_0 * p_test * self.measure
        )