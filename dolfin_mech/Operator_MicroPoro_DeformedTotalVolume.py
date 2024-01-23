#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2024                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

# ################################################################################

class DeformedTotalVolumeOperator(Operator):

    def __init__(self,
            v,
            v_test,
            U_bar,
            V0,
            Vs0,
            measure):

        self.measure = measure

        dim = U_bar.ufl_shape[0]
        F_bar = dolfin.Identity(dim) + U_bar
        J_bar = dolfin.det(F_bar)
        
        self.res_form = ((v - J_bar*V0) * v_test)/Vs0 * self.measure
