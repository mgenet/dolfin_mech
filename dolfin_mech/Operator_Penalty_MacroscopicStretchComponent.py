#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class MacroscopicStretchComponentPenaltyOperator(Operator):

    def __init__(self,
            U_bar,
            U_bar_test,
            i, j,
            measure,
            U_bar_ij_val=None, U_bar_ij_ini=None, U_bar_ij_fin=None,
            pen_val=None, pen_ini=None, pen_fin=None):

        self.measure = measure

        self.tv_U_bar_ij = dmech.TimeVaryingConstant(
            val=U_bar_ij_val, val_ini=U_bar_ij_ini, val_fin=U_bar_ij_fin)
        U_bar_ij = self.tv_U_bar_ij.val

        self.tv_pen = dmech.TimeVaryingConstant(
            val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
        pen = self.tv_pen.val

        Pi = (pen/2) * (U_bar[i,j] - U_bar_ij)**2 * self.measure
        self.res_form = dolfin.derivative(Pi, U_bar[i,j], U_bar_test[i,j])



    def set_value_at_t_step(self,
            t_step):

        self.tv_U_bar_ij.set_value_at_t_step(t_step)
        self.tv_pen.set_value_at_t_step(t_step)
